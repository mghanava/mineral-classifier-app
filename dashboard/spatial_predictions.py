"""Bind producer-order prediction rows to a freshly verified incoming graph."""

import csv
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from io import BytesIO, StringIO
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from dashboard.artifacts import Workspace
from dashboard.dvc_evidence import WorkspaceStamp, plain_path
from dashboard.provenance import (
    AppliedConfiguration,
    Availability,
    Correspondence,
    Evidence,
    check_dvc_stages,
    producer_scope,
    stage_paths,
)
from dashboard.scientific_summary import ScientificReader
from dashboard.spatial_data import (
    ArraySnapshot,
    PopulationKind,
    SpatialPopulation,
    SpatialView,
    make_population,
)
from scripts.setup_dvc import pipeline_definition
from src.configuration_files import configuration_operation

# The CSV has no explicit row key. Require the source implementing the reviewed
# positional contract even if locally customized producer code has its own lock.
_ORDER_SOURCES = {
    "src/stages/predict.py": "f4269924a538563026517c60303dfb4074f4d408d5b10e692ef938ac2349c0dd",
    "src/models/__init__.py": "8e5ad25eeedb52216174132b04c78d637dc665eb82798c902fa91218965cdb37",
    "src/models/gat.py": "4274bbb7740a7c31da05a259732d77c9b69644f790f4720c4660532c4a42e0bd",
    "src/models/gcn.py": "48bc2c443285e4db334330384f757c147bfec8c0f626620c4f380a553703593e",
    "src/utilities/pred_utils.py": "8f442d5579c2981d1168458462a8c9e79200a387bfbf617f3111cfdcacac6f05",
    "src/utilities/raw_graphs.py": "e1e99be3f19d48779b68f58d1b2cfa4094ced7b7eb2b08aee20855f892d99124",
    "src/utilities/preprocessing.py": "1105335cc8df33e5762acc010feec52b5f27ac17c4e7abb74421db22c968bb3d",
    "src/utilities/calibration_utils.py": "3dd7ddf64a037ae80e4af0e48b0b9ccbbe6d9211daca051a7e50c377279cd108",
    "src/utilities/general_utils.py": "2e020b69c0bd505b9240a716a3cd9aa3de70143669502fb33636e5e21eeb3df8",
}
# Float32 softmax/normalization, CSV decimal roundtrip and reduction roundoff.
_FLOAT32_TOL = 8 * np.finfo(np.float32).eps


class ProbabilityFamily(StrEnum):
    """Identify the score family used for all derived display fields."""

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True)
class ProbabilityRows:
    """Keep verified model probabilities and the persisted entropy in row order."""

    _probabilities: ArraySnapshot
    _entropy_nats: ArraySnapshot

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Return the producer probabilities in verified class order."""
        return self._probabilities.array()

    @property
    def entropy_nats(self) -> NDArray[np.float64]:
        """Return persisted entropy with the producer's natural-log convention."""
        return self._entropy_nats.array()

    def predicted_class_ids(self) -> NDArray[np.int64]:
        """Derive first-maximum class indices using the producer tie convention."""
        return self.probabilities.argmax(axis=1)

    def maximum_probability(self) -> NDArray[np.float64]:
        """Derive each row's largest probability."""
        return self.probabilities.max(axis=1)

    def selected_probability(self, class_id: int) -> NDArray[np.float64]:
        """Return probabilities for one verified class index."""
        values = self.probabilities
        if (
            not isinstance(class_id, Integral)
            or isinstance(class_id, bool)
            or not 0 <= class_id < values.shape[1]
        ):
            raise ValueError(
                "Choose a class index from the verified probability mapping."
            )
        return values[:, class_id]


def _probability_rows(
    probabilities: NDArray, entropy: NDArray, family: ProbabilityFamily
) -> ProbabilityRows:
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError(f"{family} probabilities must be finite and within [0, 1].")
    if np.any(np.abs(probabilities.sum(axis=1) - 1) > _FLOAT32_TOL):
        raise ValueError(f"{family} probability rows do not sum to one.")
    expected = -(probabilities * np.log(probabilities + 1e-8)).sum(axis=1)
    entropy_tolerance = _FLOAT32_TOL * (1 + np.log(probabilities.shape[1]))
    if not np.isfinite(entropy).all() or np.any(
        np.abs(entropy - expected) > entropy_tolerance
    ):
        raise ValueError(
            f"{family} entropy differs from the stored probabilities in nats."
        )
    return ProbabilityRows(
        ArraySnapshot.freeze(probabilities), ArraySnapshot.freeze(entropy)
    )


def _read_scores(
    payload: bytes, population: SpatialPopulation
) -> tuple[ProbabilityRows, ProbabilityRows]:
    count = len(population.class_names)
    reader = csv.reader(StringIO(payload.decode("utf-8"), newline=""), strict=True)
    headers = [
        *(f"calib_prob_class_{i}" for i in range(count)),
        *(f"uncalib_prob_class_{i}" for i in range(count)),
        "predicted_label",
        "true_label",
        "calibrated_entropy",
        "uncalibrated_entropy",
    ]
    if next(reader, None) != headers:
        raise ValueError(
            "Prediction CSV headers differ from the producer's ordered class schema."
        )
    rows = list(reader)
    if len(rows) != population.count:
        raise ValueError("Prediction CSV row count differs from the incoming graph.")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError(
            "Prediction CSV contains a row with an unexpected number of fields."
        )
    try:
        values = np.array(rows, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            "Prediction CSV contains a missing or invalid number."
        ) from exc
    predicted, observed, cal_entropy, uncal_entropy = values[:, 2 * count :].T
    if not np.array_equal(observed, population.observed_class_ids):
        raise ValueError(
            "Prediction CSV observed labels differ from the graph's ordered labels."
        )
    calibrated = _probability_rows(
        values[:, :count], cal_entropy, ProbabilityFamily.CALIBRATED
    )
    uncalibrated = _probability_rows(
        values[:, count : 2 * count], uncal_entropy, ProbabilityFamily.UNCALIBRATED
    )
    if not np.array_equal(predicted, calibrated.predicted_class_ids()):
        raise ValueError(
            "Prediction CSV calibrated predicted labels disagree with argmax."
        )
    return calibrated, uncalibrated


@dataclass(frozen=True)
class JoinedPredictions:
    """Point-in-time immutable predictions and their verified observation rows."""

    population: SpatialPopulation
    calibrated: ProbabilityRows
    uncalibrated: ProbabilityRows
    cycle: int
    graph_file_sha256: str
    producer_csv_sha256: str
    _root: Path
    _paths: tuple[str, ...]
    _stamp: WorkspaceStamp

    def still_observed(self, workspace: Workspace) -> bool:
        """Check whether the workspace still corresponds to this historical snapshot."""
        try:
            return (
                workspace.root.resolve() == self._root
                and self._stamp == WorkspaceStamp.capture(self._root, self._paths)
            )
        except (OSError, ValueError, RuntimeError):
            return False

    def scores(self, family: ProbabilityFamily) -> ProbabilityRows:
        """Return the selected verified probability family."""
        return (
            self.calibrated
            if ProbabilityFamily(family) == ProbabilityFamily.CALIBRATED
            else self.uncalibrated
        )

    def predicted_class_names(self, family: ProbabilityFamily) -> tuple[str, ...]:
        """Resolve predicted classes to their verified class names."""
        return tuple(
            self.population.class_names[index]
            for index in self.scores(family).predicted_class_ids()
        )

    def correctness(self, family: ProbabilityFamily) -> NDArray[np.bool_] | None:
        """Compare predictions against the verified observed classes when present."""
        labels = self.population.observed_class_ids
        return (
            self.scores(family).predicted_class_ids() == labels
            if labels is not None
            else None
        )

    def enriched_table(self, view: SpatialView | None = None) -> pd.DataFrame:
        """Return a detached presentation table with explicit export scope and schema."""
        population = self.population
        if view is not None and view.population is not population:
            raise ValueError("The view belongs to a different verified population.")
        columns = {
            "sample_id": population.sample_ids,
            **dict(zip(("x", "y", "z"), population.coordinates.T, strict=True)),
            "observed_class_id": population.observed_class_ids,
            "observed_class_name": population.observed_class_names,
            "membership": population.membership,
            "learning_cycle": self.cycle,
            "raw_schema_version": population.schema_version,
            "class_names_in_probability_order": json.dumps(population.class_names),
            "input_base_sha256": population.parent_sha256,
            "incoming_dataset_sha256": population.dataset_sha256,
            "incoming_graph_file_sha256": self.graph_file_sha256,
            "producer_csv_sha256": self.producer_csv_sha256,
            "population_count": population.count,
            "export_scope": "filtered_view"
            if view is not None and view.filtered
            else "full_population",
        }
        for family in ProbabilityFamily:
            scores = self.scores(family)
            columns[f"{family}_predicted_class_id"] = scores.predicted_class_ids()
            columns[f"{family}_predicted_class_name"] = self.predicted_class_names(
                family
            )
            columns[f"{family}_entropy_nats"] = scores.entropy_nats
            for index, probabilities in enumerate(scores.probabilities.T):
                columns[f"{family}_probability_class_{index}"] = probabilities
        table = pd.DataFrame(columns, copy=True)
        if view is not None:
            table = table.iloc[view.row_indices].reset_index(drop=True)
        return table

    def export(self, view: SpatialView | None = None) -> "PredictionExport":
        """Serialize an enriched full or filtered table in memory, including empty views."""
        table = self.enriched_table(view)
        return PredictionExport(
            self.cycle,
            "filtered_view"
            if view is not None and view.filtered
            else "full_population",
            len(table),
            self.population.count,
            table.to_csv(index=False).encode("utf-8"),
        )


@dataclass(frozen=True)
class PredictionExport:
    """Carry immutable CSV bytes and scope, including for header-only exports."""

    cycle: int
    scope: str
    row_count: int
    population_count: int
    csv_bytes: bytes


@dataclass(frozen=True)
class PredictionJoin:
    """Report scoped evidence and withhold the overlay when any check fails."""

    evidence: Evidence
    joined: JoinedPredictions | None = None


def _prediction_paths(cycle: int) -> tuple[str, str]:
    return (
        f"data/prediction/cycle_{cycle}/pred_data.pt",
        f"prediction/cycle_{cycle}/predictions.csv",
    )


def _recheck(root: Path, paths: tuple[str, ...], stamp: WorkspaceStamp) -> None:
    if stamp != WorkspaceStamp.capture(root, paths):
        raise ValueError(
            "Workspace files changed during prediction verification. Try again."
        )


def _join_in_workspace(root: Path, cycle: int) -> PredictionJoin:
    controls = WorkspaceStamp.capture(root, ()).controls
    if controls[0] is None:
        raise ValueError("An applied configuration is required.")
    applied, settings = AppliedConfiguration.read(controls[0])
    if type(cycle) is not int or not 1 <= cycle <= applied.cycles:
        raise ValueError(
            "Select a learning cycle declared in the applied configuration."
        )
    pipeline = pipeline_definition(settings)["stages"]
    name = f"predict_cycle_{cycle}"
    scope = producer_scope(pipeline, name)
    paths = stage_paths(scope)
    stamp = WorkspaceStamp.capture(root, paths)
    if stamp.controls != controls:
        raise ValueError(
            "Workspace controls changed during prediction verification. Try again."
        )
    files = graph_path, csv_path = tuple(
        f"results/{path}" for path in _prediction_paths(cycle)
    )
    snapshots = {
        path: file.read_bytes()
        for path in files
        if (file := plain_path(root, path)).is_file()
    }
    stages, problems = check_dvc_stages(
        root, settings, pipeline, scope, controls, snapshots=snapshots
    )
    _recheck(root, paths, stamp)
    prediction = stages[name]
    failures = [
        stage
        for stage in stages.values()
        if stage.correspondence != Correspondence.MATCHING
    ]
    if problems or failures or snapshots.keys() != set(files):
        reasons = tuple(
            dict.fromkeys(
                (*problems, *(reason for stage in failures for reason in stage.reasons))
            )
        )
        state = (
            Correspondence.CONFLICT
            if any(
                stage.correspondence == Correspondence.CONFLICT for stage in failures
            )
            else Correspondence.UNKNOWN
        )
        if snapshots.keys() != set(files):
            reasons += (
                "The incoming graph and original prediction CSV must both be present.",
            )
            prediction = replace(
                prediction,
                availability=Availability.PARTIAL
                if snapshots
                else Availability.MISSING,
            )
        return PredictionJoin(
            replace(prediction, correspondence=state, reasons=reasons)
        )

    try:
        for path, digest in _ORDER_SOURCES.items():
            if sha256(plain_path(root, path).read_bytes()).hexdigest() != digest:
                raise ValueError(
                    f"Producer row order has not been verified for this version of {path}."
                )
        reader = ScientificReader(root, settings)
        base = reader.base(cycle - 1)
        _, bundle, _ = reader.bound_model(cycle, base)
        reader.calibration(cycle, base, bundle)
        # Trusted-local PyG serialization, only after DVC checked these exact graph bytes.
        graph = torch.load(
            BytesIO(snapshots[graph_path]), weights_only=False, map_location="cpu"
        )
        raw = reader.batch_graph(graph, base.raw)
        population = make_population(
            PopulationKind.INCOMING,
            raw,
            graph,
            cycle=cycle,
            parent_sha256=base.raw.dataset_sha256,
        )
        calibrated, uncalibrated = _read_scores(snapshots[csv_path], population)
    except Exception as error:
        _recheck(root, paths, stamp)
        return PredictionJoin(
            replace(
                prediction,
                correspondence=Correspondence.CONFLICT,
                reasons=(f"Prediction alignment could not be verified: {error}",),
            )
        )
    _recheck(root, paths, stamp)
    return PredictionJoin(
        prediction,
        JoinedPredictions(
            population,
            calibrated,
            uncalibrated,
            cycle,
            sha256(snapshots[graph_path]).hexdigest(),
            sha256(snapshots[csv_path]).hexdigest(),
            root,
            paths,
            stamp,
        ),
    )


def verify_prediction_join(workspace: Workspace, cycle: int) -> PredictionJoin:
    """Freshly verify a selected prediction scope and bind only the captured bytes."""
    try:
        root = workspace.root.resolve()
        with configuration_operation(root / "params.yaml"):
            return _join_in_workspace(root, cycle)
    except Exception as error:
        availability = Availability.NOT_APPLICABLE
        if type(cycle) is int and cycle >= 1:
            present = sum(
                workspace.available(Path(path)) for path in _prediction_paths(cycle)
            )
            availability = (
                Availability.MISSING,
                Availability.PARTIAL,
                Availability.AVAILABLE,
            )[present]
        return PredictionJoin(
            Evidence(
                availability,
                reasons=(f"Prediction verification is unavailable: {error}",),
            )
        )
