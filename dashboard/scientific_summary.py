"""Read small scientific summaries and check bindings without loading weights."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from torch_geometric.data import Data

from dashboard.dvc_evidence import plain_path, same_value
from src.utilities.general_utils import load_data
from src.utilities.graph_protocol import (
    model_graph_settings,
    resolve_graph_protocol,
    verify_graph,
)
from src.utilities.model_bundle import read_model_bundle, verified_calibrator
from src.utilities.preprocessing import RawDataset, TrainingPartition
from src.utilities.raw_graphs import (
    build_development_graph,
    load_cycle_data,
    read_raw_graph,
    synthetic_schema,
)
from src.utilities.reproducibility import SEED_SCHEME, derive_seed


@dataclass(frozen=True)
class ClassificationMetrics:
    """Class-balanced sample-weighted classification metrics for one population."""

    accuracy: float | None
    macro_f05: float | None
    mcc: float | None


@dataclass(frozen=True)
class TestMetrics:
    """Final-test classification and probability calibration metrics."""

    classification: ClassificationMetrics
    ece: float | None
    mce: float | None


@dataclass(frozen=True)
class EvaluationMetrics:
    """Final-test metrics from the same population before and after calibration."""

    before_calibration: TestMetrics
    after_calibration: TestMetrics


@dataclass(frozen=True)
class BasePopulation:
    """Realized counts from the checked raw base and its held-out partitions."""

    snapshot: int
    total: int
    development: int
    calibration: int
    final_test: int


@dataclass(frozen=True)
class ModelMetadata:
    """Selected metadata from the bound full-development model."""

    model: str
    scaler: str | None
    development_count: int
    weights_sha256: str
    training_graph_sha256: str
    effective_device: str | None


@dataclass(frozen=True)
class CalibrationMetadata:
    """Calibration method from tracked settings and the verified fitting partition."""

    method: str
    split_sha256: str
    calibrator_sha256: str


@dataclass(frozen=True)
class CycleSummary:
    """Describe realized cohorts and small metrics for one learning cycle."""

    input_base: BasePopulation | None = None
    incoming_batch: int | None = None
    post_append: BasePopulation | None = None
    evaluation: EvaluationMetrics | None = None
    prediction: ClassificationMetrics | None = None
    model: ModelMetadata | None = None
    calibration: CalibrationMetadata | None = None


def read_json(path: Path) -> dict:
    """Read an object from a small local scientific metadata file."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return value


def _number(value: object, lower: float = 0, upper: float = 1) -> float | None:
    if type(value) in (int, float) and lower <= value <= upper and math.isfinite(value):
        return float(value)
    return None


def classification_metrics(data: object) -> ClassificationMetrics:
    """Read classification values and expose missing or invalid fields as absent."""
    values = data if isinstance(data, dict) else {}
    return ClassificationMetrics(
        _number(values.get("acc")),
        _number(values.get("f1")),
        _number(values.get("mcc"), -1),
    )


def evaluation_metrics(data: dict) -> EvaluationMetrics:
    """Read the two final-test states and keep probability metrics scoped to them."""

    def read(key: str) -> TestMetrics:
        values = data.get(key)
        if not isinstance(values, dict):
            values = {}
        return TestMetrics(
            classification_metrics(values),
            _number(values.get("ece")),
            _number(values.get("mce")),
        )

    return EvaluationMetrics(read("uncalibrated"), read("calibrated"))


def complete_metrics(metrics: ClassificationMetrics | EvaluationMetrics) -> bool:
    """Require every published metric needed by the matching summary."""
    if isinstance(metrics, ClassificationMetrics):
        return None not in (metrics.accuracy, metrics.macro_f05, metrics.mcc)
    return all(
        complete_metrics(item.classification)
        and item.ece is not None
        and item.mce is not None
        for item in (metrics.before_calibration, metrics.after_calibration)
    )


class BaseData(NamedTuple):
    """Verified partition payloads used during a single sequential refresh."""

    population: BasePopulation
    raw: RawDataset
    folds: list[tuple[Data, Data]]
    test: Data
    calibration: Data

    def development_rows(self) -> NDArray[np.int64]:
        """Return the verified development membership in parent order."""
        held_out = np.concatenate(
            (
                self.test.original_node_indices.numpy(),
                self.calibration.original_node_indices.numpy(),
            )
        )
        return np.setdiff1d(np.arange(self.population.total), held_out)


class ScientificReader:
    """Check the existing graph and bundle contracts for locally hash-matched outputs."""

    def __init__(self, root: Path, settings: dict) -> None:
        """Bind a reader to one applied configuration snapshot."""
        self.root = root
        self.settings = settings
        self.protocol = resolve_graph_protocol(settings)
        self.schema = synthetic_schema(settings["data"]["base"])

    def path(self, relative: str) -> Path:
        """Resolve a trusted path within the workspace."""
        return plain_path(self.root, f"results/{relative}")

    def base(self, snapshot: int) -> BaseData:
        """Verify a base and split membership and report realized population counts."""
        raw, folds, test, calibration = load_cycle_data(
            self.path(f"data/base/cycle_{snapshot}"), self.protocol
        )
        if (raw.feature_names, raw.class_names) != self.schema or len(
            folds
        ) != self.settings["data"]["base"]["n_splits"]:
            raise ValueError(
                "Base schema or fold count differs from the applied configuration."
            )
        total = len(raw.sample_ids)
        population = BasePopulation(
            snapshot,
            total,
            total - test.num_nodes - calibration.num_nodes,
            calibration.num_nodes,
            test.num_nodes,
        )
        return BaseData(population, raw, folds, test, calibration)

    def batch(self, cycle: int, base: BaseData) -> RawDataset:
        """Verify the incoming batch schema, graph and input-base lineage."""
        graph = load_data(
            self.path(f"data/prediction/cycle_{cycle}/pred_data.pt"), "Incoming batch"
        )
        return self.batch_graph(graph, base.raw)

    def batch_graph(self, graph: Data, parent: RawDataset) -> RawDataset:
        """Check a loaded incoming graph against its exact parent and applied schema."""
        raw = read_raw_graph(graph)
        verify_graph(graph, self.protocol, "prediction")
        if (raw.feature_names, raw.class_names) != self.schema or getattr(
            graph, "source_parent_sha256", None
        ) != parent.dataset_sha256:
            raise ValueError("Incoming batch schema or base lineage does not match.")
        if not set(raw.sample_ids).isdisjoint(parent.sample_ids):
            raise ValueError("Incoming batch reuses sample IDs from its parent base.")
        return raw

    def append(self, cycle: int, base: BaseData, batch: RawDataset) -> BaseData:
        """Verify that the post-append base retains both ordered source populations."""
        appended = self.base(cycle)
        raw = appended.raw
        if raw.sample_ids != base.raw.sample_ids + batch.sample_ids:
            raise ValueError(
                "Post-append sample IDs do not match the input base and batch."
            )
        boundary = base.population.total
        for attribute in ("features", "coordinates", "labels"):
            values = getattr(raw, attribute)
            if not np.array_equal(
                values[:boundary], getattr(base.raw, attribute)
            ) or not np.array_equal(values[boundary:], getattr(batch, attribute)):
                raise ValueError(
                    f"Post-append {attribute} differ from the input base or batch."
                )
        return appended

    def bound_model(
        self, cycle: int, base: BaseData
    ) -> tuple[ModelMetadata, dict, str]:
        """Check the deployed model and preprocessing against development membership."""
        name = self.settings["default_model"]
        directory = self.path(f"trained/cycle_{cycle}")
        state, bundle, _ = read_model_bundle(
            directory,
            *self.schema,
            self.protocol,
            settings=model_graph_settings(self.protocol, name, self.settings),
        )
        partition = TrainingPartition(
            base.raw.dataset_sha256, None, base.development_rows()
        )
        if (
            state.dataset_sha256 != base.raw.dataset_sha256
            or state.partition_sha256 != partition.partition_sha256
            or bundle["training_observation_count"] != base.population.development
            or state.scaler_type != self.settings["data"]["scaler_type"]
        ):
            raise ValueError(
                "Model and preprocessing are not bound to the expected development membership."
            )
        training = read_json(directory / "graph_binding.json")["training"]
        seeds = training.get("seed_provenance")
        device = seeds.get("device") if isinstance(seeds, dict) else None
        return (
            ModelMetadata(
                name,
                state.scaler_type,
                base.population.development,
                bundle["weights_sha256"],
                bundle["training_graph_sha256"],
                device if isinstance(device, str) else None,
            ),
            bundle,
            partition.partition_sha256,
        )

    def model(self, cycle: int, base: BaseData) -> tuple[ModelMetadata, dict, str]:
        """Also verify the reconstructed training topology for overview evidence."""
        metadata, bundle, partition = self.bound_model(cycle, base)
        development = build_development_graph(
            base.raw, base.folds, base.test, base.calibration, self.protocol
        )
        if metadata.training_graph_sha256 != development.graph_sha256:
            raise ValueError("Model graph binding differs from the development graph.")
        return metadata, bundle, partition

    def calibration(
        self, cycle: int, base: BaseData, bundle: dict
    ) -> CalibrationMetadata:
        """Check the calibration sidecar and bytes against the exact model and fit split."""
        directory = self.path(f"evaluation/cycle_{cycle}")
        verified_calibrator(directory, bundle, base.raw.dataset_sha256)
        binding = read_json(directory / "calibration_binding.json")
        if binding["calibration_split_sha256"] != base.calibration.split_sha256:
            raise ValueError(
                "Calibration binding points to a different fitting partition."
            )
        return CalibrationMetadata(
            self.settings["evaluate"]["calibration_method"],
            binding["calibration_split_sha256"],
            binding["calibrator_sha256"],
        )

    def drift(
        self,
        cycle: int,
        base: BaseData,
        batch: RawDataset,
        bundle: dict,
        partition: str,
    ) -> None:
        """Check drift metadata against the paired populations and fitted representation."""
        metadata = read_json(
            self.path(f"drift_analysis/cycle_{cycle}/drift_metadata.json")
        )
        settings = self.settings["analyze_drift"]
        expected = {
            "version": 1,
            "reference_policy": "all_previous_base_observations",
            "reference_dataset_sha256": base.raw.dataset_sha256,
            "incoming_dataset_sha256": batch.dataset_sha256,
            "feature_names": list(self.schema[0]),
            "class_names": list(self.schema[1]),
            "model_cycle": cycle,
            "training_partition_sha256": partition,
            "scaler_type": self.settings["data"]["scaler_type"],
            "configuration": settings,
            "seed_provenance": {
                "scheme": SEED_SCHEME,
                "configured_seed": settings["seed"],
                "effective_seed": derive_seed(settings["seed"], "drift", cycle),
            },
            "representations": {
                "marginal": "canonical_raw_float64",
                "multivariate": "frozen_model_transform_float64",
            },
            **bundle,
        }
        if any(
            not same_value(value, metadata.get(key)) or key not in metadata
            for key, value in expected.items()
        ):
            raise ValueError(
                "Drift metadata differs from the declared populations, model or settings."
            )
