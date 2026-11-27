"""Read selected trusted-local observations and prepare display-only graph views."""

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from torch_geometric.data import Data

from dashboard.artifacts import Workspace
from dashboard.dvc_evidence import WorkspaceStamp
from dashboard.provenance import AppliedConfiguration
from dashboard.scientific_summary import ScientificReader
from src.configuration_files import configuration_operation
from src.utilities.general_utils import load_data
from src.utilities.graph_protocol import GRAPH_PROTOCOL_VERSION, verify_graph
from src.utilities.preprocessing import RawDataset
from src.utilities.raw_graphs import RAW_SCHEMA_VERSION, read_raw_graph
from src.utilities.spatial_graph import GraphStatistics, summarize_graph


@dataclass(frozen=True)
class ArraySnapshot:
    """Keep numeric storage immutable and give each consumer its own array view."""

    payload: bytes
    dtype: np.dtype
    shape: tuple[int, ...]

    @classmethod
    def freeze(cls, values: NDArray) -> "ArraySnapshot":
        """Capture numeric values without narrowing their dtype."""
        return cls(values.tobytes(), values.dtype, values.shape)

    def array(self) -> NDArray:
        """Return a read-only view with independent array metadata."""
        return np.frombuffer(self.payload, dtype=self.dtype).reshape(self.shape)


class PopulationKind(StrEnum):
    """Identify the selected cohort independently of any display filter."""

    SNAPSHOT = "full_base"
    INPUT_BASE = "input_base"
    FINAL_TEST = "final_test"
    CALIBRATION = "calibration"
    CV_TRAIN = "cv_train"
    CV_VALIDATION = "cv_validation"
    DEVELOPMENT = "development"
    INCOMING = "incoming_batch"
    POST_APPEND = "post_append"
    FINAL_APPEND = "final_append"


class Connectivity(StrEnum):
    """Distinguish descriptive full-base edges from independent model graphs."""

    DESCRIPTIVE = "descriptive_full_base"
    INDEPENDENT = "independent_partition"


@dataclass(frozen=True)
class SpatialTopology:
    """Retain stored topology and statistics alongside deduplicated display links."""

    kind: Connectivity
    graph_sha256: str
    protocol_version: int
    split_sha256: str | None
    statistics: GraphStatistics
    edge_indices: ArraySnapshot
    edge_weights: ArraySnapshot
    edge_distances: ArraySnapshot
    links: ArraySnapshot

    @classmethod
    def read(cls, graph: Data, kind: Connectivity) -> "SpatialTopology":
        """Prepare undirected links from verified COO endpoints without creating edges."""
        version = graph.graph_protocol["version"]
        if type(version) is not int or version != GRAPH_PROTOCOL_VERSION:
            raise ValueError("Unsupported stored graph protocol version.")
        statistics = summarize_graph(graph)
        edges = graph.edge_index.numpy()
        weights, distances = graph.edge_weight.numpy(), graph.edge_distance.numpy()
        if (
            weights.shape != (edges.shape[1],)
            or distances.shape != weights.shape
            or weights.dtype != np.float32
            or distances.dtype != np.float64
            or graph.edge_attr.numpy().dtype != np.float32
            or graph.edge_attr.shape != (edges.shape[1], 1)
            or not np.array_equal(graph.edge_attr.numpy().ravel(), weights)
            or not np.isfinite(weights).all()
            or not np.isfinite(distances).all()
        ):
            raise ValueError(
                "Stored edge weights, attributes or distances are invalid."
            )
        links = np.unique(np.sort(edges[:, edges[0] != edges[1]].T, axis=1), axis=0)
        return cls(
            kind,
            graph.graph_sha256,
            version,
            getattr(graph, "split_sha256", None),
            statistics,
            *(
                ArraySnapshot.freeze(values)
                for values in (edges, weights, distances, links)
            ),
        )


@dataclass(frozen=True, eq=False)
class SpatialPopulation:
    """Hold aligned observations in source order with selected membership and context."""

    kind: PopulationKind
    cycle: int | None
    snapshot: int | None
    fold: int | None
    sample_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    dataset_sha256: str
    parent_sha256: str | None
    topology: SpatialTopology | None
    _coordinates: ArraySnapshot
    _features: ArraySnapshot
    _labels: ArraySnapshot | None
    _append_boundary: int | None
    schema_version: int = RAW_SCHEMA_VERSION

    @property
    def count(self) -> int:
        """Return the original population size."""
        return len(self.sample_ids)

    @property
    def coordinates(self) -> NDArray[np.float64]:
        """Return canonical XYZ coordinates in synthetic coordinate units."""
        return self._coordinates.array()

    @property
    def raw_features(self) -> NDArray[np.float64]:
        """Return float64 features in the stored feature order."""
        return self._features.array()

    @property
    def observed_class_ids(self) -> NDArray[np.int64] | None:
        """Return observed class indices, when present."""
        return self._labels.array() if self._labels else None

    @property
    def observed_class_names(self) -> tuple[str, ...] | None:
        """Return observed class names using the verified class mapping."""
        labels = self.observed_class_ids
        return (
            tuple(self.class_names[index] for index in labels)
            if labels is not None
            else None
        )

    @property
    def membership(self) -> tuple[str, ...]:
        """Return selected population roles or verified source roles for an append."""
        if self._append_boundary is None:
            return (self.kind.value,) * self.count
        return ("input_base",) * self._append_boundary + ("incoming_batch",) * (
            self.count - self._append_boundary
        )

    def view(
        self,
        *,
        depth: tuple[float, float] | None = None,
        class_ids: Collection[int] | None = None,
        sample_ids: Collection[str] | None = None,
    ) -> "SpatialView":
        """Restrict visible rows by inclusive stored Z, observed class or stable ID."""
        mask = np.ones(self.count, dtype=bool)
        if depth is not None:
            lower, upper = depth
            if not np.isfinite(depth).all() or lower > upper:
                raise ValueError("Choose finite depth bounds in ascending order.")
            z = self.coordinates[:, 2]
            mask &= (lower <= z) & (z <= upper)
        if class_ids is not None:
            labels = self.observed_class_ids
            if labels is None:
                raise ValueError(
                    "Observed classes are unavailable for this population."
                )
            if not all(
                isinstance(index, Integral)
                and not isinstance(index, bool)
                and 0 <= index < len(self.class_names)
                for index in class_ids
            ):
                raise ValueError("Choose known observed class indices.")
            mask &= np.isin(labels, list(class_ids))
        if sample_ids is not None:
            selected = set(sample_ids)
            mask &= np.fromiter(
                (sid in selected for sid in self.sample_ids), dtype=bool
            )
        return SpatialView(
            self,
            ArraySnapshot.freeze(mask),
            any(value is not None for value in (depth, class_ids, sample_ids)),
        )


@dataclass(frozen=True)
class SpatialView:
    """Track visible rows separately from the original population and graph statistics."""

    population: SpatialPopulation
    _visible: ArraySnapshot
    filtered: bool

    @property
    def row_indices(self) -> NDArray[np.int64]:
        """Return visible row positions in original order."""
        return np.flatnonzero(self._visible.array())

    @property
    def count(self) -> int:
        """Return the visible observation count."""
        return int(np.count_nonzero(self._visible.array()))

    def edge_rows(self) -> NDArray[np.int64]:
        """Return stored physical links restricted to visible population row indices."""
        topology = self.population.topology
        if topology is None:
            return np.empty((0, 2), dtype=np.int64)
        mask, links = self._visible.array(), topology.links.array()
        return links[mask[links].all(axis=1)]

    def edge_ids(self) -> tuple[tuple[str, str], ...]:
        """Resolve each visible physical link to stable observation IDs."""
        ids = self.population.sample_ids
        return tuple((ids[left], ids[right]) for left, right in self.edge_rows())


def make_population(
    kind: PopulationKind,
    raw: RawDataset,
    graph: Data | None,
    *,
    cycle: int | None = None,
    snapshot: int | None = None,
    fold: int | None = None,
    rows: NDArray[np.int64] | None = None,
    parent_sha256: str | None = None,
    append_boundary: int | None = None,
) -> SpatialPopulation:
    """Freeze verified in-order rows and the selected stored graph, if any."""
    selection = slice(None) if rows is None else rows
    topology = None
    if graph is not None:
        connectivity = (
            Connectivity.DESCRIPTIVE
            if graph.split_type == "overview"
            else Connectivity.INDEPENDENT
        )
        topology = SpatialTopology.read(graph, connectivity)
    return SpatialPopulation(
        kind,
        cycle,
        snapshot,
        fold,
        raw.sample_ids
        if rows is None
        else tuple(raw.sample_ids[index] for index in rows),
        raw.class_names,
        raw.feature_names,
        raw.dataset_sha256,
        parent_sha256,
        topology,
        ArraySnapshot.freeze(raw.coordinates[selection]),
        ArraySnapshot.freeze(raw.features[selection]),
        ArraySnapshot.freeze(raw.labels[selection]) if raw.labels is not None else None,
        append_boundary,
    )


_BASE_FILES = ("base_data.pt", "fold_data.pt", "test_data.pt", "calib_data.pt")


def _base_path(snapshot: int, filename: str = "base_data.pt") -> str:
    return f"data/base/cycle_{snapshot}/{filename}"


def _full_base(reader: ScientificReader, snapshot: int) -> tuple[RawDataset, Data]:
    graph = load_data(reader.path(_base_path(snapshot)), "Full base")
    raw = read_raw_graph(graph)
    verify_graph(graph, reader.protocol, "overview")
    if (raw.feature_names, raw.class_names) != reader.schema:
        raise ValueError("Base schema differs from the applied configuration.")
    return raw, graph


def _load_population(
    reader: ScientificReader, kind: PopulationKind, cycle: int, fold: int | None
) -> SpatialPopulation:
    if kind == PopulationKind.INCOMING:
        parent, _ = _full_base(reader, cycle - 1)
        graph = load_data(
            reader.path(f"data/prediction/cycle_{cycle}/pred_data.pt"), "Incoming batch"
        )
        raw = reader.batch_graph(graph, parent)
        return make_population(
            kind, raw, graph, cycle=cycle, parent_sha256=parent.dataset_sha256
        )
    snapshot = (
        cycle
        if kind
        in (
            PopulationKind.SNAPSHOT,
            PopulationKind.POST_APPEND,
            PopulationKind.FINAL_APPEND,
        )
        else cycle - 1
    )
    if kind in (
        PopulationKind.SNAPSHOT,
        PopulationKind.INPUT_BASE,
        PopulationKind.POST_APPEND,
        PopulationKind.FINAL_APPEND,
    ):
        raw, graph = _full_base(reader, snapshot)
        parent_sha256 = append_boundary = None
        if kind in (PopulationKind.POST_APPEND, PopulationKind.FINAL_APPEND):
            previous = reader.base(cycle - 1)
            appended = reader.append(cycle, previous, reader.batch(cycle, previous))
            if raw.dataset_sha256 != appended.raw.dataset_sha256:
                raise ValueError(
                    "The selected final base differs from the verified append."
                )
            parent_sha256 = previous.raw.dataset_sha256
            append_boundary = previous.population.total
        return make_population(
            kind,
            raw,
            graph,
            cycle=None if kind == PopulationKind.SNAPSHOT else cycle,
            snapshot=snapshot,
            parent_sha256=parent_sha256,
            append_boundary=append_boundary,
        )

    base = reader.base(snapshot)
    if kind == PopulationKind.DEVELOPMENT:
        graph, rows = None, base.development_rows()
    else:
        if kind in (PopulationKind.CV_TRAIN, PopulationKind.CV_VALIDATION):
            if type(fold) is not int or not 0 <= fold < len(base.folds):
                raise ValueError("Select an available zero-based CV fold.")
            graph = base.folds[fold][int(kind == PopulationKind.CV_VALIDATION)]
        else:
            graph = base.test if kind == PopulationKind.FINAL_TEST else base.calibration
        rows = graph.original_node_indices.numpy()
    return make_population(
        kind,
        base.raw,
        graph,
        cycle=cycle,
        snapshot=snapshot,
        fold=fold,
        rows=rows,
        parent_sha256=base.raw.dataset_sha256,
    )


def population_paths(kind: PopulationKind, cycle: int) -> tuple[str, ...]:
    """List the raw artifacts that identify a selected population snapshot."""
    if kind == PopulationKind.SNAPSHOT:
        paths = [_base_path(cycle)]
    elif kind == PopulationKind.INPUT_BASE:
        paths = [_base_path(cycle - 1)]
    elif kind == PopulationKind.INCOMING:
        paths = [_base_path(cycle - 1), f"data/prediction/cycle_{cycle}/pred_data.pt"]
    elif kind in (PopulationKind.POST_APPEND, PopulationKind.FINAL_APPEND):
        paths = [
            *(
                _base_path(index, name)
                for index in (cycle - 1, cycle)
                for name in _BASE_FILES
            ),
            f"data/prediction/cycle_{cycle}/pred_data.pt",
        ]
    else:
        paths = [_base_path(cycle - 1, name) for name in _BASE_FILES]
    return tuple(f"results/{path}" for path in paths)


def read_population(
    workspace: Workspace, kind: PopulationKind, *, cycle: int, fold: int | None = None
) -> SpatialPopulation:
    """Read and structurally verify a trusted local population under the workspace lock.

    Snapshot selections use the base index (including bootstrap 0). Other
    selections use the learning cycle. Currentness evidence is supplied
    separately by the existing diagnostic evidence route.
    """
    root: Path = workspace.root.resolve()
    kind = PopulationKind(kind)
    if type(cycle) is not int or cycle < int(kind != PopulationKind.SNAPSHOT):
        raise ValueError("Choose a non-negative snapshot or positive learning cycle.")
    if fold is not None and kind not in (
        PopulationKind.CV_TRAIN,
        PopulationKind.CV_VALIDATION,
    ):
        raise ValueError("Fold selection applies only to CV populations.")
    paths = population_paths(kind, cycle)
    with configuration_operation(root / "params.yaml"):
        stamp = WorkspaceStamp.capture(root, paths)
        params = stamp.controls[0]
        if params is None:
            raise ValueError(
                "An applied configuration is required to verify the raw schema."
            )
        applied, settings = AppliedConfiguration.read(params)
        if kind == PopulationKind.FINAL_APPEND and cycle != applied.cycles:
            raise ValueError(
                "Select the last declared learning cycle for the final append."
            )
        population = _load_population(
            ScientificReader(root, settings), kind, cycle, fold
        )
        if stamp != WorkspaceStamp.capture(root, paths):
            raise ValueError(
                "Workspace files changed while loading observations. Try again."
            )
        return population
