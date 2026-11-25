"""Build and summarize independent spatial graphs from canonical sample rows."""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torch_geometric.data import Data


def _coordinates(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    if (
        not isinstance(coordinates, np.ndarray)
        or coordinates.dtype != np.float64
        or coordinates.ndim != 2
        or coordinates.shape[1] != 3
    ):
        raise ValueError("Supply canonical float64 coordinates with shape (N, 3).")
    if not len(coordinates):
        raise ValueError("Cannot construct a spatial graph without observations.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates must be finite.")
    return coordinates


def _finite_number(value: float, name: str) -> None:
    try:
        if (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            return
    except OverflowError:
        pass
    raise ValueError(f"{name} must be a finite number.")


def _positive(value: float, name: str) -> None:
    _finite_number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _distances(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    with np.errstate(over="ignore"):
        distances = np.hypot.reduce(left - right, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError(
            "Spatial distances exceed float64 range. Check the coordinate units."
        )
    return distances


def _knn_pairs(
    coordinates: NDArray[np.float64], sample_ids: tuple[str, ...], quota: int
) -> NDArray[np.int64]:
    tree = KDTree(coordinates)
    ranks = np.argsort(np.argsort(sample_ids))
    targets = np.empty((len(coordinates), quota), dtype=np.int64)
    # One row at a time: the extra candidate allows explicit self exclusion.
    for source, point in enumerate(coordinates):
        distances, nearest = tree.query(point, k=quota + 1)
        if not np.isfinite(distances).all():
            raise ValueError(
                "KNN distances exceed KDTree range. Check the coordinate units."
            )
        boundary = distances[nearest != source][quota - 1]
        # Retrieve the whole boundary tie before sorting by stable sample ID.
        radius = np.nextafter(boundary, np.inf)
        candidates = np.union1d(nearest, tree.query_ball_point(point, r=radius))
        candidates = candidates[candidates != source]
        order = np.lexsort(
            (ranks[candidates], _distances(coordinates[candidates], point))
        )
        targets[source] = candidates[order[:quota]]
    sources = np.repeat(np.arange(len(coordinates)), quota)
    return np.column_stack((sources, targets.ravel()))


def _radius_pairs(coordinates: NDArray[np.float64], radius: float) -> NDArray[np.int64]:
    pairs = KDTree(coordinates).query_pairs(radius, output_type="ndarray")
    distances = _distances(coordinates[pairs[:, 0]], coordinates[pairs[:, 1]])
    return pairs[distances < radius]


def _percentile_pairs(
    coordinates: NDArray[np.float64], percentile: float
) -> NDArray[np.int64]:
    indices = np.arange(len(coordinates))
    rows = []
    for source, point in enumerate(coordinates):
        others = indices[indices != source]
        distances = _distances(coordinates[others], point)
        threshold = np.percentile(distances, percentile, method="linear")
        neighbors = others[distances <= threshold]
        rows.append(np.column_stack((np.full(neighbors.size, source), neighbors)))
    return np.concatenate(rows)


def build_spatial_graph(
    coordinates: NDArray[np.float64],
    sample_ids: tuple[str, ...] | list[str],
    *,
    k_nearest: int | None = None,
    connection_radius: float | None = None,
    distance_percentile: float | None = None,
    affinity_scale: float | None = None,
    add_self_loops: bool = False,
    loop_affinity: float = 1.0,
) -> Data:
    """Build one partition's undirected graph in canonical sample row order.

    Supply exactly one connectivity setting. KNN caps the non-self quota at
    N-1 and breaks distance ties by sample ID. Radius uses a strict cutoff.
    Percentiles are graph-local and per node, with linear interpolation and
    an inclusive cutoff over distances to other samples.

    With no affinity scale, non-loop weights are one. A positive scale uses
    inverse quadratic affinity in coordinate units. Loops have an explicit
    weight independent of the selected affinity rule.

    Returns:
        PyG data with float64 coordinates and edge distances, int64 COO
        indices, float32 weights and a one-column float32 affinity attribute.

    """
    points = _coordinates(coordinates)
    if (
        not isinstance(sample_ids, tuple | list)
        or len(sample_ids) != len(points)
        or any(not isinstance(name, str) or not name.strip() for name in sample_ids)
        or len(set(sample_ids)) != len(points)
    ):
        raise ValueError(
            "Supply a unique non-empty sample ID for every coordinate row."
        )
    ids = tuple(sample_ids)
    if (
        sum(
            rule is not None
            for rule in (k_nearest, connection_radius, distance_percentile)
        )
        != 1
    ):
        raise ValueError(
            "Specify exactly one connectivity rule: KNN, radius or percentile."
        )
    if not isinstance(add_self_loops, bool):
        raise ValueError("add_self_loops must be a boolean.")
    _positive(loop_affinity, "Loop affinity")
    if (
        not float(np.finfo(np.float32).smallest_subnormal)
        <= loop_affinity
        <= float(np.finfo(np.float32).max)
    ):
        raise ValueError("Loop affinity must fit in positive float32 range.")
    if affinity_scale is not None:
        _positive(affinity_scale, "Affinity scale")

    pairs = np.empty((0, 2), dtype=np.int64)
    effective_k = None
    if k_nearest is not None:
        if (
            isinstance(k_nearest, bool)
            or not isinstance(k_nearest, Integral)
            or k_nearest < 1
        ):
            raise ValueError("KNN neighbor count must be a positive integer.")
        effective_k = min(int(k_nearest), len(points) - 1)
        if effective_k:
            pairs = _knn_pairs(points, ids, effective_k)
    elif connection_radius is not None:
        _positive(connection_radius, "Connection radius")
        pairs = _radius_pairs(points, float(connection_radius))
    else:
        _finite_number(distance_percentile, "Distance percentile")
        if not 0 <= distance_percentile <= 100:
            raise ValueError("Distance percentile must be between 0 and 100.")
        if len(points) > 1:
            pairs = _percentile_pairs(points, distance_percentile)

    links = np.unique(np.sort(pairs, axis=1), axis=0)
    edges = np.concatenate((links, links[:, ::-1]))
    if add_self_loops:
        nodes = np.arange(len(points))
        edges = np.concatenate((edges, np.column_stack((nodes, nodes))))
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    src, dst = edges.T
    distances = _distances(points[src], points[dst])
    if affinity_scale is None:
        weights = np.ones_like(distances)
    else:
        with np.errstate(over="ignore"):
            weights = 1 / (1 + np.square(distances / float(affinity_scale)))
    weights[src == dst] = float(loop_affinity)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return Data(
        num_nodes=len(points),
        sample_ids=ids,
        coordinates=torch.from_numpy(points.copy()),
        edge_index=torch.tensor(edges.T, dtype=torch.int64),
        edge_distance=torch.from_numpy(distances),
        edge_weight=edge_weight,
        edge_attr=edge_weight.unsqueeze(1),
        effective_k=effective_k,
    )


@dataclass(frozen=True)
class DistributionSummary:
    """Summarize a non-empty set of numeric measurements."""

    minimum: float
    median: float
    mean: float
    maximum: float


def _summary(values: NDArray) -> DistributionSummary:
    # Divide first to avoid overflowing the sum of large finite distances.
    return DistributionSummary(
        minimum=float(values.min()),
        median=float(np.quantile(values, 0.5)),
        mean=float((values / len(values)).sum()),
        maximum=float(values.max()),
    )


@dataclass(frozen=True)
class GraphStatistics:
    """Describe unique graph edges, with degrees and density excluding loops."""

    node_count: int
    undirected_edge_count: int
    directed_edge_count: int
    self_loop_count: int
    isolated_node_count: int
    non_loop_degree: DistributionSummary
    density: float
    non_loop_edge_length: DistributionSummary | None
    effective_k: int | None


def summarize_graph(graph: Data) -> GraphStatistics:
    """Measure unique stored COO edges and summarize their physical lengths.

    Count each distinct directed edge once, including loops. Treat a pair of
    endpoints as one undirected link for density, degree and length summaries.
    A graph with no non-loop edges has no edge-length summary.
    """
    points = _coordinates(graph.coordinates.detach().cpu().numpy())
    if graph.num_nodes != len(points):
        raise ValueError("Graph node count does not match its coordinate rows.")
    edge_index = graph.edge_index
    if (
        edge_index.dtype != torch.int64
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or ((edge_index < 0) | (edge_index >= len(points))).any()
    ):
        raise ValueError(
            "Graph edges must contain in-bounds int64 COO indices of shape (2, E)."
        )
    edges = np.unique(edge_index.detach().cpu().numpy().T, axis=0)
    loops = edges[:, 0] == edges[:, 1]
    links = np.unique(np.sort(edges[~loops], axis=1), axis=0)
    degrees = np.bincount(links.ravel(), minlength=len(points))
    lengths = _distances(points[links[:, 0]], points[links[:, 1]])
    count = len(points)
    return GraphStatistics(
        node_count=count,
        undirected_edge_count=len(links),
        directed_edge_count=len(edges),
        self_loop_count=int(loops.sum()),
        isolated_node_count=int(np.count_nonzero(degrees == 0)),
        non_loop_degree=_summary(degrees),
        density=2 * len(links) / (count * (count - 1)) if count > 1 else 0.0,
        non_loop_edge_length=_summary(lengths) if len(lengths) else None,
        effective_k=getattr(graph, "effective_k", None),
    )
