"""Verify independent topology, edge affinities and graph diagnostics on CPU."""

import math

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.utilities.spatial_graph import (
    DistributionSummary,
    build_spatial_graph,
    summarize_graph,
)


def _line(*positions: float) -> np.ndarray:
    return np.column_stack((positions, np.zeros((len(positions), 2))))


def _directed(graph: Data) -> list[tuple[int, int]]:
    return [tuple(edge) for edge in graph.edge_index.T.tolist()]


def _assert_links(graph: Data, undirected: set[tuple[int, int]]) -> None:
    expected = undirected | {(dst, src) for src, dst in undirected}
    edges = _directed(graph)
    assert edges == sorted(expected)


@pytest.mark.parametrize(
    ("rule", "points", "links"),
    [
        ({"k_nearest": 1}, (0, 1, 4, 10), {(0, 1), (1, 2), (2, 3)}),
        ({"connection_radius": 3}, (0, 1, 4, 10), {(0, 1)}),
        (
            {"distance_percentile": 75},
            (0, 1, 4, 10),
            {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)},
        ),
    ],
)
def test_exact_edges_for_each_rule(
    rule: dict, points: tuple, links: set[tuple[int, int]]
) -> None:
    """Return the exact reciprocal union and exclude self and duplicate edges."""
    graph = build_spatial_graph(_line(*points), ("a", "b", "c", "d"), **rule)
    _assert_links(graph, links)
    assert graph.edge_index.dtype == torch.int64
    assert graph.edge_weight.dtype == graph.edge_attr.dtype == torch.float32
    assert graph.edge_distance.dtype == graph.coordinates.dtype == torch.float64
    assert graph.edge_index.device.type == "cpu"


def test_knn_clamps_quota_to_available_observations() -> None:
    """Include all distinct neighbors when k exceeds the population size."""
    graph = build_spatial_graph(_line(0, 4, 8), ("a", "b", "c"), k_nearest=25)
    _assert_links(graph, {(0, 1), (0, 2), (1, 2)})
    assert summarize_graph(graph).effective_k == 2


def test_knn_ties_preserve_sample_alignment_under_reordering() -> None:
    """Break a boundary tie by ID even if input row order changes."""
    points = _line(0, -1, 1, -1.125, 1.125)
    ids = ("query", "z", "a", "left", "right")
    expected = {("query", "a"), ("z", "left"), ("a", "right")}
    expected |= {(dst, src) for src, dst in expected}
    graph = build_spatial_graph(points, ids, k_nearest=1)

    reversed_graph = build_spatial_graph(points[::-1], ids[::-1], k_nearest=1)
    for built, rows, sample_ids in (
        (graph, points, ids),
        (reversed_graph, points[::-1], ids[::-1]),
    ):
        edges = {
            (built.sample_ids[src], built.sample_ids[dst])
            for src, dst in _directed(built)
        }
        assert edges == expected
        assert built.sample_ids == sample_ids
        np.testing.assert_array_equal(built.coordinates.numpy(), rows)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ({"k_nearest": 1}, {(0, 1), (1, 2)}),
        ({"connection_radius": 0.5}, {(0, 1), (0, 2), (1, 2)}),
        ({"distance_percentile": 0}, {(0, 1), (0, 2), (1, 2)}),
    ],
)
def test_colocated_samples_are_distinct(
    rule: dict, expected: set[tuple[int, int]]
) -> None:
    """Keep non-loop zero-distance links and resolve tied KNN candidates by ID."""
    graph = build_spatial_graph(
        _line(0, 0, 0), ("c", "a", "b"), affinity_scale=0.5, **rule
    )
    _assert_links(graph, expected)
    torch.testing.assert_close(
        graph.edge_distance, torch.zeros(len(expected) * 2, dtype=torch.float64)
    )
    torch.testing.assert_close(graph.edge_weight, torch.ones(len(expected) * 2))


def test_percentile_is_per_node_linear_and_inclusive() -> None:
    """Include threshold ties and handle the smallest nontrivial distance row."""
    graph = build_spatial_graph(
        _line(0, -1, 1, -1.125, 1.125),
        ("q", "left", "right", "outer_left", "outer_right"),
        distance_percentile=0,
    )
    _assert_links(graph, {(0, 1), (0, 2), (1, 3), (2, 4)})
    pair = build_spatial_graph(_line(0, 1), ("a", "b"), distance_percentile=100)
    _assert_links(pair, {(0, 1)})


@pytest.mark.parametrize(
    ("loops", "loop_weight"), [(False, 1.0), (True, 1.0), (True, 2.0)]
)
def test_loop_ownership_and_affinity(loops: bool, loop_weight: float) -> None:
    """Keep exactly one explicit loop with its own configured positive weight."""
    graph = build_spatial_graph(
        _line(0, 1, 3),
        ("a", "b", "c"),
        connection_radius=4,
        add_self_loops=loops,
        loop_affinity=loop_weight,
        affinity_scale=2.0,
    )
    expected = {(0, 1), (0, 2), (1, 2)}
    if loops:
        expected.update((i, i) for i in range(3))
    _assert_links(graph, expected)
    expected_weights = [
        loop_weight if src == dst else 1 / (1 + (float(distance) / 2) ** 2)
        for (src, dst), distance in zip(
            _directed(graph), graph.edge_distance, strict=True
        )
    ]
    torch.testing.assert_close(graph.edge_weight, torch.tensor(expected_weights))
    torch.testing.assert_close(graph.edge_attr, graph.edge_weight[:, None])
    assert summarize_graph(graph).self_loop_count == (3 if loops else 0)


def test_unit_weights_and_extreme_positive_affinity_scale() -> None:
    """Support unit weights and keep the limit of very short scales finite."""
    coords = _line(0, 0, 1)
    ids = ("a", "b", "c")
    unit = build_spatial_graph(coords, ids, k_nearest=2)
    torch.testing.assert_close(unit.edge_weight, torch.ones(6))
    scaled = build_spatial_graph(coords, ids, k_nearest=2, affinity_scale=1e-320)
    assert torch.isfinite(scaled.edge_weight).all()
    torch.testing.assert_close(scaled.edge_weight, (scaled.edge_distance == 0).float())


@pytest.mark.parametrize(
    "rule", [{"k_nearest": 3}, {"connection_radius": 1}, {"distance_percentile": 5}]
)
@pytest.mark.parametrize("loops", [False, True])
def test_singleton_graph_and_empty_tensor_shapes(rule: dict, loops: bool) -> None:
    """Keep a singleton even when the declared loop policy creates no edges."""
    graph = build_spatial_graph(_line(1), ("solo",), add_self_loops=loops, **rule)
    assert graph.num_nodes == 1
    assert graph.edge_index.shape == (2, int(loops))
    assert graph.edge_distance.shape == graph.edge_weight.shape == (int(loops),)
    assert graph.edge_attr.shape == (int(loops), 1)
    assert _directed(graph) == ([(0, 0)] if loops else [])
    stats = summarize_graph(graph)
    assert stats.isolated_node_count == 1
    assert stats.density == 0.0
    assert stats.non_loop_edge_length is None
    assert stats.effective_k == (0 if "k_nearest" in rule else None)


def test_disconnected_graph_and_statistics_exclude_duplicate_inflation() -> None:
    """Measure unique stored links, degrees, lengths and isolates."""
    graph = build_spatial_graph(
        _line(0, 1, 10, 12, 100),
        tuple("abcde"),
        connection_radius=3,
        add_self_loops=True,
    )
    assert graph.num_nodes == 5
    assert set(_directed(graph)) == {
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
        *((row, row) for row in range(5)),
    }
    stats = summarize_graph(graph)
    assert stats.node_count == 5
    assert stats.undirected_edge_count == 2
    assert stats.directed_edge_count == 9
    assert stats.self_loop_count == 5
    assert stats.isolated_node_count == 1
    assert stats.non_loop_degree == DistributionSummary(0, 1, 0.8, 1)
    assert stats.density == pytest.approx(0.2)
    assert stats.non_loop_edge_length == DistributionSummary(1, 1.5, 1.5, 2)

    repeated = Data(
        num_nodes=5,
        coordinates=graph.coordinates,
        edge_index=torch.cat((graph.edge_index, graph.edge_index[:, :3]), dim=1),
    )
    assert summarize_graph(repeated) == stats

    all_isolates = build_spatial_graph(_line(0, 5), ("a", "b"), connection_radius=1)
    assert all_isolates.edge_index.shape == (2, 0)
    assert summarize_graph(all_isolates).isolated_node_count == 2


@pytest.mark.parametrize(
    ("coords", "ids", "message"),
    [
        (_line(), (), "without observations"),
        (np.zeros((2, 2)), ("a", "b"), "shape"),
        (np.array([0.0, 1.0, 2.0]), ("a",), "shape"),
        (_line(0).astype(np.float32), ("a",), "float64"),
        (_line(math.inf), ("a",), "finite"),
        (_line(math.nan), ("a",), "finite"),
        (_line(0, 1), ("a", "a"), "unique"),
        (_line(0, 1), ("a",), "coordinate row"),
        (_line(0), (" ",), "non-empty sample ID"),
    ],
)
def test_invalid_observations_fail_with_actionable_errors(
    coords: np.ndarray, ids: tuple[str, ...], message: str
) -> None:
    """Reject malformed canonical observations and identifiers."""
    with pytest.raises(ValueError, match=message):
        build_spatial_graph(coords, ids, k_nearest=1)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({}, "exactly one"),
        ({"k_nearest": 1, "connection_radius": 1}, "exactly one"),
        ({"k_nearest": 0}, "positive integer"),
        ({"k_nearest": 2.0}, "positive integer"),
        ({"k_nearest": True}, "positive integer"),
        ({"connection_radius": 0}, "positive"),
        ({"connection_radius": math.inf}, "finite"),
        ({"distance_percentile": -1}, "between 0 and 100"),
        ({"distance_percentile": 101}, "between 0 and 100"),
        ({"distance_percentile": math.nan}, "finite"),
        ({"k_nearest": 1, "affinity_scale": 0}, "Affinity scale must be positive"),
        (
            {"k_nearest": 1, "affinity_scale": math.inf},
            "Affinity scale must be a finite number",
        ),
        ({"k_nearest": 1, "affinity_scale": -1}, "Affinity scale must be positive"),
        ({"k_nearest": 1, "add_self_loops": 1}, "must be a boolean"),
        ({"k_nearest": 1, "loop_affinity": 0}, "Loop affinity must be positive"),
        ({"k_nearest": 1, "loop_affinity": 1e100}, "positive float32 range"),
    ],
)
def test_invalid_settings_fail_with_actionable_errors(
    settings: dict, message: str
) -> None:
    """Reject invalid rules, affinities and loop options."""
    with pytest.raises(ValueError, match=message):
        build_spatial_graph(_line(1), ("a",), **settings)


@pytest.mark.parametrize(
    "rule", [{"k_nearest": 1}, {"connection_radius": 7}, {"distance_percentile": 75}]
)
def test_training_graph_is_independent_of_held_out_coordinates(rule: dict) -> None:
    """Reconstruct from designated rows even when nearby held-out points change."""
    train = _line(0.000000001, 2, 6)
    train_ids = ("a", "b", "c")
    expected = build_spatial_graph(train, train_ids, affinity_scale=2.0, **rule)
    for held_out in (_line(1), _line(2.000000001, 9, 100)):
        parent = np.vstack((held_out, train))
        sample_ids = tuple(f"held-{row}" for row in range(len(held_out))) + train_ids
        indices = np.arange(len(held_out), len(parent))
        graph = build_spatial_graph(
            parent[indices],
            [sample_ids[row] for row in indices],
            affinity_scale=2.0,
            **rule,
        )
        for attribute in ("coordinates", "edge_index", "edge_distance", "edge_weight"):
            torch.testing.assert_close(
                graph[attribute], expected[attribute], rtol=0, atol=0
            )
        assert graph.sample_ids == train_ids
