"""Regression tests for graph alignment, partitioning, and CPU inference."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.models.gat import MineralDepositGAT
from src.models.gcn import MineralDepositGCN
from src.utilities.general_utils import load_params
from src.utilities.graph_protocol import resolve_graph_protocol
from src.utilities.raw_graphs import build_cycle_graphs, synthetic_raw
from src.utilities.spatial_graph import build_spatial_graph

N_NODES = 80
N_FEATURES = 5
N_CLASSES = 4
GraphSplits = tuple[Data, list[tuple[Data, Data]], Data, Data]


@pytest.fixture(scope="module")
def graph_splits() -> GraphSplits:
    """Build a small graph with fixed inputs and stratified splits."""
    coordinates = np.random.default_rng(3).uniform(size=(N_NODES, 3))
    features = np.random.default_rng(4).normal(size=(N_NODES, N_FEATURES))
    labels = np.arange(N_NODES) % N_CLASSES
    params = load_params()
    base_params = params["data"]["base"]
    base_params.update(distance_percentile=None, connection_radius=0.5)
    raw = synthetic_raw(features, coordinates, labels, base_params)
    graphs = build_cycle_graphs(raw, resolve_graph_protocol(params), base_params)
    base, _, _, _ = graphs
    torch.testing.assert_close(base.x, torch.tensor(features))
    torch.testing.assert_close(base.y, torch.tensor(labels, dtype=torch.long))
    return graphs


@pytest.mark.parametrize(
    ("model_type", "model_options"),
    [(MineralDepositGCN, {"normalize": True}), (MineralDepositGAT, {})],
    ids=["gcn", "gat"],
)
def test_model_forward_cpu(
    graph_splits: GraphSplits,
    model_type: type[MineralDepositGCN] | type[MineralDepositGAT],
    model_options: dict[str, bool],
) -> None:
    """Return finite CPU logits for every node and class."""
    base, _, _, _ = graph_splits
    base = base.clone()
    base.x = base.x.float()
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.random.default_generator.manual_seed(42)
        model = (
            model_type(
                n_classes=N_CLASSES,
                in_channels=N_FEATURES,
                hidden_channels=8,
                add_self_loops=False,
                **model_options,
            )
            .cpu()
            .eval()
        )
        logits = model(base)

    assert logits.device.type == "cpu"
    assert logits.shape == (N_NODES, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_graph_edges_are_valid(graph_splits: GraphSplits) -> None:
    """Check edge bounds and finite attributes on the base and split graphs."""
    base, folds, test, calibration = graph_splits
    fold_graphs = [graph for fold in folds for graph in fold]
    for graph in (base, *fold_graphs, test, calibration):
        edges = graph.edge_index
        assert edges.dtype == torch.long
        assert edges.ndim == 2 and edges.shape[0] == 2
        assert ((edges >= 0) & (edges < graph.num_nodes)).all()
        assert graph.edge_attr.shape[0] == edges.shape[1]
        assert graph.edge_weight.numel() == edges.shape[1]
        assert torch.isfinite(graph.edge_attr).all()
        assert torch.isfinite(graph.edge_weight).all()
    assert (base.edge_index[0] != base.edge_index[1]).any()


def test_partitions_are_disjoint_and_aligned(graph_splits: GraphSplits) -> None:
    """Check per-fold coverage and verify each row maps to its original node."""
    base, folds, test, calibration = graph_splits
    assert len(folds) == 3
    for train, validation in folds:
        seen = set()
        for graph in (train, validation, test, calibration):
            indices = graph.original_node_indices
            members = set(indices.tolist())
            assert len(members) == graph.num_nodes > 0
            assert seen.isdisjoint(members)
            seen.update(members)
            torch.testing.assert_close(graph.x, base.x[indices])
            torch.testing.assert_close(graph.y, base.y[indices])
        assert seen == set(range(N_NODES))


def test_partitions_use_independent_edges(graph_splits: GraphSplits) -> None:
    """Rebuild each graph from its own coordinates and require exact COO edges."""
    _, folds, test, calibration = graph_splits
    fold_graphs = [graph for fold in folds for graph in fold]
    for graph in (*fold_graphs, test, calibration):
        expected = build_spatial_graph(
            graph.coordinates.numpy(),
            graph.sample_ids,
            connection_radius=0.5,
            affinity_scale=460,
            add_self_loops=True,
        )
        torch.testing.assert_close(graph.edge_index, expected.edge_index)
        torch.testing.assert_close(graph.edge_weight, expected.edge_weight)
        assert len(set(map(tuple, graph.edge_index.T.tolist()))) == graph.num_edges
