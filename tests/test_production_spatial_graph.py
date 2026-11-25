"""Probe production graph semantics using small deterministic CPU graphs."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.models import get_model
from src.stages.train import run_training
from src.utilities.data_utils import export_graph_to_html
from src.utilities.general_utils import load_params
from src.utilities.graph_protocol import (
    model_graph_settings,
    resolve_graph_protocol,
    verify_graph,
)
from src.utilities.preprocessing import RawDataset, TrainingPartition, fit_preprocessing
from src.utilities.raw_graphs import (
    build_cycle_graphs,
    build_raw_graph,
    model_graph,
    synthetic_raw,
)
from src.utilities.spatial_graph import summarize_graph


def _raw(params: dict, distances: list[float]) -> RawDataset:
    rows = np.arange(len(distances), dtype=np.float64)
    points = np.column_stack((distances, rows * 0, rows * 0))
    features = np.random.default_rng(10).normal(size=(len(rows), 5))
    labels = np.arange(len(rows), dtype=np.int64) % 4
    return synthetic_raw(features, points, labels, params["data"]["base"])


def _prepare(raw: RawDataset, params: dict, name: str) -> Data:
    protocol = resolve_graph_protocol(params)
    graph = build_raw_graph(raw, protocol, "prediction")
    partition = TrainingPartition(
        raw.dataset_sha256, 0, np.arange(len(raw.labels), dtype=np.int64)
    )
    state = fit_preprocessing(raw, partition, None)
    return model_graph(graph, raw, state, model_graph_settings(protocol, name, params))


def _model(params: dict, name: str):
    return get_model(name, {**params["models"][name], "add_self_loops": False}).eval()


def test_held_out_coordinates_do_not_change_production_train_graph_or_logits() -> None:
    """Verify the partition-first production factory with a fixed GCN."""
    params = load_params()
    raw = _raw(params, list(range(80)))
    protocol = resolve_graph_protocol(params)
    settings = model_graph_settings(protocol, "gcn", params)
    original = build_cycle_graphs(raw, protocol, params["data"]["base"])[1][0][0]
    train_indices = original.original_node_indices.numpy()
    locations = raw.coordinates.copy()
    outside_training = np.setdiff1d(np.arange(len(locations)), train_indices)
    locations[outside_training, 0] += 25000
    changed_raw = replace(raw, coordinates=locations)
    changed = build_cycle_graphs(changed_raw, protocol, params["data"]["base"])[1][0][0]

    inputs = []
    for graph, parent in ((original, raw), (changed, changed_raw)):
        partition = TrainingPartition(
            parent.dataset_sha256, 0, graph.original_node_indices.numpy()
        )
        state = fit_preprocessing(parent, partition, "RobustScaler")
        inputs.append(model_graph(graph, parent, state, settings))
    first, second = inputs
    for field in (
        "original_node_indices",
        "coordinates",
        "edge_index",
        "edge_weight",
        "x",
        "y",
    ):
        torch.testing.assert_close(first[field], second[field], atol=0, rtol=0)
    assert first.sample_ids == second.sample_ids
    assert original.graph_sha256 != changed.graph_sha256

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.random.default_generator.manual_seed(19)
        model = _model(params, "gcn")
        torch.testing.assert_close(model(first), model(second), atol=0, rtol=0)


@pytest.mark.parametrize(("improved", "loop_affinity"), [(False, 1.0), (True, 2.0)])
def test_gcn_uses_explicit_loop_affinity_and_layer_degree_normalization(
    improved: bool, loop_affinity: float
) -> None:
    """Check the actual first GCN layer against a weighted two-node calculation."""
    params = load_params()
    params["models"]["gcn"]["improved"] = improved
    params["data"]["base"].update(distance_percentile=None, connection_radius=600)
    graph = _prepare(_raw(params, [0, 460]), params, "gcn")
    assert graph.edge_index.T.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    torch.testing.assert_close(
        graph.edge_weight, torch.tensor([loop_affinity, 0.5, 0.5, loop_affinity])
    )

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.random.default_generator.manual_seed(23)
        model = _model(params, "gcn")
        assert all(
            layer.normalize and not layer.add_self_loops for layer in model.convs
        )
        layer = model.convs[0]
        projected = layer.lin(graph.x)
        expected = (loop_affinity * projected + 0.5 * projected.flip(0)) / (
            loop_affinity + 0.5
        )
        expected += layer.bias
        torch.testing.assert_close(
            layer(graph.x, graph.edge_index, graph.edge_weight), expected
        )


@pytest.mark.parametrize("name", ["gcn", "gat"])
def test_unit_weights_and_empty_edge_singleton_inference(name: str) -> None:
    """Run both real model architectures on unit weights and on empty edges."""
    params = load_params()
    params["add_self_loops"] = False
    params["data"]["edge_affinity"] = {"method": "unit", "scale": None}
    params["data"]["base"].update(distance_percentile=None, connection_radius=25)
    singleton = _prepare(_raw(params, [0]), params, name)
    pair = _prepare(_raw(params, [0, 10]), params, name)
    assert singleton.edge_index.shape == (2, 0)
    assert singleton.edge_attr.shape == (0, 1)
    torch.testing.assert_close(pair.edge_weight, torch.ones(2))

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.random.default_generator.manual_seed(29)
        model = _model(params, name)
        assert all(not layer.add_self_loops for layer in model.convs)
        for graph in (singleton, pair):
            logits = model(graph)
            assert logits.shape == (graph.num_nodes, 4)
            assert torch.isfinite(logits).all()
        if name == "gat":
            changed = pair.clone()
            changed.edge_attr.fill_(50)
            changed.edge_weight.zero_()
            assert all(layer.lin_edge is None for layer in model.convs)
            torch.testing.assert_close(model(pair), model(changed), atol=0, rtol=0)


def test_singleton_training_with_batch_norm_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail with a useful message before preprocessing or training starts."""
    params = load_params()
    raw = _raw(params, [0])
    monkeypatch.setattr(
        "src.stages.train.load_cycle_data",
        lambda *_: (raw, [(Data(num_nodes=1), Data(num_nodes=1))], None, None),
    )
    with pytest.raises(ValueError, match=r"one training node.*batch_norm"):
        run_training({"base_data": tmp_path, "output": tmp_path}, params, "gcn", 1)


def test_legacy_graph_configuration_has_explicit_migration_error() -> None:
    """Reject legacy weights and ambiguous rules instead of applying precedence."""
    params = load_params()
    params["data"]["make_edge_weight_method"] = "minmax"
    with pytest.raises(ValueError, match=r"Legacy edge weighting.*data\.edge_affinity"):
        resolve_graph_protocol(params)
    del params["data"]["make_edge_weight_method"]
    params["data"]["base"]["k_nearest"] = 3
    with pytest.raises(ValueError, match="exactly one"):
        resolve_graph_protocol(params)
    params["data"]["base"]["distance_percentile"] = None
    params["models"]["gcn"]["normalize"] = False
    with pytest.raises(ValueError, match=r"models\.gcn\.normalize"):
        model_graph_settings(resolve_graph_protocol(params), "gcn", params)


def test_overview_plot_uses_stored_links_and_noncontiguous_labels(
    tmp_path: Path,
) -> None:
    """Draw one segment per physical link and show actual graph statistics."""
    params = load_params()
    params["data"]["base"].update(distance_percentile=None, connection_radius=2)
    raw = replace(_raw(params, [0, 1, 3]), labels=np.array([0, 2, 2], dtype=np.int64))
    protocol = resolve_graph_protocol(params)
    graph = build_raw_graph(raw, protocol, "overview")
    export_graph_to_html(
        graph, str(tmp_path), dict(enumerate(raw.class_names)), dataset_tag="base_data"
    )
    figure = json.loads((tmp_path / "base_data_graph.json").read_text(encoding="utf-8"))
    assert (tmp_path / "base_data_graph.html").exists()
    assert figure["data"][0]["x"] == [0, 1, None]
    assert figure["data"][1]["text"] == [
        "Prospectivity: Class 0",
        "Prospectivity: Class 2",
        "Prospectivity: Class 2",
    ]
    title = figure["layout"]["title"]["text"]
    stats = summarize_graph(graph)
    for label, count in (
        ("Directed edges", stats.directed_edge_count),
        ("Unique links", stats.undirected_edge_count),
        ("Self-loops", stats.self_loop_count),
        ("Isolates", stats.isolated_node_count),
    ):
        assert f"{label} {count}" in title
    assert "Descriptive full-base overview" in title
    assert "Physical link length" in title

    graph.edge_weight[0] = 0.25
    with pytest.raises(ValueError, match="Graph payload digest mismatch"):
        verify_graph(graph, protocol, "overview")
