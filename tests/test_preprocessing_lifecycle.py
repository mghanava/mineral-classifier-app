"""Verify the model-bound preprocessing lifecycle with tiny synthetic artifacts."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib import pyplot
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data

from src.stages import (
    analyze_drift,
    evaluate,
    generate_base_data,
    generate_pred_data,
    predict,
    prepare_next_cycle_data,
    train,
)
from src.utilities import eval_utils, pred_utils
from src.utilities.calibration_utils import CalibratedModel
from src.utilities.drift_detection_utils import AnalyzeDrift
from src.utilities.general_utils import load_data, load_params
from src.utilities.graph_protocol import model_graph_settings, resolve_graph_protocol
from src.utilities.model_bundle import publish_model
from src.utilities.preprocessing import (
    TrainingPartition,
    fit_preprocessing,
    load_preprocessing,
)
from src.utilities.raw_graphs import (
    build_development_graph,
    load_cycle_data,
    read_raw_graph,
    stamp_raw_graph,
)
from src.utilities.spatial_graph import build_spatial_graph
from src.utilities.training_schedule import FoldTrainingResult, final_training_plan


def _observations(
    size: int, offset: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(size, dtype=np.float64)
    labels = (rows % 2).astype(np.int64)
    coords = np.column_stack((rows + offset + 1e-9, rows % 5, -rows))
    values = np.column_stack(
        ((2 * labels - 1) * (rows + offset + 0.25), rows / 7 + 1e-9)
    )
    return coords, values, labels, coords[:2], values[:2]


@pytest.fixture
def generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path, Path]:
    """Run the raw producers with deterministic observations and no plotting."""
    params = load_params()
    params["data"].update(scaler_type="StandardScaler", scaler_params={})
    params["data"]["base"].update(
        n_features=2,
        n_classes=2,
        n_splits=2,
        test_size=0.4,
        calib_size=0.5,
        distance_percentile=None,
        connection_radius=25,
    )
    params["evaluate"]["calibration_method"] = "isotonic"
    for stage in (generate_base_data, generate_pred_data, prepare_next_cycle_data):
        monkeypatch.setattr(stage, "export_graph_to_html", Mock())
    for stage in (generate_base_data, prepare_next_cycle_data):
        monkeypatch.setattr(stage, "export_all_graphs_to_html", Mock())
        monkeypatch.setattr(stage, "analyze_feature_discrimination", Mock())
    monkeypatch.setattr(
        generate_base_data, "generate_mineral_data", lambda **_: _observations(40, 0)
    )
    monkeypatch.setattr(
        generate_pred_data, "generate_mineral_data", lambda **_: _observations(8, 100)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    base, incoming = tmp_path / "base", tmp_path / "incoming"
    generate_base_data.prepare_base_data(base, params)
    generate_pred_data.prepare_pred_data(
        {"base_data": base, "bootstrap_data": base, "output": incoming}, params
    )
    return params, base, incoming


def test_raw_round_trip_and_next_cycle(
    generated: tuple[dict, Path, Path], tmp_path: Path
) -> None:
    """Preserve precision, sample IDs, topology and partition provenance."""
    params, base, incoming = generated
    historical, folds, test, calib = load_cycle_data(
        base, resolve_graph_protocol(params)
    )
    batch_graph = load_data(incoming / "pred_data.pt", "Prediction")
    batch = read_raw_graph(batch_graph)
    for dataset, offset in ((historical, 0), (batch, 100)):
        coords, values, labels, *_ = _observations(len(dataset.sample_ids), offset)
        np.testing.assert_array_equal(dataset.coordinates, coords)
        np.testing.assert_array_equal(dataset.features, values)
        np.testing.assert_array_equal(dataset.labels, labels)
        assert dataset.coordinates.dtype == dataset.features.dtype == np.float64
    assert not set(historical.sample_ids) & set(batch.sample_ids)
    assert batch_graph.source_parent_sha256 == historical.dataset_sha256

    development, holdout = train_test_split(
        np.arange(len(historical.labels)),
        test_size=0.4,
        stratify=historical.labels,
        random_state=110,
    )
    expected_test, expected_calib = train_test_split(
        holdout, test_size=0.5, stratify=historical.labels[holdout], random_state=110
    )
    expected_folds = [
        (development[training], development[validation])
        for training, validation in StratifiedKFold(
            n_splits=2, shuffle=True, random_state=110
        ).split(historical.features[development], historical.labels[development])
    ]
    graphs = [test, calib]
    memberships = [expected_test, expected_calib]
    for pair, expected in zip(folds, expected_folds, strict=True):
        graphs.extend(pair)
        memberships.extend(expected)
    for graph, indices in zip(graphs, memberships, strict=True):
        np.testing.assert_array_equal(graph.original_node_indices, indices)
        sample_ids = tuple(historical.sample_ids[i] for i in indices)
        expected = build_spatial_graph(
            historical.coordinates[indices],
            sample_ids,
            connection_radius=25,
            affinity_scale=460,
            add_self_loops=True,
        )
        assert graph.sample_ids == sample_ids
        for key in ("edge_index", "edge_weight", "edge_distance", "coordinates"):
            torch.testing.assert_close(graph[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(graph.y, torch.tensor(historical.labels[indices]))
        assert graph.graph_protocol == resolve_graph_protocol(params)
    assert batch_graph.graph_protocol == resolve_graph_protocol(params)

    next_base = tmp_path / "next"
    prepare_next_cycle_data.combine_split_data(
        {"base_data": base, "prediction": incoming, "output": next_base}, params
    )
    combined, _, _, _ = load_cycle_data(next_base, resolve_graph_protocol(params))
    assert combined.sample_ids == historical.sample_ids + batch.sample_ids
    assert combined.dataset_sha256 != historical.dataset_sha256
    np.testing.assert_array_equal(
        combined.features, np.concatenate((historical.features, batch.features))
    )
    np.testing.assert_array_equal(
        combined.coordinates,
        np.concatenate((historical.coordinates, batch.coordinates)),
    )
    np.testing.assert_array_equal(
        combined.labels, np.concatenate((historical.labels, batch.labels))
    )


class RecordingModel(nn.Module):
    """Return deterministic two-class logits and capture each prepared graph."""

    def __init__(self, calls: list[Data]) -> None:
        """Initialize a scalar weight and share a forward-pass record."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = calls

    def forward(self, data: Data) -> torch.Tensor:
        """Record a copy of input data before returning logits."""
        self.calls.append(data.clone())
        score = data.x[:, 0] * self.weight
        return torch.stack((-score, score), dim=1)


def _assert_graph(
    actual: Data, original: Data, expected: np.ndarray, *, with_labels: bool = True
) -> None:
    assert actual.x.dtype == torch.float32
    np.testing.assert_allclose(actual.x.numpy(), expected.astype(np.float32), atol=1e-7)
    for key in ("edge_index", "edge_distance"):
        torch.testing.assert_close(actual[key], original[key])
    weights = original.edge_weight.clone()
    src, dst = original.edge_index
    weights[src == dst] = 2.0
    torch.testing.assert_close(actual.edge_weight, weights)
    torch.testing.assert_close(actual.edge_attr, weights[:, None])
    assert torch.all(original.edge_weight[src == dst] == 1)
    if with_labels:
        torch.testing.assert_close(actual.y, original.y)
    else:
        assert actual.y is None


def test_final_model_evaluation_and_prediction_share_frozen_state(
    generated: tuple[dict, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train via a controlled double and run the real calibration and scoring path."""
    params, base, incoming = generated
    params["device"] = "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    raw, folds, test_graph, calibration_graph = load_cycle_data(
        base, resolve_graph_protocol(params)
    )
    fitted = [
        StandardScaler().fit(raw.features[training.original_node_indices.numpy()])
        for training, _ in folds
    ]
    assert not np.array_equal(fitted[0].mean_, fitted[1].mean_)
    model_calls = []

    def model_double(_name, options):
        assert options["add_self_loops"] is False
        return RecordingModel(model_calls)

    monkeypatch.setattr(train, "get_model", model_double)
    monkeypatch.setattr(evaluate, "get_model", model_double)
    monkeypatch.setattr(predict, "get_model", model_double)
    protocol = resolve_graph_protocol(params)
    development = build_development_graph(
        raw, folds, test_graph, calibration_graph, protocol
    )
    excluded = set(test_graph.original_node_indices.tolist()) | set(
        calibration_graph.original_node_indices.tolist()
    )
    rows = development.original_node_indices.numpy()
    expected_rows = sorted(set(range(len(raw.sample_ids))) - excluded)
    assert rows.tolist() == expected_rows
    assert development.sample_ids == tuple(
        raw.sample_ids[index] for index in expected_rows
    )
    independent = build_spatial_graph(
        raw.coordinates[expected_rows],
        development.sample_ids,
        connection_radius=25,
        affinity_scale=460,
        add_self_loops=True,
    )
    torch.testing.assert_close(development.edge_index, independent.edge_index)
    torch.testing.assert_close(development.edge_weight, independent.edge_weight)
    torch.testing.assert_close(development.y, torch.tensor(raw.labels[expected_rows]))
    final_scaler = StandardScaler().fit(raw.features[expected_rows])
    params["train"]["n_epochs"] = 4
    fit_rows = []
    original_fit = StandardScaler.fit

    def record_fit(estimator, features):
        fit_rows.append(features.copy())
        return original_fit(estimator, features)

    cv_models = []

    def mock_train(pair, model, *, dataset_idx, **_):
        cv_models.append(model)
        for actual, original in zip(pair, folds[dataset_idx - 1], strict=True):
            _assert_graph(
                actual, original, fitted[dataset_idx - 1].transform(original.x.numpy())
            )
        with torch.no_grad():
            model.weight.fill_(float(dataset_idx))
        if dataset_idx == 1:
            return FoldTrainingResult(0.1, 1, 2, [0.1, 0.05])
        return FoldTrainingResult(0.2, 4, None, [0.1, 0.08, 0.06, 0.04])

    def mock_refit(graph, model, *, learning_rates, **_):
        assert all(model is not fold_model for fold_model in cv_models)
        assert model.weight.item() == 1
        assert learning_rates == [0.1, 0.08, 0.06]
        _assert_graph(
            graph, development, final_scaler.transform(raw.features[expected_rows])
        )
        with torch.no_grad():
            model.weight.fill_(3)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    monkeypatch.setattr(train, "train", mock_train)
    monkeypatch.setattr(train, "train_final", mock_refit)
    trained = tmp_path / "trained"
    train.run_training({"base_data": base, "output": trained}, params, "gcn", 1)
    assert len(fit_rows) == 3
    for actual, (training, _) in zip(fit_rows[:2], folds, strict=True):
        np.testing.assert_array_equal(actual, training.x.numpy())

    np.testing.assert_array_equal(fit_rows[-1], raw.features[expected_rows])
    pool = TrainingPartition(raw.dataset_sha256, None, rows)
    state = load_preprocessing(
        trained / "preprocessing.json",
        trained / "model.pt",
        feature_names=raw.feature_names,
        class_names=raw.class_names,
        partition=pool,
    )
    np.testing.assert_allclose(state.center, final_scaler.mean_)
    assert state.fold_index is None
    assert torch.load(trained / "model.pt", weights_only=True)["weight"].item() == 3
    artifact = json.loads((trained / "graph_binding.json").read_text())
    assert artifact["training"]["epochs"] == 3
    assert artifact["training"]["fold_best_epochs"] == [1, 4]
    assert artifact["training"]["lr_source_fold_index"] == 1
    assert artifact["training_graph_sha256"] == development.graph_sha256

    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("Feature preprocessing was refit at inference.")

    monkeypatch.setattr(StandardScaler, "fit", unexpected_fit)
    monkeypatch.setattr(eval_utils, "plot_confusion_matrix", Mock())
    monkeypatch.setattr(eval_utils, "plot_reliability_diagram", Mock())
    monkeypatch.setattr(pred_utils, "plot_confusion_matrix", Mock())
    monkeypatch.setattr(pd.DataFrame, "hist", Mock())
    monkeypatch.setattr(pred_utils.pyplot, "savefig", Mock())
    calibration_fit = []
    original_calibrate = CalibratedModel.fit

    def record_calibration(calibrator, logits, labels, weights):
        calibration_fit.append((logits.detach().clone(), labels.clone()))
        return original_calibrate(calibrator, logits, labels, weights)

    monkeypatch.setattr(CalibratedModel, "fit", record_calibration)
    evaluation = tmp_path / "evaluation"
    evaluate.run_evaluation(
        {"base": base, "model": trained, "output": evaluation}, params, "gcn"
    )
    assert len(model_calls) == 2
    for actual, original in zip(
        model_calls, (test_graph, calibration_graph), strict=True
    ):
        _assert_graph(actual, original, final_scaler.transform(original.x.numpy()))
    [(logits, labels)] = calibration_fit
    score = model_calls[1].x[:, 0] * 3
    torch.testing.assert_close(logits, torch.stack((-score, score), dim=1))
    torch.testing.assert_close(labels, calibration_graph.y)
    assert (evaluation / "metrics.json").exists()
    calibration = json.loads(
        (evaluation / "calibration_binding.json").read_text(encoding="utf-8")
    )
    assert (
        calibration["graph_binding_sha256"]
        == hashlib.sha256((trained / "graph_binding.json").read_bytes()).hexdigest()
    )

    predictions = tmp_path / "predictions"
    paths = {
        "model": trained,
        "evaluation": evaluation,
        "prediction_data": incoming,
        "output": predictions,
    }
    predict.run_prediction(paths, params, "gcn")
    original = load_data(incoming / "pred_data.pt", "Prediction")
    assert len(model_calls) == 4
    for actual in model_calls[2:]:
        _assert_graph(
            actual,
            original,
            final_scaler.transform(original.x.numpy()),
            with_labels=False,
        )
    results = pd.read_csv(predictions / "predictions.csv")
    np.testing.assert_array_equal(results["true_label"], original.y.numpy())
    assert {"predicted_label", "calib_prob_class_0", "uncalib_prob_class_0"} <= set(
        results
    )
    assert (predictions / "metrics.json").is_file()

    params["models"]["gcn"]["improved"] = False
    with pytest.raises(ValueError, match="Model graph protocol does not match"):
        predict.run_prediction(paths, params, "gcn")
    with pytest.raises(ValueError, match="Model graph protocol does not match"):
        evaluate.run_evaluation(
            {"base": base, "model": trained, "output": evaluation}, params, "gcn"
        )
    params["models"]["gcn"]["improved"] = True

    graph_binding = trained / "graph_binding.json"
    binding_bytes = graph_binding.read_bytes()
    graph_binding.unlink()
    with pytest.raises(ValueError, match="Retrain using independent graphs"):
        predict.run_prediction(paths, params, "gcn")
    graph_binding.write_bytes(binding_bytes)
    bad_schedule = json.loads(binding_bytes)
    bad_schedule["training"]["epochs"] = 1
    graph_binding.write_text(json.dumps(bad_schedule), encoding="utf-8")
    with pytest.raises(
        ValueError, match="Invalid development-pool training provenance"
    ):
        predict.run_prediction(paths, params, "gcn")
    graph_binding.write_bytes(binding_bytes)
    np.testing.assert_array_equal(
        read_raw_graph(original).features, _observations(8, 100)[1]
    )
    np.testing.assert_array_equal(
        load_cycle_data(base, resolve_graph_protocol(params))[0].features, raw.features
    )

    calibrator = evaluation / "calibrator.pt"
    old_calibrator = calibrator.read_bytes()
    calibrator.write_bytes(b"stale")
    with pytest.raises(ValueError, match="Calibrator digest mismatch"):
        predict.run_prediction(paths, params, "gcn")
    calibrator.write_bytes(old_calibrator)

    sidecar = trained / "preprocessing.json"
    sidecar.write_bytes(sidecar.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Model graph binding is stale"):
        predict.run_prediction(paths, params, "gcn")

    graph_path = trained / "graph_binding.json"
    binding = json.loads(graph_path.read_text(encoding="utf-8"))
    binding["preprocessing_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    graph_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="Calibration binding does not match"):
        predict.run_prediction(paths, params, "gcn")
    torch.save({"weight": torch.tensor(8.0)}, trained / "model.pt")
    with pytest.raises(ValueError, match="Model weights digest mismatch"):
        predict.run_prediction(paths, params, "gcn")
    assert len(model_calls) == 4


def test_legacy_and_incompatible_artifacts_fail_explicitly(
    generated: tuple[dict, Path, Path], tmp_path: Path
) -> None:
    """Reject unverifiable raw data and relabeled split artifacts."""
    params, base, incoming = generated
    with pytest.raises(ValueError, match="Legacy or incompatible raw graph"):
        read_raw_graph(Data(x=torch.ones((3, 2))))
    pred = load_data(incoming / "pred_data.pt", "Prediction")
    pred.source_parent_sha256 = "different cycle"
    torch.save(pred, incoming / "pred_data.pt")
    with pytest.raises(ValueError, match="do not match this base cycle"):
        prepare_next_cycle_data.combine_split_data(
            {"base_data": base, "prediction": incoming, "output": tmp_path / "next"},
            params,
        )
    test_graph = load_data(base / "test_data.pt", "Test")
    test_graph.original_node_indices = test_graph.original_node_indices.flip(0)
    torch.save(test_graph, base / "test_data.pt")
    with pytest.raises(ValueError, match="row order"):
        load_cycle_data(base, resolve_graph_protocol(params))


def _drift_bundle(
    generated: tuple[dict, Path, Path], tmp_path: Path, scaler_type: str | None
) -> tuple[dict, dict]:
    params, base, incoming = generated
    protocol = resolve_graph_protocol(params)
    raw, folds, test, calibration = load_cycle_data(base, protocol)
    graph = build_development_graph(raw, folds, test, calibration, protocol)
    partition = TrainingPartition(
        raw.dataset_sha256, None, graph.original_node_indices.numpy()
    )
    trained = tmp_path / "trained"
    trained.mkdir()
    training = {
        "strategy": "full_development_pool",
        "dataset_sha256": raw.dataset_sha256,
        "partition_sha256": partition.partition_sha256,
        "observation_count": graph.num_nodes,
        **final_training_plan(
            [FoldTrainingResult(0.1, 1, None, [0.1])] * len(folds), 1
        ),
    }
    publish_model(
        trained,
        RecordingModel([]),
        fit_preprocessing(raw, partition, scaler_type),
        model_graph_settings(protocol, "gcn", params),
        graph.graph_sha256,
        training,
    )
    params["analyze_drift"].update(n_permutations=2, early_stopping={"enabled": False})
    return params, {
        "base_data": base,
        "prediction_data": incoming,
        "model": trained,
        "output": tmp_path / "drift",
    }


@pytest.mark.parametrize("scaler_type", ["StandardScaler", None])
def test_drift_uses_raw_marginals_and_shared_multivariate_transform(
    generated: tuple[dict, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scaler_type: str | None,
) -> None:
    """Exercise both representations and real drift statistics with two permutations."""
    params, paths = _drift_bundle(generated, tmp_path, scaler_type)
    reference = read_raw_graph(load_data(paths["base_data"] / "base_data.pt", "Base"))
    incoming = read_raw_graph(
        load_data(paths["prediction_data"] / "pred_data.pt", "Pred")
    )
    raw_inputs = (reference.features, incoming.features)
    if scaler_type is None:
        transformed = raw_inputs
        params["analyze_drift"]["gamma"] = 0.25
    else:
        state = load_preprocessing(
            paths["model"] / "preprocessing.json",
            paths["model"] / "model.pt",
            feature_names=reference.feature_names,
            class_names=reference.class_names,
        )
        transformed = tuple((rows - state.center) / state.scale for rows in raw_inputs)
        assert transformed[1][:, 0].std() > 4 * transformed[0][:, 0].std()

    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("Drift analysis attempted to fit a feature scaler.")

    monkeypatch.setattr(StandardScaler, "fit", unexpected_fit)
    calls = {}

    def capture(name: str) -> None:
        original = getattr(AnalyzeDrift, name)

        def record(self, left, right):
            calls.setdefault(name, []).append(
                (np.asarray(left).copy(), np.asarray(right).copy())
            )
            return original(self, left, right)

        monkeypatch.setattr(AnalyzeDrift, name, record)

    for method in (
        "_median_heuristic_gamma",
        "_mmd_unbiased",
        "_energy_statistic",
        "_wasserstein_distance",
        "_plot_marginal_distributions",
        "_plot_distance_distributions",
        "_plot_pairwise_relationships",
        "_plot_pca_comparison",
    ):
        capture(method)

    original_savefig = Figure.savefig

    def small_figure(figure, path, **kwargs):
        return original_savefig(figure, path, **(kwargs | {"dpi": 30}))

    monkeypatch.setattr(Figure, "savefig", small_figure)
    monkeypatch.setattr(pyplot, "show", Mock())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        try:
            analyze_drift.run_analyze_drift(paths, params, 1)
        finally:
            pyplot.close("all")

    for name, inputs in calls.items():
        expected = raw_inputs if name == "_plot_marginal_distributions" else transformed
        for actual, wanted in zip(inputs[0], expected, strict=True):
            assert actual.dtype == np.float64
            np.testing.assert_allclose(actual, wanted)
        if name in ("_mmd_unbiased", "_energy_statistic", "_wasserstein_distance"):
            assert len(inputs) == 3
            combined = sorted(map(tuple, np.vstack(transformed)))
            for left, right in inputs[1:]:
                np.testing.assert_allclose(
                    sorted(map(tuple, np.vstack((left, right)))), combined
                )

    assert ("_median_heuristic_gamma" in calls) == (scaler_type is not None)
    assert len(calls) == (8 if scaler_type is not None else 7)
    output = paths["output"]
    metadata = json.loads((output / "drift_metadata.json").read_text(encoding="utf-8"))
    assert metadata["reference_policy"] == "all_previous_base_observations"
    assert metadata["reference_dataset_sha256"] == reference.dataset_sha256
    assert metadata["incoming_dataset_sha256"] == incoming.dataset_sha256
    assert metadata["feature_names"] == list(reference.feature_names)
    assert metadata["model_cycle"] == 1
    assert metadata["training_strategy"] == "full_development_pool"
    assert "fold_index" not in metadata
    assert metadata["scaler_type"] == scaler_type
    assert metadata["representations"] == {
        "marginal": "canonical_raw_float64",
        "multivariate": "frozen_model_transform_float64",
    }
    for filename, field in (
        ("preprocessing.json", "preprocessing_sha256"),
        ("model.pt", "weights_sha256"),
    ):
        assert (
            metadata[field]
            == hashlib.sha256((paths["model"] / filename).read_bytes()).hexdigest()
        )
    assert metadata["configuration"]["n_permutations"] == 2
    assert (
        metadata["seed_provenance"]["configured_seed"]
        == params["analyze_drift"]["seed"]
    )
    assert isinstance(metadata["seed_provenance"]["effective_seed"], int)
    if scaler_type is None:
        assert metadata["effective_gamma"] == 0.25
    else:
        values = torch.from_numpy(np.vstack(transformed))
        distances = torch.cdist(values, values)
        mask = ~torch.eye(len(values), dtype=torch.bool)
        expected_gamma = 1 / (2 * (distances[mask].median().item() + 1e-8) ** 2)
        assert metadata["effective_gamma"] == pytest.approx(expected_gamma)
    assert (output / "drift_results.txt").read_text(encoding="utf-8").count(
        "Statistic:"
    ) == 3
    assert len(list(output.glob("*.png"))) == 4
    np.testing.assert_array_equal(
        load_cycle_data(paths["base_data"], resolve_graph_protocol(params))[0].features,
        raw_inputs[0],
    )
    persisted = read_raw_graph(
        load_data(paths["prediction_data"] / "pred_data.pt", "Pred")
    )
    np.testing.assert_array_equal(persisted.features, raw_inputs[1])


def test_drift_rejects_stale_missing_and_incompatible_artifacts(
    generated: tuple[dict, Path, Path], tmp_path: Path
) -> None:
    """Reject broken bindings, reordered features and wrong source datasets."""
    params, paths = _drift_bundle(generated, tmp_path, "StandardScaler")
    sidecar = paths["model"] / "preprocessing.json"
    original = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(ValueError, match=r"preprocessing\.json.*Retrain"):
        analyze_drift.run_analyze_drift(paths, params, 1)
    sidecar.write_bytes(original)

    artifact = json.loads(original)
    artifact["state"]["feature_names"].reverse()
    sidecar.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="Feature order mismatch"):
        analyze_drift.run_analyze_drift(paths, params, 1)

    artifact = json.loads(original)
    artifact["state"]["dataset_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(artifact), encoding="utf-8")
    graph_path = paths["model"] / "graph_binding.json"
    binding = json.loads(graph_path.read_text(encoding="utf-8"))
    binding["preprocessing_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    binding["training"]["dataset_sha256"] = "0" * 64
    graph_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="reference does not match"):
        analyze_drift.run_analyze_drift(paths, params, 1)

    sidecar.write_bytes(original)
    (paths["model"] / "model.pt").write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="Model weights digest mismatch"):
        analyze_drift.run_analyze_drift(paths, params, 1)
    assert not paths["output"].exists()


def test_drift_rejects_wrong_incoming_schema_or_parent(
    generated: tuple[dict, Path, Path], tmp_path: Path
) -> None:
    """Validate incoming schema and provenance before attempting model loading."""
    params, base, incoming = generated
    paths = {
        "base_data": base,
        "prediction_data": incoming,
        "model": tmp_path / "unused",
        "output": tmp_path / "drift",
    }
    path = incoming / "pred_data.pt"
    graph = load_data(path, "Prediction")
    raw = read_raw_graph(graph)
    source = graph.source_parent_sha256
    stamp_raw_graph(
        graph, replace(raw, class_names=raw.class_names[::-1]), source_parent=source
    )
    torch.save(graph, path)
    with pytest.raises(ValueError, match="class mapping"):
        analyze_drift.run_analyze_drift(paths, params, 1)

    stamp_raw_graph(graph, raw, source_parent="0" * 64)
    torch.save(graph, path)
    with pytest.raises(ValueError, match="does not match the historical base"):
        analyze_drift.run_analyze_drift(paths, params, 1)
    assert not paths["output"].exists()


@pytest.mark.parametrize("method", ["max_sliced_wasserstein", "sinkhorn_wasserstein"])
def test_configured_wasserstein_used_for_observed_and_permuted_inputs(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the requested statistic with both original and permuted populations."""
    reference = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    incoming = reference + 2
    analyzer = AnalyzeDrift(
        Data(x=reference, unscaled_features=reference),
        Data(x=incoming, unscaled_features=incoming),
        use_max_sliced_wasserstein=method == "max_sliced_wasserstein",
        use_sinkhorn_wasserstein=method == "sinkhorn_wasserstein",
        sinkhorn_wasserstein_reg=1.0,
        n_projections=3,
        n_permutations=2,
    )
    assert analyzer.wasserstein_method == method
    original = getattr(analyzer, f"_{method}")
    spy = Mock(wraps=original)
    monkeypatch.setattr(analyzer, f"_{method}", spy)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        observed, _ = analyzer._perform_permutation_test(method)
    assert np.isfinite(observed) and observed > 0
    assert spy.call_count == 3
    torch.testing.assert_close(spy.call_args_list[0].args[0], reference)
    torch.testing.assert_close(spy.call_args_list[0].args[1], incoming)


def test_seeded_producers_replay_raw_data_and_partition_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay the real generators and compare split indices and seed metadata."""
    params = load_params()
    params["data"]["base"].update(n_samples=120, min_samples_per_class=12, n_splits=2)
    params["data"]["pred"].update(n_samples=40, seed=31)
    for stage in (generate_base_data, generate_pred_data, prepare_next_cycle_data):
        monkeypatch.setattr(stage, "export_graph_to_html", Mock())
    for stage in (generate_base_data, prepare_next_cycle_data):
        monkeypatch.setattr(stage, "export_all_graphs_to_html", Mock())
        monkeypatch.setattr(stage, "analyze_feature_discrimination", Mock())

    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        base = directory / "base"
        incoming = directory / "incoming"
        generate_base_data.prepare_base_data(base, params)
        generate_pred_data.prepare_pred_data(
            {"base_data": base, "bootstrap_data": base, "output": incoming}, params
        )
        prepare_next_cycle_data.combine_split_data(
            {"base_data": base, "prediction": incoming, "output": directory / "next"},
            params,
        )

    for relative in ("base/base_data.pt", "incoming/pred_data.pt", "next/base_data.pt"):
        a, b = (load_data(directory / relative, "Raw") for directory in (first, second))
        assert a.dataset_sha256 == b.dataset_sha256
        assert a.graph_sha256 == b.graph_sha256
        assert a.seed_provenance == b.seed_provenance
        assert a.x.device.type == "cpu"
    for name in ("hotspots.npy", "prototypes.npy"):
        np.testing.assert_array_equal(
            np.load(first / "base" / name), np.load(second / "base" / name)
        )

    protocol = resolve_graph_protocol(params)
    base_seed = params["data"]["base"]["seed"]
    for base in ("base", "next"):
        graph = load_data(first / base / "base_data.pt", "Base")
        assert graph.seed_provenance["partition"] == base_seed
        a, b = (
            load_cycle_data(directory / base, protocol) for directory in (first, second)
        )
        _, folds_a, test_a, calib_a = a
        _, folds_b, test_b, calib_b = b
        for pair_a, pair_b in zip(folds_a, folds_b, strict=True):
            for graph_a, graph_b in zip(pair_a, pair_b, strict=True):
                torch.testing.assert_close(
                    graph_a.original_node_indices, graph_b.original_node_indices
                )
        for graph_a, graph_b in ((test_a, test_b), (calib_a, calib_b)):
            torch.testing.assert_close(
                graph_a.original_node_indices, graph_b.original_node_indices
            )
    assert (
        load_data(first / "base/base_data.pt", "Base").seed_provenance["generation"]
        == base_seed
    )
    assert load_data(first / "incoming/pred_data.pt", "Prediction").seed_provenance == {
        "generation": 31
    }


def test_training_replays_independent_torch_streams(
    generated: tuple[dict, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay model initialization and training draws on each fold and refit."""
    params, base, _ = generated
    params["train"].update(seed=17, n_epochs=1)
    params["device"] = "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train, "get_model", lambda *_: nn.Linear(2, 2))
    draws = []

    def record_training(_graph, model, **_kwargs):
        draws.append((model.weight.detach().clone(), torch.rand(8)))
        return FoldTrainingResult(0.1, 1, None, [0.0005])

    monkeypatch.setattr(train, "train", record_training)
    monkeypatch.setattr(train, "train_final", record_training)
    provenances = []
    for index, cycle in enumerate((1, 1, 2)):
        output = tmp_path / f"trained_{index}"
        train.run_training({"base_data": base, "output": output}, params, "gcn", cycle)
        binding = json.loads(
            (output / "graph_binding.json").read_text(encoding="utf-8")
        )
        provenances.append(binding["training"]["seed_provenance"])

    for first, repeated in zip(draws[:3], draws[3:6], strict=True):
        torch.testing.assert_close(first[0], repeated[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], repeated[1], rtol=0, atol=0)
    assert len({tuple(weights.flatten().tolist()) for weights, _ in draws}) == 6
    assert provenances[0] == provenances[1]
    assert provenances[0]["device"] == "cpu"
    assert (
        len(set(provenances[0]["cv_seeds"]) | {provenances[0]["final_refit_seed"]}) == 3
    )
    assert provenances[0]["cv_seeds"] != provenances[2]["cv_seeds"]
