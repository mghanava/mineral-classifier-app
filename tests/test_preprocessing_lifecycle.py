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
from src.utilities.data_utils import construct_graph
from src.utilities.drift_detection_utils import AnalyzeDrift
from src.utilities.general_utils import load_data, load_params
from src.utilities.model_bundle import publish_model
from src.utilities.preprocessing import (
    TrainingPartition,
    fit_preprocessing,
    load_preprocessing,
)
from src.utilities.raw_graphs import load_cycle_data, read_raw_graph, stamp_raw_graph


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
    historical, folds, _, _ = load_cycle_data(base)
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

    legacy_topology, legacy_folds, _, _ = construct_graph(
        historical.coordinates,
        historical.features,
        historical.labels,
        connection_radius=25,
        n_splits=2,
        test_size=0.4,
        calib_size=0.5,
        seed=110,
        scaler=None,
    )
    base_graph = load_data(base / "base_data.pt", "Base")
    torch.testing.assert_close(base_graph.edge_index, legacy_topology.edge_index)
    torch.testing.assert_close(base_graph.edge_weight, legacy_topology.edge_weight)
    for actual, expected in zip(folds, legacy_folds, strict=True):
        for graph, legacy in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                graph.original_node_indices, legacy.original_node_indices
            )
            torch.testing.assert_close(graph.edge_index, legacy.edge_index)
            torch.testing.assert_close(graph.edge_weight, legacy.edge_weight)

    next_base = tmp_path / "next"
    prepare_next_cycle_data.combine_split_data(
        {"base_data": base, "prediction": incoming, "output": next_base}, params
    )
    combined, _, _, _ = load_cycle_data(next_base)
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
    for key in ("edge_index", "edge_attr", "edge_weight"):
        torch.testing.assert_close(actual[key], original[key])
    if with_labels:
        torch.testing.assert_close(actual.y, original.y)
    else:
        assert actual.y is None


def test_fold_winner_evaluation_and_prediction_share_frozen_state(
    generated: tuple[dict, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train via a controlled double and run the real calibration and scoring path."""
    params, base, incoming = generated
    raw, folds, test_graph, calibration_graph = load_cycle_data(base)
    fitted = [
        StandardScaler().fit(raw.features[training.original_node_indices.numpy()])
        for training, _ in folds
    ]
    assert not np.array_equal(fitted[0].mean_, fitted[1].mean_)
    model_calls = []
    monkeypatch.setattr(train, "get_model", lambda *_: RecordingModel(model_calls))
    monkeypatch.setattr(evaluate, "get_model", lambda *_: RecordingModel(model_calls))
    monkeypatch.setattr(predict, "get_model", lambda *_: RecordingModel(model_calls))
    fit_rows = []
    original_fit = StandardScaler.fit

    def record_fit(estimator, features):
        fit_rows.append(features.copy())
        return original_fit(estimator, features)

    def mock_train(pair, model, *, dataset_idx, **_):
        for actual, original in zip(pair, folds[dataset_idx - 1], strict=True):
            _assert_graph(
                actual, original, fitted[dataset_idx - 1].transform(original.x.numpy())
            )
        with torch.no_grad():
            model.weight.fill_(float(dataset_idx))
        return model, float(dataset_idx)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    monkeypatch.setattr(train, "train", mock_train)
    trained = tmp_path / "trained"
    train.run_training({"base_data": base, "output": trained}, params, "gcn", 1)
    assert len(fit_rows) == 2
    for actual, (training, _) in zip(fit_rows, folds, strict=True):
        np.testing.assert_array_equal(actual, training.x.numpy())

    first_fold = TrainingPartition(
        raw.dataset_sha256, 0, folds[0][0].original_node_indices.numpy()
    )
    state = load_preprocessing(
        trained / "preprocessing.json",
        trained / "model.pt",
        feature_names=raw.feature_names,
        class_names=raw.class_names,
        partition=first_fold,
    )
    np.testing.assert_allclose(state.center, fitted[0].mean_)
    assert torch.load(trained / "model.pt", weights_only=True)["weight"].item() == 1

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
        _assert_graph(actual, original, fitted[0].transform(original.x.numpy()))
    [(logits, labels)] = calibration_fit
    score = model_calls[1].x[:, 0]
    torch.testing.assert_close(logits, torch.stack((-score, score), dim=1))
    torch.testing.assert_close(labels, calibration_graph.y)
    assert (evaluation / "metrics.json").exists()
    assert (evaluation / "calibration_binding.json").exists()

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
            actual, original, fitted[0].transform(original.x.numpy()), with_labels=False
        )
    results = pd.read_csv(predictions / "predictions.csv")
    np.testing.assert_array_equal(results["true_label"], original.y.numpy())
    assert {"predicted_label", "calib_prob_class_0", "uncalib_prob_class_0"} <= set(
        results
    )
    assert (predictions / "metrics.json").is_file()
    np.testing.assert_array_equal(
        read_raw_graph(original).features, _observations(8, 100)[1]
    )
    np.testing.assert_array_equal(load_cycle_data(base)[0].features, raw.features)

    calibrator = evaluation / "calibrator.pt"
    old_calibrator = calibrator.read_bytes()
    calibrator.write_bytes(b"stale")
    with pytest.raises(ValueError, match="Calibrator digest mismatch"):
        predict.run_prediction(paths, params, "gcn")
    calibrator.write_bytes(old_calibrator)

    sidecar = trained / "preprocessing.json"
    sidecar.write_bytes(sidecar.read_bytes() + b"\n")
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
        load_cycle_data(base)


def _drift_bundle(
    generated: tuple[dict, Path, Path], tmp_path: Path, scaler_type: str | None
) -> tuple[dict, dict]:
    params, base, incoming = generated
    raw, folds, _, _ = load_cycle_data(base)
    partition = TrainingPartition(
        raw.dataset_sha256, 0, folds[0][0].original_node_indices.numpy()
    )
    trained = tmp_path / "trained"
    trained.mkdir()
    publish_model(
        trained, RecordingModel([]), fit_preprocessing(raw, partition, scaler_type)
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
        load_cycle_data(paths["base_data"])[0].features, raw_inputs[0]
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
