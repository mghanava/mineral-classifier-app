"""Regression checks for scientific figure labels and their underlying data."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, fbeta_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight
from torch_geometric.data import Data

from src.utilities import eval_utils, pred_utils
from src.utilities.calibration_utils import CalibrationMetrics
from src.utilities.cycles_performance_utils import (
    load_metrics_from_cycles,
    plot_metrics,
)
from src.utilities.train_utils import plot_training


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close fixture figures after each test."""
    yield
    pyplot.close("all")


def test_training_validation_label_and_series() -> None:
    """Display macro F0.5 without changing the validation series."""
    scores = [0.3, 0.55]
    figure = plot_training(*([[1.0, 0.5]] * 3), scores, *([[1.0, 0.5]] * 4))
    line = figure.axes[1].lines[0]
    assert line.get_label() == "Validation Macro F0.5"
    assert list(line.get_ydata()) == scores


@pytest.mark.parametrize("include_calibrated", [False, True])
def test_reliability_labels_and_bin_weights(include_calibrated: bool) -> None:
    """Place the confidence labels correctly and plot sums of sample weights."""
    metrics = CalibrationMetrics(n_bins=4)
    probabilities = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.1, 0.9]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([1.0, 2.0, 4.0])
    stats = metrics.calculate_metrics(
        probabilities, labels, weights, adaptive_binning=False
    )
    calibrated = metrics.calculate_metrics(
        torch.tensor([[0.6, 0.4], [0.2, 0.8], [0.1, 0.9]]),
        labels,
        weights,
        adaptive_binning=False,
    )
    figure = eval_utils.plot_reliability_diagram(
        stats, calibrated if include_calibrated else None
    )
    primary, secondary = figure.axes
    diagonal = primary.lines[0]
    assert list(diagonal.get_xdata()) == list(diagonal.get_ydata()) == [0, 1]
    annotations = {item.get_text(): item.get_position() for item in primary.texts}
    assert set(annotations) == {"Overconfident", "Underconfident"}
    x, y = annotations["Overconfident"]
    assert y < x
    x, y = annotations["Underconfident"]
    assert y > x

    assert secondary.get_ylabel() == "Sum of Sample Weights per Bin"
    heights = [bar.get_height() for bar in secondary.patches]
    assert heights == ([3, 4, 1, 6] if include_calibrated else [3, 4])
    assert stats.ece == pytest.approx(0.2)
    assert stats.mce == pytest.approx(1 / 3)


class FixedLogitModel(torch.nn.Linear):
    """Provide a model with fixed logits and a device-bearing parameter."""

    def forward(self, data: Data) -> torch.Tensor:
        """Return the fixture logits stored in the graph features."""
        return data.x


def test_metric_artifact_contracts_and_cycle_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep published names, keys, columns, and scores stable."""
    labels = torch.tensor([0, 0, 0, 1])
    probabilities = torch.tensor([[0.65, 0.35], [0.35, 0.65], [0.2, 0.8], [0.05, 0.95]])
    graph = Data(x=probabilities.log(), y=labels)
    model = FixedLogitModel(2, 2)
    pipeline = Mock()
    pipeline.predict_from_logits.return_value = probabilities
    pipeline.predict.return_value = probabilities
    monkeypatch.setattr(eval_utils, "CalibrationPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(
        pred_utils, "CalibrationPipeline", Mock(load=Mock(return_value=pipeline))
    )

    figures = {}

    def record_figure(path, **kwargs):
        figures[Path(path).name] = pyplot.gcf()

    monkeypatch.setattr(pyplot, "savefig", record_figure)
    evaluation = tmp_path / "evaluation" / "cycle_1"
    prediction = tmp_path / "prediction" / "cycle_1"
    prediction.mkdir(parents=True)

    eval_utils.evaluate_with_calibration(
        graph,
        graph,
        model,
        calibration_method="isotonic",
        initial_temperature=1.0,
        n_bins=2,
        class_names=["A", "B"],
        lr=0.1,
        n_epochs=1,
        weight_decay_adam_optimizer=0.0,
        reg_lambda=0.0,
        reg_mu=0.0,
        eps=1e-8,
        factor_learning_rate_scheduler=0.5,
        patience_learning_rate_scheduler=1,
        patience_early_stopping=1,
        min_delta_early_stopping=0.0,
        save_path=str(evaluation),
    )
    pred_utils.prediction(graph, model, "unused.pt", ["A", "B"], str(prediction))
    assert set(figures) == {
        "reliability_diagram.png",
        "confusion_matrix.png",
        "histograms.png",
        "confussion_matrix_.png",
    }
    assert {path.name for path in evaluation.iterdir()} == {"metrics.json"}
    assert {path.name for path in prediction.iterdir()} == {
        "metrics.json",
        "predictions.csv",
    }
    assert pd.read_csv(prediction / "predictions.csv").columns.tolist() == [
        "calib_prob_class_0",
        "calib_prob_class_1",
        "uncalib_prob_class_0",
        "uncalib_prob_class_1",
        "predicted_label",
        "true_label",
        "calibrated_entropy",
        "uncalibrated_entropy",
    ]

    weights = compute_sample_weight("balanced", labels)
    predicted = probabilities.argmax(dim=1)
    expected = {
        "acc": accuracy_score(labels, predicted, sample_weight=weights),
        "f1": fbeta_score(
            labels, predicted, beta=0.5, average="macro", sample_weight=weights
        ),
        "mcc": matthews_corrcoef(labels, predicted, sample_weight=weights),
    }
    assert expected["f1"] != pytest.approx(
        fbeta_score(labels, predicted, beta=1, average="macro", sample_weight=weights)
    )
    assert json.loads((prediction / "metrics.json").read_text()) == pytest.approx(
        expected
    )
    report = json.loads((evaluation / "metrics.json").read_text())
    assert set(report) == {"calibrated", "uncalibrated"}
    stats = CalibrationMetrics(n_bins=2).calculate_metrics(
        probabilities, labels, torch.tensor(weights, dtype=torch.float32)
    )
    for state in report.values():
        assert state == pytest.approx({**expected, "ece": stats.ece, "mce": stats.mce})

    cycles = load_metrics_from_cycles(tmp_path, [1])
    assert plot_metrics(cycles, tmp_path) == tmp_path / "cycles_performance.png"
    for axis in figures["cycles_performance.png"].axes:
        assert [item.get_text() for item in axis.get_legend().texts] == [
            "ACCURACY",
            "Macro F0.5",
            "MCC",
        ]
        assert list(axis.lines[1].get_ydata()) == pytest.approx([expected["f1"]])
