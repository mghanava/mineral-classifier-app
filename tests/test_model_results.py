"""Verify population, score weighting and read-only model diagnostics."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_spatial_predictions import spatial_fixture as spatial_fixture
from test_spatial_predictions import workspace as workspace

from dashboard.artifacts import Workspace
from dashboard.model_results import (
    BatchDiagnostics,
    calibration_change,
    classification_table,
    optimizer_applicability,
    scope_summary,
)
from dashboard.provenance import verify_workspace
from dashboard.scientific_summary import ClassificationMetrics
from dashboard.spatial_predictions import ProbabilityFamily, verify_prediction_join

APP = Path(__file__).resolve().parents[1] / "app.py"


def _find(app, kind, label):
    return next(element for element in getattr(app, kind) if element.label == label)


def test_published_tables_and_calibration_directions():
    """Preserve signed MCC and keep undefined scores separate from zero."""
    table = classification_table(
        ClassificationMetrics(0.31, None, -0.24), ClassificationMetrics(0.42, 0.2, 0)
    )
    assert table["Metric"].tolist() == ["Accuracy", "Macro F0.5", "MCC"]
    assert table["Before calibration"].tolist() == ["31.00%", "Unavailable", "-0.240"]
    assert table["After calibration"].tolist() == ["42.00%", "20.00%", "0.000"]
    assert calibration_change(0.03, 0.05) == "Increased · 2.00 pp"
    assert calibration_change(0.05, 0.03) == "Decreased · 2.00 pp"
    assert calibration_change(0.05, 0.05) == "Unchanged · 0.00 pp"
    assert calibration_change(0.05, 0.050001) == "Increased · <0.01 pp"
    assert calibration_change(None, 0.03) == "Unavailable"
    assert optimizer_applicability("isotonic", True)
    assert optimizer_applicability("dirichlet", False)
    assert optimizer_applicability("dirichlet", True) is None


def test_confusion_preserves_class_mapping_and_undefined_rates():
    """Include every class and distinguish missing denominators from zero rates."""
    counts = np.array([[2, 0, 1], [4, 0, 0], [0, 0, 0]])
    batch = BatchDiagnostics(("Quartz", "Mica", "Feldspar"), counts)
    table = batch.per_class()
    assert table["Observed support"].tolist() == [3, 4, 0]
    assert table["Precision"].tolist() == ["33.33%", "Undefined", "0.00%"]
    assert table["Recall"].tolist() == ["66.67%", "0.00%", "Undefined"]
    figure = batch.figure(dark=True)
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.layout.xaxis.title.text == "Predicted class"
    assert figure.layout.yaxis.title.text == "Observed class"
    assert tuple(figure.data[0].x) == tuple(figure.data[0].y) == batch.classes
    np.testing.assert_array_equal(figure.data[0].z, counts)


@pytest.mark.parametrize("family", list(ProbabilityFamily))
def test_gated_fixture_batch_counts(workspace, family):
    """Derive only raw counts from each exact joined probability family."""
    joined = verify_prediction_join(workspace, 1).joined
    assert joined
    batch = BatchDiagnostics.from_join(joined, family)
    assert batch.classes == joined.population.class_names
    observed = joined.population.observed_class_ids
    predicted = joined.scores(family).predicted_class_ids()
    expected = np.zeros_like(batch.counts)
    for row, column in zip(observed, predicted, strict=True):
        expected[row, column] += 1
    np.testing.assert_array_equal(batch.counts, expected)


def _open_dashboard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=25).run()
    app.switch_page("dashboard/views/results.py").run()
    _find(app, "segmented_control", "Results mode").select("Cycle explorer").run()
    return app


def _copy_quick(tmp_path):
    value = os.environ.get("MINERAL_QUICK_DEMO_WORKSPACE")
    if not value:
        pytest.skip(
            "Configure the genuine Quick demo fixture with MINERAL_QUICK_DEMO_WORKSPACE."
        )
    source = Path(value).expanduser()
    for name in ("params.yaml", "dvc.yaml", "dvc.lock", ".dvcignore"):
        if (source / name).exists():
            shutil.copy2(source / name, tmp_path / name)
    for name in ("src", "results"):
        shutil.copytree(source / name, tmp_path / name)
    (tmp_path / ".dvc").mkdir()
    shutil.copy2(source / ".dvc/config", tmp_path / ".dvc/config")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("cycle", "test_count", "development"), [(1, 92, 308), (2, 113, 378)]
)
def test_quick_demo_result_journey(
    tmp_path, monkeypatch, cycle, test_count, development
):
    """Check genuine weighted scores, fold sizes, saved figures and stale joins."""
    _copy_quick(tmp_path)
    workspace = Workspace(tmp_path)
    checked = verify_workspace(workspace)
    app = _open_dashboard(tmp_path, monkeypatch)
    app.session_state["mineral_dashboard"].result_verification = checked
    reads = []
    read = Workspace.read

    def record_read(workspace, path):
        reads.append(path)
        return read(workspace, path)

    monkeypatch.setattr(Workspace, "read", record_read)
    _find(app, "selectbox", "Learning cycle").select(cycle).run()
    _find(app, "selectbox", "Result section").select(
        "Model performance and training"
    ).run()
    assert str(reads[-1]) == f"evaluation/cycle_{cycle}/confusion_matrix.png"
    assert any(f"{test_count} observations" in item.value for item in app.caption)
    assert list(app.dataframe[0].value["Metric"]) == ["Accuracy", "Macro F0.5", "MCC"]
    reads.clear()
    _find(app, "segmented_control", "Classification population").select(
        "Incoming prediction batch"
    ).run()
    assert not reads
    assert len(app.metric) == 3
    published = [item.value for item in app.metric]
    joined = app.session_state["_classification_join"][1].joined
    raw_counts = BatchDiagnostics.from_join(joined, ProbabilityFamily.CALIBRATED).counts
    published_summary, _ = scope_summary(checked, cycle, "predict")
    assert published[0] == f"{published_summary.prediction.accuracy:.2%}"
    assert published[0] != f"{raw_counts.trace() / raw_counts.sum():.2%}"
    _find(app, "segmented_control", "Batch probability family").select(
        ProbabilityFamily.UNCALIBRATED
    ).run()
    assert [item.value for item in app.metric] == published
    assert app.dataframe[0].value["Observed support"].sum() == 100
    assert len(app.get("plotly_chart")) == 1
    _find(app, "segmented_control", "Scientific result view").select("Training").run()
    training, _ = scope_summary(checked, cycle, "train")
    assert training.model.development_count == development
    folds = app.dataframe[0].value
    assert len(folds) == 3
    assert (folds["Train"] + folds["Validation"] == development).all()
    assert any("starts fresh" in item.value for item in app.caption)
    reads.clear()
    _find(app, "selectbox", "Training fold").select(
        Path(f"trained/cycle_{cycle}/training_plots_dataset_2.png")
    ).run()
    assert reads == [Path(f"trained/cycle_{cycle}/training_plots_dataset_2.png")]
    reads.clear()
    _find(app, "selectbox", "Result section").select("Probability calibration").run()
    assert reads == [Path(f"evaluation/cycle_{cycle}/reliability_diagram.png")]
    assert any("10 reporting bins" in item.value for item in app.markdown)
    assert any("Same final-test" in item.value for item in app.caption)
    reads.clear()
    _find(app, "toggle", "Calibration optimizer details").set_value(True).run()
    assert len(reads) == 1 and reads[0].name.endswith("_loss_and_gradient.png")
    assert _find(app, "selectbox", "Learning cycle").value == cycle
    _find(app, "selectbox", "Result section").select(
        "Model performance and training"
    ).run()
    _find(app, "segmented_control", "Classification population").select(
        "Incoming prediction batch"
    ).run()
    assert app.get("plotly_chart")
    csv = tmp_path / f"results/prediction/cycle_{cycle}/predictions.csv"
    csv.write_bytes(csv.read_bytes() + b"\n")
    app.run()
    assert not app.get("plotly_chart") and not app.metric and not app.exception
