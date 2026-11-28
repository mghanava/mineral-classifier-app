"""Cover the drift reader, evidence-scoped comparison and Results navigation."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_model_results import _copy_quick, _find, _open_dashboard

from dashboard.artifacts import Workspace
from dashboard.cycle_comparison import (
    POPULATIONS,
    ComparisonCycle,
    ComparisonView,
    chart_incompatibility,
    drift_context_changes,
)
from dashboard.drift_results import DIAGNOSTICS, NATIVE, SETTINGS, shared_histogram
from dashboard.drift_summary import (
    MODEL_REPRESENTATION,
    RAW_REPRESENTATION,
    REPRESENTATIONS,
    parse_drift_report,
    read_drift_summary,
)
from dashboard.model_results import scope_summary
from dashboard.provenance import verify_workspace
from dashboard.scientific_summary import (
    CalibrationMetadata,
    ClassificationMetrics,
    EvaluationMetrics,
)
from dashboard.scientific_summary import TestMetrics as PublishedTestMetrics


def _report(method="wasserstein"):
    """Make a producer-format report with all three cutoff boundary cases."""
    return (
        "".join(
            f"{key.upper()} Statistic: -1.25e-5\np-value: {p}\n"
            f"{finding} detected between reference and prediction datasets using {key} method.\n"
            for key, p, finding in (
                ("mmd", 0.04999, "Domain shift"),
                ("energy", 0.05, "No domain shift"),
                (method, 0.05001, "No domain shift"),
            )
        )
        + "\n"
    )


def _metadata(sliced=None, sinkhorn=None):
    return {
        "version": 1,
        "representations": REPRESENTATIONS,
        "model_cycle": 1,
        "reference_dataset_sha256": "a" * 64,
        "incoming_dataset_sha256": "b" * 64,
        "preprocessing_sha256": "c" * 64,
        "feature_names": ["Quartz", "Mica"],
        "scaler_type": None,
        "seed_provenance": {"configured_seed": None, "effective_seed": None},
        "configuration": {
            "n_permutations": 19,
            "gamma": None,
            "early_stopping": {},
            "use_max_sliced_wasserstein": sliced,
            "use_sinkhorn_wasserstein": sinkhorn,
            "n_projections": 100,
            "sinkhorn_wasserstein_reg": 0.1,
        },
    }


def _read(tmp_path, expected, document, method="wasserstein"):
    metadata = tmp_path / "metadata.json"
    report = tmp_path / "report.txt"
    metadata.write_text(json.dumps(document), encoding="utf-8")
    report.write_text(_report(method), encoding="utf-8")
    return read_drift_summary(
        metadata, report, expected=expected, reference_count=440, incoming_count=100
    )


def _document(expected, method="wasserstein"):
    return {
        **expected,
        "effective_gamma": 0.11,
        "effective_early_stopping": {
            "enabled": False,
            "confidence_threshold": 0.01,
            "min_permutations": 100,
        },
        "wasserstein_method": method,
    }


@pytest.mark.parametrize(
    ("sliced", "sinkhorn", "method", "detail"),
    [
        (None, None, "wasserstein", "squared Euclidean"),
        (False, True, "sinkhorn_wasserstein", "regularization"),
        (True, True, "max_sliced_wasserstein", "projections"),
    ],
)
def test_wasserstein_and_report_threshold(tmp_path, sliced, sinkhorn, method, detail):
    """Check variant selection, the 1D fallback and the strict report cutoff."""
    expected = _metadata(sliced, sinkhorn)
    drift = _read(tmp_path, expected, _document(expected, method), method)
    assert drift.wasserstein == method and detail in drift.wasserstein_context
    assert [item.detected for item in drift.tests] == [True, False, False]
    assert drift.tests[0].statistic == -1.25e-5
    assert not drift.early_stopping.enabled
    expected["feature_names"] = ["Quartz"]
    one_dim = _read(tmp_path, expected, _document(expected, method), method)
    assert "shared 1D transport" in one_dim.wasserstein_context
    assert one_dim.projections is one_dim.regularization is None


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        ("ENERGY Statistic:", "WRONG Statistic:"),
        ("0.04999", "1e999"),
        ("0.04999", "nan"),
        ("0.04999", "-0.01"),
        ("0.04999", "1.01"),
        ("0.04999", "0.05"),
        ("-1.25e-5", "inf"),
        ("p-value: 0.05\n", ""),
        ("MMD", "ENERGY"),
    ],
)
def test_reject_malformed_report(original, changed):
    """Reject missing fields, invalid numbers and mismatched decisions."""
    with pytest.raises(ValueError):
        parse_drift_report(_report().replace(original, changed, 1), "wasserstein")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("effective_gamma", None),
        ("effective_gamma", float("inf")),
        ("effective_gamma", True),
        ("effective_early_stopping", {"enabled": False}),
        ("wasserstein_method", "sinkhorn_wasserstein"),
        ("reference_dataset_sha256", "different reference"),
        ("incoming_dataset_sha256", "different batch"),
        ("representations", {"marginal": "scaled", "multivariate": "raw"}),
        ("feature_names", ["Mica", "Quartz"]),
    ],
)
def test_reject_malformed_or_unbound_metadata(tmp_path, key, value):
    """Reject inconsistent, nonfinite and mismatched binding fields."""
    expected = _metadata()
    document = deepcopy(_document(expected))
    document[key] = value
    with pytest.raises(ValueError):
        _read(tmp_path, expected, document)


def test_missing_drift_artifacts_and_fields(tmp_path):
    """Require the published report and its effective metadata."""
    expected = _metadata()
    document = _document(expected)
    del document["effective_gamma"]
    with pytest.raises(ValueError, match="gamma"):
        _read(tmp_path, expected, document)
    metadata, report = tmp_path / "metadata.json", tmp_path / "report.txt"
    metadata.write_text(json.dumps(_document(expected)), encoding="utf-8")
    report.unlink()
    with pytest.raises(FileNotFoundError):
        read_drift_summary(
            metadata, report, expected=expected, reference_count=440, incoming_count=100
        )


def test_raw_histogram_proportions_and_diagnostic_representations():
    """Use identical bins and normalize each cohort by its own size."""
    refs, batch = np.array([0, 0, 1]), np.array([1, 2])
    edges, reference, incoming = shared_histogram(refs, batch)
    assert len(edges) == 31 and edges[[0, -1]].tolist() == [0, 2]
    np.testing.assert_allclose(reference * len(refs), np.histogram(refs, bins=edges)[0])
    np.testing.assert_allclose(
        incoming * len(batch), np.histogram(batch, bins=edges)[0]
    )
    assert reference.sum() == incoming.sum() == 1
    assert shared_histogram(np.ones(4), np.ones(2))[0].size == 31
    caption = DIAGNOSTICS["Original raw feature distributions"][1]
    assert RAW_REPRESENTATION in caption and "transformed features" in caption
    assert all(
        MODEL_REPRESENTATION in text
        for name, (_, text) in DIAGNOSTICS.items()
        if name != "Original raw feature distributions"
    )


def test_comparison_compatibility_and_metric_semantics():
    """Require matched semantics and preserve signed and population-specific scores."""
    evaluation = EvaluationMetrics(
        PublishedTestMetrics(ClassificationMetrics(0.2, 0.3, -0.4), 0.51, 0.6),
        PublishedTestMetrics(ClassificationMetrics(0.3, 0.4, -0.5), 0.32, 0.4),
    )
    one = ComparisonCycle(
        1,
        "Verified matching",
        evaluation=evaluation,
        prediction=ClassificationMetrics(0.3, 0.4, -0.25),
        calibration=CalibrationMetadata("dirichlet", "split", "calibrator", 10),
        features=("feature",),
        classes=("Quartz", "Mica"),
    )
    two = replace(one, cycle=2)
    assert one.classification(POPULATIONS[0], "mcc") == -0.4
    assert one.classification(POPULATIONS[1], "mcc") == -0.5
    assert one.classification(POPULATIONS[2], "mcc") == -0.25
    assert one.calibration_errors("ece") == (0.51, 0.32)
    assert one.calibration_errors("mce") == (0.6, 0.4)
    classification, calibration = (
        ComparisonView.CLASSIFICATION,
        ComparisonView.CALIBRATION,
    )
    mcc = ((-0.4, -0.5),)
    assert chart_incompatibility(one, two, classification, mcc) is None
    assert chart_incompatibility(one, one, classification, mcc)
    assert "schema" in chart_incompatibility(
        one, replace(two, classes=two.classes[::-1]), classification, mcc
    )
    assert "verified values" in chart_incompatibility(
        one, two, classification, ((None, -0.5),)
    )
    assert chart_incompatibility(one, two, calibration, ((0.51, 0.32),)) is None
    other_bins = replace(two, calibration=replace(two.calibration, reporting_bins=20))
    assert "bin counts differ" in chart_incompatibility(
        one, other_bins, calibration, ((0.51, 0.32),)
    )
    historical = ComparisonCycle.from_evidence(3, 2, None)
    assert (
        historical.status.startswith("Historical")
        and historical.counts() == (None,) * 6
    )
    assert "verified evidence" in chart_incompatibility(
        one, historical, classification, mcc
    )


@pytest.mark.integration
def test_genuine_drift_comparison_and_lazy_navigation(tmp_path, monkeypatch):
    """Reuse both real reports and exercise lazy diagnostic and comparison routing."""
    _copy_quick(tmp_path)
    workspace = Workspace(tmp_path)
    checked = verify_workspace(workspace)
    assert checked.latest_complete_matching_cycle == 2
    first, second = (ComparisonCycle.from_evidence(i, 2, checked) for i in (1, 2))
    assert first.counts() == (440, 308, 40, 92, 100, 540)
    assert second.counts() == (540, 378, 49, 113, 100, 640)
    for cycle, counts, expected_p in (
        (1, (440, 100), [11 / 19, 14 / 19, 1 / 19]),
        (2, (540, 100), [1, 1, 8 / 19]),
    ):
        summary, _ = scope_summary(checked, cycle, "analyze_drift")
        drift = summary.drift
        assert drift.cycle == cycle
        assert (drift.reference_count, drift.incoming_count) == counts
        assert drift.reference_sha256 != drift.incoming_sha256
        assert drift.representation == REPRESENTATIONS["multivariate"]
        assert drift.wasserstein == "wasserstein" and drift.budget == 19
        assert drift.early_stopping.minimum == 100
        assert drift.effective_seed in (2345895759, 837289528)
        assert [test.p_value for test in drift.tests] == pytest.approx(expected_p)
        assert all(not test.detected for test in drift.tests)
        assert drift.tests[0].statistic < 0
    assert drift_context_changes(first.drift, second.drift) == [
        "Fitted preprocessing",
        "Effective RBF gamma",
        "Permutation seed provenance",
    ]

    app = _open_dashboard(tmp_path, monkeypatch)
    app.session_state["mineral_dashboard"].result_verification = checked
    reads = []
    read = Workspace.read

    def record(workspace, path):
        reads.append(path)
        return read(workspace, path)

    monkeypatch.setattr(Workspace, "read", record)
    _find(app, "selectbox", "Learning cycle").select(1).run()
    _find(app, "selectbox", "Result section").select("Distribution drift").run()
    assert not reads and not app.get("plotly_chart")
    assert app.dataframe[0].value["Finding"].tolist() == ["Not detected"] * 3
    _find(app, "selectbox", "Supporting diagnostic").select(SETTINGS).run()
    assert not reads
    _find(app, "selectbox", "Supporting diagnostic").select(
        "Within- and cross-set distances"
    ).run()
    assert reads == [Path("drift_analysis/cycle_1/distance_distributions.png")]
    _find(app, "selectbox", "Supporting diagnostic").select(NATIVE).run()
    assert len(reads) == 1 and len(app.get("plotly_chart")) == 1
    raw_snapshot = app.session_state["_drift_raw_snapshot"]
    _find(app, "selectbox", "Raw feature").select("Feature_3").run()
    assert app.session_state["_drift_raw_snapshot"] is raw_snapshot
    _find(app, "segmented_control", "Results mode").select("Compare two cycles").run()
    assert len(app.get("plotly_chart")) == 1 and len(reads) == 1
    _find(app, "segmented_control", "Comparison view").select(
        ComparisonView.CALIBRATION
    ).run()
    assert "ECE · Before calibration" in app.dataframe[0].value["Measure"].tolist()
    plotted = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    assert [trace["marker"]["color"] for trace in plotted] == ["#BC591E", "#0072B2"]
    assert plotted[0]["text"] == ["7.12%", "2.83%"]
    assert plotted[1]["text"] == ["9.07%", "4.02%"]
    _find(app, "selectbox", "Learning cycle A").select(2).run()
    assert not app.get("plotly_chart")
    _find(app, "selectbox", "Learning cycle B").select(1).run()
    app.switch_page("dashboard/views/configure.py").run()
    app.switch_page("dashboard/views/results.py").run()
    assert _find(app, "selectbox", "Learning cycle A").value == 2
    assert _find(app, "selectbox", "Learning cycle B").value == 1
    assert len(app.get("plotly_chart")) == 1 and len(reads) == 1
    _find(app, "segmented_control", "Results mode").select("Cycle explorer").run()
    assert _find(app, "selectbox", "Learning cycle").value == 1
    _find(app, "selectbox", "Supporting diagnostic").select(NATIVE).run()
    report = tmp_path / "results/drift_analysis/cycle_1/drift_results.txt"
    report.unlink()
    app.run()
    assert not app.metric and not app.get("plotly_chart") and not app.exception
    assert any("Workspace evidence changed" in item.value for item in app.caption)
    assert "_drift_raw_snapshot" not in app.session_state
