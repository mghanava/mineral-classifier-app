"""Check population semantics, verified projections and Results navigation."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import provenance
from dashboard.artifacts import Workspace
from dashboard.dvc_evidence import WorkspaceStamp
from dashboard.experiment_summary import (
    ExperimentSummary,
    Metric,
    growth_figure,
    metric_series,
    trajectory_figure,
)
from dashboard.overview import ece_change
from dashboard.provenance import (
    AppliedConfiguration,
    Availability,
    Correspondence,
    CycleEvidence,
    Evidence,
    Verification,
)
from dashboard.scientific_summary import (
    BasePopulation,
    CycleSummary,
    classification_metrics,
    evaluation_metrics,
)
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml

ROOT = Path(__file__).resolve().parents[1]
MATCH = Evidence(Availability.AVAILABLE, Correspondence.MATCHING, True)
STALE = Evidence(Availability.AVAILABLE, Correspondence.CONFLICT, True)


def _base(cycle):
    return BasePopulation(cycle, 410 + 80 * cycle, 280 + 80 * cycle, 39, 91)


def _cycle(cycle, evidence=MATCH):
    return CycleEvidence(
        cycle,
        evidence,
        CycleSummary(
            _base(cycle - 1),
            80,
            _base(cycle),
            evaluation_metrics(
                {
                    "uncalibrated": {
                        "acc": 0.4,
                        "f1": 0.2,
                        "mcc": -0.5,
                        "ece": 0.2,
                        "mce": 0.23,
                    },
                    "calibrated": {
                        "acc": 0.5,
                        "f1": 0.3,
                        "mcc": -0.4,
                        "ece": 0.1,
                        "mce": 0.3,
                    },
                }
            ),
            classification_metrics({"acc": 0.6, "f1": 0.25, "mcc": -0.2}),
        ),
    )


def _verified(root, declared, cycles, *, applied=None):
    return Verification(
        root=root,
        checked_at=datetime.now(UTC),
        elapsed=0.1,
        applied=applied
        or AppliedConfiguration(
            "fingerprint", "gcn", "cpu", declared, 3, 400, 100, "dirichlet"
        ),
        discovered_cycles=tuple(item.cycle for item in cycles),
        bootstrap=_base(0),
        cycles=tuple(cycles),
        stages=(
            ("bootstrap", MATCH),
            *(
                (f"prepare_next_cycle_data_{item.cycle}", item.evidence)
                for item in cycles
            ),
        ),
    )


def test_summary_uses_latest_matching_cycle_and_leaves_missing_or_stale_gaps(tmp_path):
    """Reject partial, historical and stale data even when local values are readable."""
    partial = Evidence(Availability.PARTIAL, Correspondence.MATCHING)
    checked = _verified(
        tmp_path,
        6,
        [_cycle(1), _cycle(3), _cycle(4, STALE), _cycle(5, partial), _cycle(7)],
    )
    summary = ExperimentSummary.from_verification(checked)
    assert summary.latest == 3
    assert set(summary.learning) == {1, 3}
    assert set(summary.snapshots) == {0, 1, 3}
    domain = checked.declared_cycles
    series = metric_series(summary, Metric.MCC, domain)
    figure = trajectory_figure(Metric.MCC, domain, series)
    assert list(figure.data[0].x) == list(range(1, 7))
    assert list(figure.data[0].y) == [-0.4, None, -0.4, None, None, None]
    assert list(figure.data[1].y) == [-0.2, None, -0.2, None, None, None]
    assert figure.layout.yaxis.range == (-1, 1)
    assert figure.layout.xaxis.dtick == 1
    assert not figure.data[0].connectgaps and not figure.data[1].connectgaps
    assert figure.data[0].marker.symbol != figure.data[1].marker.symbol
    assert figure.data[0].line.dash != figure.data[1].line.dash
    assert figure.data[0].text[0] == "-0.4 · n = 91"
    assert figure.data[1].text[0] == "-0.2 · n = 80"


@pytest.mark.parametrize(
    ("metric", "test_value", "second_value", "second_population", "second_count"),
    [
        (Metric.ACCURACY, 0.5, 0.6, "Prediction batch · calibrated", 80),
        (Metric.MACRO_F05, 0.3, 0.25, "Prediction batch · calibrated", 80),
        (Metric.ECE, 0.1, 0.2, "Final test · before calibration", 91),
        (Metric.MCE, 0.3, 0.23, "Final test · before calibration", 91),
    ],
)
def test_trajectory_metric_and_population_semantics(
    tmp_path, metric, test_value, second_value, second_population, second_count
):
    """Never reuse the incoming cohort for a calibration comparison."""
    checked = _verified(tmp_path, 1, [_cycle(1)])
    summary = ExperimentSummary.from_verification(checked)
    domain = checked.declared_cycles
    first, second = metric_series(summary, metric, domain)
    assert first.name == (
        "Final test · after calibration"
        if metric.calibration_error
        else "Final test · calibrated"
    )
    assert first.values == [test_value] and first.counts == [91]
    assert second.name == second_population
    assert second.values == [second_value] and second.counts == [second_count]
    figure = trajectory_figure(metric, domain, (first, second))
    assert len(figure.data) == 2 and list(figure.data[0].x) == [1]
    assert "markers" in figure.data[0].mode
    assert list(figure.layout.xaxis.ticktext) == ["Cycle 1"]
    if metric == Metric.ECE:
        assert figure.layout.yaxis.tickformat == ".0%"
        assert figure.data[0].text[0] == "10.0000% · n = 91"
        assert figure.data[1].text[0] == "20.0000% · n = 91"
        assert list(figure.data[0].y) == [0.1]


@pytest.mark.parametrize("declared", [0, 1, 40])
def test_growth_uses_verified_realized_counts_and_stays_bounded(tmp_path, declared):
    """Show bootstrap and post-append totals even if a model diagnostic is partial."""
    cycles = [_cycle(cycle) for cycle in range(1, declared + 1)]
    checked = _verified(tmp_path, declared, cycles)
    if declared:
        cycles[-1] = replace(cycles[-1], evidence=Evidence(Availability.PARTIAL))
        checked = replace(checked, cycles=tuple(cycles))
    summary = ExperimentSummary.from_verification(checked)
    growth = growth_figure(summary, declared)
    assert list(growth.data[0].x) == list(range(declared + 1))
    assert growth.data[0].y[0] == 410
    assert growth.data[0].y[-1] == 410 + 80 * declared
    assert growth.layout.height == 312 and len(growth.data) == 1
    assert summary.latest == (declared - 1 if declared > 1 else None)
    assert growth.layout.xaxis.ticktext[0] == "Bootstrap"
    if declared:
        assert growth.layout.xaxis.tickvals[-1] == declared
        assert growth.layout.xaxis.ticktext[-1] == "Final append"
    assert len(growth.layout.xaxis.tickvals) <= 7


def test_growth_distinguishes_zero_and_missing_snapshots(tmp_path):
    """Leave missing snapshots blank and keep a verified zero value."""
    checked = _verified(tmp_path, 3, [_cycle(1), _cycle(3, STALE)])
    checked = replace(checked, bootstrap=BasePopulation(0, 0, 0, 0, 0))
    growth = growth_figure(ExperimentSummary.from_verification(checked), 3)
    assert list(growth.data[0].y) == [0, 490, None, None]
    assert not growth.data[0].connectgaps


@pytest.mark.parametrize(
    ("after", "before", "message"),
    [
        (0.040227, 0.028284, "Increased by 1.19 pp"),
        (0.028284, 0.040227, "Decreased by 1.19 pp"),
        (0.051, 0.051, "Unchanged (0.00 pp)"),
        (0.01, 0.009999, "Increased by <0.01 pp"),
    ],
)
def test_ece_percentage_point_change(after, before, message):
    """Keep the actual direction even below the display precision."""
    assert ece_change(after, before) == message


def test_partition_chart_uses_verified_counts_and_preserves_unavailable_positions(
    tmp_path,
):
    """Stack only real partition counts and label the final append explicitly."""
    checked = _verified(tmp_path, 3, [_cycle(1, STALE), _cycle(2), _cycle(3)])
    figure = growth_figure(
        ExperimentSummary.from_verification(checked), 3, composition=True
    )
    assert figure.layout.barmode == "stack"
    assert [trace.name for trace in figure.data] == [
        "Development",
        "Calibration",
        "Final test",
    ]
    assert [list(trace.y) for trace in figure.data] == [
        [280, None, 440, 520],
        [39, None, 39, 39],
        [91, None, 91, 91],
    ]
    assert len({trace.marker.pattern.shape for trace in figure.data}) == 3
    assert list(figure.data[0].customdata) == [
        "Bootstrap",
        "After cycle 1",
        "After cycle 2",
        "Final append · cycle 3",
    ]
    assert list(figure.layout.xaxis.ticktext) == [
        "Bootstrap",
        "After cycle 1",
        "After cycle 2",
        "Final append",
    ]


@pytest.mark.integration
def test_headlines_match_published_scientific_results(monkeypatch):
    """Cross-check the actual Results headline formatting against published files."""
    root = Path(
        os.environ.get("MINERAL_QUICK_DEMO_WORKSPACE", "/tmp/mineral-m25.SkdwUb/a")
    )
    if not (root / "dvc.lock").is_file():
        pytest.skip("Local verified Quick demo workspace unavailable")
    monkeypatch.chdir(root)
    app = (
        AppTest.from_file(str(ROOT / "app.py"))
        .switch_page("dashboard/views/results.py")
        .run()
    )
    app.button[0].click().run(timeout=20)
    assert not app.exception
    summary = ExperimentSummary.from_verification(
        app.session_state["mineral_dashboard"].result_verification
    )
    assert summary.latest == 2
    test = json.loads((root / "results/evaluation/cycle_2/metrics.json").read_text())
    batch = json.loads((root / "results/prediction/cycle_2/metrics.json").read_text())
    assert [metric.label for metric in app.metric] == [
        "Final-test accuracy",
        "Final-test ECE after calibration",
        "Prediction-batch accuracy",
    ]
    assert [metric.value for metric in app.metric] == [
        f"{test['calibrated']['acc']:.2%}",
        f"{test['calibrated']['ece']:.2%}",
        f"{batch['acc']:.2%}",
    ]
    assert any(
        f"Before {test['uncalibrated']['ece']:.2%}" in caption.value
        for caption in app.caption
    )
    assert len(app.get("plotly_chart")) == 2
    assert [snapshot.total for snapshot in summary.snapshots.values()] == [
        440,
        540,
        640,
    ]


def test_overview_drill_through_restores_selection_and_invalidates_evidence(
    tmp_path, monkeypatch
):
    """Keep mode/cycle/section across pages and avoid hidden scientific reads."""
    settings = parse_configuration_yaml((ROOT / "params.yaml").read_bytes())
    settings["cycles"] = 2
    params = tmp_path / "params.yaml"
    params.write_text(dump_configuration_yaml(settings))
    applied, _ = AppliedConfiguration.read(params.read_bytes())
    paths = [
        "drift_analysis/cycle_1/drift_results.txt",
        "drift_analysis/cycle_9/drift_results.txt",
    ]
    for relative in paths:
        file = tmp_path / "results" / relative
        file.parent.mkdir(parents=True)
        file.write_text("Historical UI fixture")

    checks, reads = [], []
    tracked = (f"results/{paths[0]}",)

    def refresh(workspace):
        checks.append(workspace.root)
        return replace(
            _verified(tmp_path, 2, [_cycle(1), _cycle(2, STALE)], applied=applied),
            paths=tracked,
            stamp=WorkspaceStamp.capture(tmp_path, tracked),
        )

    read = Workspace.read

    def capture(workspace, path):
        reads.append(path.as_posix())
        return read(workspace, path)

    monkeypatch.setattr(provenance, "verify_workspace", refresh)
    monkeypatch.setattr(Workspace, "read", capture)
    monkeypatch.chdir(tmp_path)
    app = (
        AppTest.from_file(str(ROOT / "app.py"))
        .switch_page("dashboard/views/results.py")
        .run()
    )
    assert app.segmented_control[0].value == "Experiment overview"
    assert not app.exception and not reads and not checks and not app.metric
    app.button[0].click().run()
    assert not app.exception
    assert any("Outdated or conflicting" in item.value for item in app.markdown)
    assert any("1 of 2 learning cycles verified" in item.value for item in app.markdown)
    assert any(
        "historical cycle outside the applied scope" in item.value
        for item in app.caption
    )
    assert app.dataframe[0].value["Cycle"].tolist() == [
        "Bootstrap · 0",
        "Cycle 1",
        "Cycle 2",
    ]
    app.selectbox[0].select(Metric.MCC).run()
    app.segmented_control[1].select("Partitions").run()
    charts = app.get("plotly_chart")
    assert len(charts) == 2
    assert [trace["type"] for trace in json.loads(charts[-1].proto.spec)["data"]] == [
        "bar",
        "bar",
        "bar",
    ]
    app.segmented_control[1].select("Totals").run()
    assert len(app.get("plotly_chart")) == 2
    assert len(checks) == 1 and not reads

    next(box for box in app.selectbox if box.label == "Cycle to inspect").select(
        1
    ).run()
    next(
        button for button in app.button if button.label == "Inspect cycle"
    ).click().run()
    assert app.segmented_control[0].value == "Cycle explorer"
    assert app.selectbox[0].value == 1
    assert any("Input base 0 + incoming batch 1" in item.value for item in app.caption)
    app.selectbox[1].select("Distribution drift").run()
    assert reads == [paths[0]]
    app.switch_page("dashboard/views/home.py").run()
    app.switch_page("dashboard/views/results.py").run()
    assert app.segmented_control[0].value == "Cycle explorer"
    assert (
        app.selectbox[0].value == 1 and app.selectbox[1].value == "Distribution drift"
    )

    app.selectbox[0].select(9).run()
    assert app.selectbox[0].value == 9
    assert app.selectbox[1].value == "Distribution drift"
    assert app.selectbox[2].value == Path(paths[1])
    assert reads[-1] == paths[1]
    assert any(
        "Historical cycle outside the applied configuration" in item.value
        for item in app.markdown
    )
    app.selectbox[0].select(0).run()
    assert app.selectbox[1].value == "Data and spatial graphs"
    app.selectbox[0].select(1).run()
    assert app.selectbox[1].value == "Distribution drift"

    app.segmented_control[0].select("Experiment overview").run()
    assert len(checks) == 1
    (tmp_path / tracked[0]).write_text("Changed bytes")
    app.run()
    assert not app.exception and not app.metric and not app.get("plotly_chart")
    assert len(checks) == 1
    assert any("Workspace evidence changed" in item.value for item in app.caption)
