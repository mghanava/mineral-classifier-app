"""Regression tests for dashboard navigation, configuration and diagnostics."""

import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import plotly.graph_objects as go
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from dashboard.artifacts import Workspace

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Start the dashboard with only its configuration file."""
    copyfile(APP.parent / "params.yaml", tmp_path / "params.yaml")
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(str(APP)).run(timeout=20)


def _page(dashboard: AppTest, page: str) -> AppTest:
    return dashboard.switch_page(f"dashboard/views/{page}.py").run(timeout=20)


def _select(dashboard: AppTest, label: str, value: str | int | Path) -> AppTest:
    return (
        next(widget for widget in dashboard.selectbox if widget.label == label)
        .select(value)
        .run()
    )


@pytest.fixture
def configure(dashboard: AppTest) -> AppTest:
    """Open the expert editor in the isolated workspace."""
    return _page(dashboard, "configure").radio[0].set_value("Expert").run()


def test_dashboard_renders_only_selected_page(dashboard, tmp_path, monkeypatch):
    """Navigation leaves applied bytes alone and reads no scientific payloads."""
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    artifact = tmp_path / "results/evaluation/cycle_1/metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"calibrated": {}}')

    def unexpected_read(*args):
        pytest.fail("Navigation attempted to read an unrelated artifact")

    monkeypatch.setattr(Workspace, "read", unexpected_read)
    for page, title, controls in (
        ("home", "Mineral Classifier", {"Configure Quick demo"}),
        ("configure", "Configure", {"Load preset into draft", "Validate & Preview"}),
        ("run", "Run", {"Run full experiment", "Run stage"}),
        ("results", "Results", {"Refresh result evidence", "Inspect cycle"}),
    ):
        _page(dashboard, page)
        assert not dashboard.exception and not dashboard.error and not dashboard.tabs
        assert [item.value for item in dashboard.title] == [title]
        assert {item.label for item in dashboard.button} == controls
        assert active.read_bytes() == original
        assert not (tmp_path / "dvc.yaml").exists()
        if page in {"home", "results"}:
            assert any(
                "source revision is unverified" in item.value
                for item in [*dashboard.info, *dashboard.caption]
            )


def _explorer(dashboard: AppTest) -> AppTest:
    return dashboard.segmented_control[0].select("Cycle explorer").run()


def _original(dashboard: AppTest) -> AppTest:
    return (
        next(
            widget
            for widget in dashboard.segmented_control
            if widget.label == "Spatial view"
        )
        .select("Original diagnostics")
        .run()
    )


def test_empty_results_and_directory_presence(dashboard, tmp_path):
    """An empty output directory provides no evidence of available diagnostics."""
    (tmp_path / "results/evaluation/cycle_1").mkdir(parents=True)
    _page(dashboard, "results")
    assert "No display artifacts found" in dashboard.info[0].value
    _explorer(dashboard)
    assert "absent or partial" in dashboard.warning[0].value


@pytest.mark.parametrize(
    ("setup_status", "dvc_status", "failed_module"),
    [
        (0, 0, None),
        (7, 0, "scripts.setup_dvc"),
        (0, 8, "dvc"),
        (None, 0, "scripts.setup_dvc"),
        (0, None, "dvc"),
    ],
)
def test_dashboard_pipeline_status_and_short_circuit(
    dashboard: AppTest,
    monkeypatch: pytest.MonkeyPatch,
    setup_status: int | None,
    dvc_status: int | None,
    failed_module: str | None,
) -> None:
    """Stream subprocess output and run DVC only after successful setup."""
    calls = []
    popen = subprocess.Popen
    statuses = {"scripts.setup_dvc": setup_status, "dvc": dvc_status}

    def run_fixture(command, *args, **kwargs):
        if not isinstance(command, list) or command[:2] != [sys.executable, "-m"]:
            return popen(command, *args, **kwargs)
        module = command[2]
        if module not in statuses:
            return popen(command, *args, **kwargs)
        assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(
            Path(sys.executable).parent
        )
        calls.append(command[2:])
        status = statuses[module]
        if status is None:
            raise FileNotFoundError("fixture executable unavailable")
        code = f"import sys; print('fixture stage output'); sys.exit({status})"
        return popen([sys.executable, "-c", code], *args, **kwargs)

    _page(dashboard, "run")
    monkeypatch.setattr(subprocess, "Popen", run_fixture)
    button = next(b for b in dashboard.button if b.label == "Run full experiment")
    button.click().run(timeout=20)

    expected = [["scripts.setup_dvc"]]
    if setup_status == 0:
        expected.append(["dvc", "repro"])
    assert calls == expected
    assert not dashboard.exception
    if setup_status is not None:
        assert any("fixture stage output" in item.value for item in dashboard.code)
    if failed_module is None:
        assert not dashboard.error
        assert [item.value for item in dashboard.success] == [
            "Completed · Full experiment"
        ]
    else:
        assert not dashboard.success
        error = dashboard.error[1].value
        assert failed_module in error or (failed_module == "dvc" and "DVC" in error)
        status = statuses[failed_module]
        assert ("Could not start" if status is None else f"status {status}") in error
    _page(dashboard, "home")
    _page(dashboard, "run")
    assert (
        dashboard.session_state["mineral_dashboard"].last_run.scope == "Full experiment"
    )
    assert any(
        "Full experiment" in item.value
        for item in [*dashboard.success, *dashboard.error]
    )


def _click(dashboard: AppTest, label: str) -> AppTest:
    return (
        next(button for button in dashboard.button if button.label == label)
        .click()
        .run(timeout=20)
    )


def test_preset_and_draft_only_write_on_explicit_apply(
    configure, tmp_path, monkeypatch
):
    """Selecting, loading, editing and previewing a preset leave production state alone."""
    dashboard = configure
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    draft = dashboard.text_area[0].value + "# My edit\n"
    dashboard.text_area[0].set_value(draft).run()
    dashboard.selectbox[0].select("quick_demo").run()
    assert dashboard.text_area[0].value == draft
    assert active.read_bytes() == original

    _click(dashboard, "Load preset into draft")
    loaded = dashboard.text_area[0].value
    assert "n_epochs: 15\n" in loaded
    assert active.read_bytes() == original
    # Exercise a second rerun after editing, independent of the validation event.
    dashboard.text_area[0].set_value(loaded.replace("cycles: 2", "cycles: 0")).run()
    dashboard.run()
    assert active.read_bytes() == original
    _page(dashboard, "run")
    assert any("Learning cycles** 3" in item.value for item in dashboard.markdown)
    _page(dashboard, "configure")

    _click(dashboard, "Validate & Preview")
    assert any(
        "regeneration is required for changes to cycles" in item.value
        for item in dashboard.warning
    )
    assert any("cannot trigger" in item.value for item in dashboard.warning)
    assert any("+cycles: 0" in item.value for item in dashboard.code)
    assert active.read_bytes() == original

    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("Applying configuration must not run DVC or training")

    monkeypatch.setattr(subprocess, "Popen", unexpected_subprocess)
    _click(dashboard, "Apply configuration")
    assert not dashboard.exception and not dashboard.error
    assert any("Configuration applied" in item.value for item in dashboard.success)
    assert "cycles: 0\n" in active.read_text()
    _page(dashboard, "run")
    _select(dashboard, "Stage", "Training")
    assert not dashboard.number_input
    assert next(
        button for button in dashboard.button if button.label == "Run stage"
    ).disabled
    assert not dashboard.warning


def test_semantic_and_parse_errors_prevent_dashboard_apply(configure, tmp_path):
    """Keep active bytes when the draft contains invalid values or duplicate keys."""
    dashboard = configure
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    draft = dashboard.text_area[0].value
    dashboard.text_area[0].set_value(draft.replace("cycles: 3", "cycles: -1")).run()
    _click(dashboard, "Validate & Preview")
    assert any("cycles" in item.value for item in dashboard.error)
    assert next(
        button for button in dashboard.button if button.label == "Apply configuration"
    ).disabled
    assert not dashboard.success
    dashboard.text_area[0].set_value(draft + "\ncycles: 1\n").run()
    _click(dashboard, "Validate & Preview")
    assert any("Duplicate mapping key" in item.value for item in dashboard.error)
    assert all(button.label != "Apply configuration" for button in dashboard.button)
    assert active.read_bytes() == original


def test_dashboard_preview_stale_after_external_edit(configure, tmp_path):
    """Preserve draft edits and require a refreshed preview after an active change."""
    dashboard = configure
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    draft = dashboard.text_area[0].value.replace("cycles: 3", "cycles: 1")
    dashboard.text_area[0].set_value(draft).run()
    _click(dashboard, "Validate & Preview")
    newer = original + b"\n# External edit\n"
    _page(dashboard, "home")
    active.write_bytes(newer)
    _page(dashboard, "configure")
    _click(dashboard, "Apply configuration")
    assert not dashboard.success
    assert any("changed since preview" in item.value for item in dashboard.warning)
    assert dashboard.text_area[0].value == draft
    assert active.read_bytes() == newer


def test_dashboard_failed_write_cannot_report_success(configure, tmp_path, monkeypatch):
    """Show a failed write and leave the reviewed draft available."""
    dashboard = configure
    from src import configuration_files

    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    _click(dashboard, "Validate & Preview")

    def fail(*args):
        raise PermissionError("Read-only fixture")

    monkeypatch.setattr(configuration_files.os, "replace", fail)
    _click(dashboard, "Apply configuration")
    assert any("Read-only fixture" in item.value for item in dashboard.error)
    assert not dashboard.success
    assert active.read_bytes() == original


def test_targeted_run_regenerates_from_active_without_using_draft(
    configure, tmp_path, monkeypatch
):
    """Build the actual DVC definition from active YAML before issuing a targeted repro."""
    dashboard = configure
    from scripts.setup_dvc import generate_fine_grained_dvc_yaml

    active = tmp_path / "params.yaml"
    draft = dashboard.text_area[0].value.replace(
        "default_model: gcn", "default_model: gat"
    )
    dashboard.text_area[0].set_value(draft).run()
    calls = []
    expected_model = "gcn"
    popen = subprocess.Popen

    def fake_process(command, *args, **kwargs):
        module = command[2]
        calls.append(command[2:])
        if module == "scripts.setup_dvc":
            generate_fine_grained_dvc_yaml()
        else:
            assert f"--model {expected_model}" in (tmp_path / "dvc.yaml").read_text()
        return popen([sys.executable, "-c", "pass"], *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_process)
    _page(dashboard, "run")
    _select(dashboard, "Stage", "Training")
    _click(dashboard, "Run stage")
    assert calls == [["scripts.setup_dvc"], ["dvc", "repro", "train_cycle_1"]]
    assert "default_model: gcn" in active.read_text()
    assert any("Unapplied draft changes" in item.value for item in dashboard.warning)
    _page(dashboard, "configure")
    _click(dashboard, "Validate & Preview")
    _click(dashboard, "Apply configuration")
    assert "default_model: gat" in active.read_text()
    assert len(calls) == 2
    assert "--model gcn" in (tmp_path / "dvc.yaml").read_text()

    expected_model = "gat"
    _page(dashboard, "run")
    _select(dashboard, "Stage", "Training")
    _click(dashboard, "Run stage")
    assert calls[2:] == [["scripts.setup_dvc"], ["dvc", "repro", "train_cycle_1"]]


def test_run_blocks_invalid_active_configuration(dashboard, tmp_path, monkeypatch):
    """Stop before pipeline generation if an external edit invalidates the active file."""
    active = tmp_path / "params.yaml"
    active.write_text(active.read_text().replace("n_splits: 3", "n_splits: 0"))

    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("Invalid active configuration must block execution")

    _page(dashboard, "run")
    monkeypatch.setattr(subprocess, "Popen", unexpected_subprocess)
    _click(dashboard, "Run full experiment")
    assert any("data.base.n_splits" in item.value for item in dashboard.error)
    assert not dashboard.success


def test_other_session_cannot_apply_during_reproduction(
    dashboard, tmp_path, monkeypatch
):
    """Keep the active configuration fixed until the running command exits."""
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    popen = subprocess.Popen
    attempted = []
    apply_in_other_session = """
import sys
from streamlit.testing.v1 import AppTest

dashboard = AppTest.from_file(sys.argv[1]).switch_page("dashboard/views/configure.py").run()
dashboard.radio[0].set_value("Expert").run()
draft = dashboard.text_area[0].value.replace("cycles: 3", "cycles: 2")
dashboard.text_area[0].set_value(draft).run()
for label in ("Validate & Preview", "Apply configuration"):
    next(button for button in dashboard.button if button.label == label).click().run()
assert not dashboard.exception
for error in dashboard.error:
    print(error.value)
"""

    def fixture_process(command, *args, **kwargs):
        if command[2:] != ["dvc", "repro"]:
            return popen([sys.executable, "-c", "pass"], *args, **kwargs)
        process = popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            *args,
            stdin=subprocess.PIPE,
            **kwargs,
        )
        try:
            with popen(
                [sys.executable, "-c", apply_in_other_session, str(APP)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONPATH": str(APP.parent)},
            ) as other:
                output, errors = other.communicate(timeout=15)
                assert other.returncode == 0, errors
            attempted.append((active.read_bytes(), output))
        finally:
            process.stdin.close()
        return process

    _page(dashboard, "run")
    monkeypatch.setattr(subprocess, "Popen", fixture_process)
    _click(dashboard, "Run full experiment")
    assert not dashboard.exception
    assert attempted and attempted[0][0] == original
    assert "pipeline run is active" in attempted[0][1]

    _page(dashboard, "configure").radio[0].set_value("Expert").run()
    draft = dashboard.text_area[0].value.replace("cycles: 3", "cycles: 2")
    dashboard.text_area[0].set_value(draft).run()
    _click(dashboard, "Validate & Preview")
    _click(dashboard, "Apply configuration")
    assert "cycles: 2\n" in active.read_text()


def test_home_quick_demo_preserves_draft_and_requires_explicit_load(
    configure, tmp_path
):
    """Highlight the Quick demo without changing reviewed draft or active bytes."""
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    draft = configure.text_area[0].value + "\n# Unapplied edit"
    configure.text_area[0].set_value(draft).run()
    _select(configure, "Experiment preset", "research")
    _click(configure, "Validate & Preview")

    _page(configure, "run")
    _page(configure, "configure")
    assert configure.selectbox[0].value == "research"
    assert configure.text_area[0].value == draft
    assert any(item.label == "Apply configuration" for item in configure.button)

    _page(configure, "home")
    _click(configure, "Configure Quick demo")
    assert configure.title[0].value == "Configure"
    assert configure.selectbox[0].value == "quick_demo"
    assert configure.text_area[0].value == draft
    assert any(item.label == "Apply configuration" for item in configure.button)

    # AppTest retains the explicitly requested page hash across st.switch_page.
    _page(configure, "configure")
    configure.text_area[0].set_value(draft + "\n# Another edit").run()
    configure.text_area[0].set_value(draft).run()
    assert all(item.label != "Apply configuration" for item in configure.button)
    assert active.read_bytes() == original


@pytest.mark.parametrize(
    ("count", "stage", "target"),
    [
        (0, "Bootstrap", "bootstrap"),
        (0, "Training", None),
        (1, "Cross-cycle performance aggregation", None),
        (2, "Cross-cycle performance aggregation", "analyze_cycles_performance"),
        (2, "Evaluation", "evaluate_cycle_2"),
        (2, "Prediction", "predict_cycle_2"),
        (2, "Drift analysis", "analyze_drift_cycle_2"),
    ],
)
def test_stage_targets_and_cycle_restrictions(
    dashboard, tmp_path, monkeypatch, count, stage, target
):
    """Issue exactly the existing target from the applied cycle limits."""
    from dashboard import execution

    active = tmp_path / "params.yaml"
    active.write_text(active.read_text().replace("cycles: 3", f"cycles: {count}"))
    calls = []

    def capture(command, attempt, notify):
        calls.append(command[2:])
        return 0

    monkeypatch.setattr(execution, "run_command", capture)
    _page(dashboard, "run")
    _select(dashboard, "Stage", stage)
    if dashboard.number_input:
        dashboard.number_input[0].set_value(count).run()
    button = next(item for item in dashboard.button if item.label == "Run stage")
    assert button.disabled == (target is None)
    if target:
        _click(dashboard, "Run stage")
        assert calls == [["scripts.setup_dvc"], ["dvc", "repro", target]]
    else:
        assert not calls


def _fixture_artifact(workspace, relative, contents):
    path = workspace / "results" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if contents == "png":
        Image.new("RGB", (16, 16), "white").save(path)
    else:
        path.write_text(contents)
    return Path(relative)


def test_result_cycle_and_population_routing(dashboard, tmp_path, monkeypatch):
    """Read the selected base or batch export and restore shared Results selection."""
    graph = go.Figure(go.Scatter3d(x=[1], y=[2], z=[3])).to_json()
    bootstrap = _fixture_artifact(
        tmp_path, "data/base/cycle_0/base_data_graph.json", graph
    )
    final = _fixture_artifact(tmp_path, "data/base/cycle_2/base_data_graph.json", graph)
    _fixture_artifact(tmp_path, str(final.with_suffix(".html")), "<p>Legacy HTML</p>")
    fold = _fixture_artifact(tmp_path, "data/base/cycle_2/train_graph_1.json", graph)
    batch = _fixture_artifact(
        tmp_path, "data/prediction/cycle_2/prediction_data_graph.json", graph
    )
    report = _fixture_artifact(
        tmp_path, "drift_analysis/cycle_2/drift_results.txt", "Fixture report"
    )
    active = tmp_path / "params.yaml"
    active.write_text(active.read_text().replace("cycles: 3", "cycles: 2"))
    reads = []
    read = Workspace.read

    def capture(workspace, path):
        reads.append(path)
        return read(workspace, path)

    monkeypatch.setattr(Workspace, "read", capture)
    _page(dashboard, "results")
    assert not reads
    _explorer(dashboard)
    assert not reads
    _original(dashboard)
    assert reads == [bootstrap]
    _select(dashboard, "Learning cycle", 0)
    assert reads == [bootstrap, bootstrap]
    assert dashboard.selectbox[1].options == [
        "Data and spatial graphs",
        "Cross-cycle performance",
    ]
    _select(dashboard, "Learning cycle", 2)
    assert len(reads) == 2  # input base 1 is absent
    _select(dashboard, "Population", "Incoming batch 2")
    assert reads[-1] == batch
    _select(dashboard, "Population", "Post-append base · snapshot 2")
    assert reads[-1] == final
    assert len(dashboard.get("plotly_chart")) == 1
    _select(dashboard, "Diagnostic", fold)
    assert reads[-1] == fold

    _select(dashboard, "Result section", "Distribution drift")
    _page(dashboard, "configure")
    _page(dashboard, "results")
    assert dashboard.selectbox[0].value == 2
    assert dashboard.selectbox[1].value == "Distribution drift"
    assert reads[-2:] == [report, report]

    active.write_text(active.read_text().replace("cycles: 2", "cycles: 0"))
    dashboard.run()
    assert dashboard.selectbox[0].value == 2
    assert "Cycle 2 · existing files only" in dashboard.selectbox[0].options
    for relative in (final, final.with_suffix(".html"), fold, batch, report):
        (tmp_path / "results" / relative).unlink()
    dashboard.run()
    assert dashboard.selectbox[0].value == 0
    assert dashboard.selectbox[1].value == "Data and spatial graphs"
    assert not dashboard.exception


@pytest.mark.parametrize(
    ("section", "diagnostics"),
    [
        ("Data and spatial graphs", ["data/base/cycle_0/tsne_comparison.png"]),
        (
            "Model performance and training",
            [
                "evaluation/cycle_1/confusion_matrix.png",
                "evaluation/cycle_1/metrics.json",
                "prediction/cycle_1/confussion_matrix_.png",
                "prediction/cycle_1/metrics.json",
                "trained/cycle_1/training_plots_dataset_1.png",
                "trained/cycle_1/training_plots_dataset_2.png",
                "trained/cycle_1/graph_binding.json",
                "trained/cycle_1/preprocessing.json",
            ],
        ),
        (
            "Probability calibration",
            [
                "evaluation/cycle_1/reliability_diagram.png",
                "evaluation/cycle_1/calibration_dirichlet_loss_and_gradient.png",
                "evaluation/cycle_1/optimum_params.txt",
                "evaluation/cycle_1/calibration_binding.json",
            ],
        ),
        (
            "Spatial predictions",
            ["prediction/cycle_1/predictions.csv", "prediction/cycle_1/histograms.png"],
        ),
        (
            "Distribution drift",
            [
                "drift_analysis/cycle_1/distance_distributions.png",
                "drift_analysis/cycle_1/marginal_distributions.png",
                "drift_analysis/cycle_1/mutual_information.png",
                "drift_analysis/cycle_1/pca_kpca_grid.png",
                "drift_analysis/cycle_1/drift_results.txt",
                "drift_analysis/cycle_1/drift_metadata.json",
            ],
        ),
        (
            "Cross-cycle performance",
            ["cycles_performance_analysis/cycles_performance.png"],
        ),
    ],
)
def test_existing_diagnostics_are_selectable(
    dashboard, tmp_path, monkeypatch, section, diagnostics
):
    """Route each original diagnostic through the selected-file reader."""
    paths = []
    for path in diagnostics:
        contents = {
            ".json": '{"fixture": 1}',
            ".txt": "Fixture report",
            ".csv": "label\n1\n",
            ".png": "png",
        }
        paths.append(_fixture_artifact(tmp_path, path, contents[Path(path).suffix]))
    reads = []
    read = Workspace.read

    def capture(workspace, path):
        reads.append(path)
        return read(workspace, path)

    monkeypatch.setattr(Workspace, "read", capture)
    _page(dashboard, "results")
    _explorer(dashboard)
    _select(dashboard, "Result section", section)
    if section in ("Data and spatial graphs", "Spatial predictions"):
        _original(dashboard)
    assert len(dashboard.selectbox[-1].options) == len(paths)
    for path in paths:
        reads.clear()
        _select(dashboard, "Diagnostic", path)
        assert reads == [path]
        assert not dashboard.exception and not dashboard.warning
        if path.stem == "reliability_diagram":
            assert any(
                "Below the diagonal is overconfidence" in item.value
                for item in dashboard.caption
            )
        if path.stem == "metrics" or path.stem.startswith(
            ("training_plots_", "cycles_performance")
        ):
            assert any("Macro F0.5" in item.value for item in dashboard.caption)


@pytest.mark.parametrize(
    ("path", "section", "contents"),
    [
        (
            "data/base/cycle_0/base_data_graph.json",
            "Data and spatial graphs",
            '{"data": [',
        ),
        (
            "trained/cycle_1/training_plots_dataset_1.png",
            "Model performance and training",
            "partial",
        ),
        (
            "prediction/cycle_1/predictions.csv",
            "Spatial predictions",
            'label\n"unfinished',
        ),
        ("drift_analysis/cycle_1/drift_results.txt", "Distribution drift", ""),
    ],
)
def test_partial_artifacts_are_reported(dashboard, tmp_path, path, section, contents):
    """Incomplete writes remain inspectable as warnings rather than page errors."""
    _fixture_artifact(tmp_path, path, contents)
    _page(dashboard, "results")
    _explorer(dashboard)
    _select(dashboard, "Result section", section)
    if section in ("Data and spatial graphs", "Spatial predictions"):
        _original(dashboard)
    assert not dashboard.exception and not dashboard.error
    assert any("incomplete or malformed" in item.value for item in dashboard.warning)


def test_legacy_html_and_external_artifact_link(dashboard, tmp_path, monkeypatch):
    """Require explicit HTML display and reject paths that leave results."""
    html = _fixture_artifact(
        tmp_path, "data/base/cycle_0/base_data_graph.html", "<p>Legacy HTML</p>"
    )
    external = tmp_path / "external.json"
    external.write_text("{}")
    (tmp_path / "results" / html.with_name("test_graph.json")).symlink_to(external)
    reads = []
    read = Workspace.read

    def capture(workspace, path):
        reads.append(path)
        return read(workspace, path)

    monkeypatch.setattr(Workspace, "read", capture)
    _page(dashboard, "results")
    _explorer(dashboard)
    _original(dashboard)
    assert dashboard.selectbox[-1].options == ["Full base graph"]
    assert not reads
    dashboard.checkbox[0].check().run()
    assert reads == [html]
    assert not dashboard.exception
