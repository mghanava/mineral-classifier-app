"""Regression tests for the dashboard before results are generated."""

import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Start the dashboard with only its configuration file."""
    copyfile(APP.parent / "params.yaml", tmp_path / "params.yaml")
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(str(APP)).run(timeout=20)


def test_dashboard_renders_without_artifacts(
    dashboard: AppTest, tmp_path: Path
) -> None:
    """Render the dashboard and its empty states in an isolated directory."""
    assert not dashboard.exception
    assert not dashboard.error
    tabs = {tab.label: tab for tab in dashboard.tabs}
    assert {
        "Parameters",
        "Data",
        "Train",
        "Evaluation",
        "Prediction",
        "Drift Analysis",
        "Performance",
    } <= tabs.keys()
    for name in (
        "Data",
        "Train",
        "Evaluation",
        "Prediction",
        "Drift Analysis",
        "Performance",
    ):
        assert any("not found" in warning.value for warning in tabs[name].warning), name
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("setup_status", "dvc_status", "failed_module"),
    [
        (0, 0, None),
        (7, 0, "scripts.setup_dvc"),
        (0, 8, "dvc"),
        (None, 0, "scripts.setup_dvc"),
    ],
)
def test_dashboard_pipeline_status_and_short_circuit(
    dashboard: AppTest,
    monkeypatch: pytest.MonkeyPatch,
    setup_status: int | None,
    dvc_status: int,
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

    monkeypatch.setattr(subprocess, "Popen", run_fixture)
    button = next(
        b for b in dashboard.button if b.label == "Generate & Run Full Pipeline"
    )
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
            "Pipeline completed successfully!"
        ]
    else:
        assert not dashboard.success
        error = dashboard.error[0].value
        assert failed_module in error
        status = statuses[failed_module]
        assert ("Could not start" if status is None else f"status {status}") in error


def _click(dashboard: AppTest, label: str) -> AppTest:
    return (
        next(button for button in dashboard.button if button.label == label)
        .click()
        .run(timeout=20)
    )


def test_preset_and_draft_only_write_on_explicit_apply(
    dashboard, tmp_path, monkeypatch
):
    """Selecting, loading, editing and previewing a preset leave production state alone."""
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
    assert dashboard.number_input[0].max == 3

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
    assert dashboard.number_input[0].max == 0
    assert next(
        button for button in dashboard.button if button.label == "Run Training Stage"
    ).disabled


def test_semantic_and_parse_errors_prevent_dashboard_apply(dashboard, tmp_path):
    """Keep active bytes when the draft contains invalid values or duplicate keys."""
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


def test_dashboard_preview_stale_after_external_edit(dashboard, tmp_path):
    """Preserve draft edits and require a refreshed preview after an active change."""
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    draft = dashboard.text_area[0].value.replace("cycles: 3", "cycles: 1")
    dashboard.text_area[0].set_value(draft).run()
    _click(dashboard, "Validate & Preview")
    newer = original + b"\n# External edit\n"
    active.write_bytes(newer)
    _click(dashboard, "Apply configuration")
    assert not dashboard.success
    assert any("changed since preview" in item.value for item in dashboard.warning)
    assert dashboard.text_area[0].value == draft
    assert active.read_bytes() == newer


def test_dashboard_failed_write_cannot_report_success(dashboard, tmp_path, monkeypatch):
    """Show a failed write and leave the reviewed draft available."""
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
    dashboard, tmp_path, monkeypatch
):
    """Build the actual DVC definition from active YAML before issuing a targeted repro."""
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
    _click(dashboard, "Run Training Stage")
    assert calls == [["scripts.setup_dvc"], ["dvc", "repro", "train_cycle_1"]]
    assert "default_model: gcn" in active.read_text()
    _click(dashboard, "Validate & Preview")
    _click(dashboard, "Apply configuration")
    assert "default_model: gat" in active.read_text()
    assert len(calls) == 2
    assert "--model gcn" in (tmp_path / "dvc.yaml").read_text()

    expected_model = "gat"
    _click(dashboard, "Run Training Stage")
    assert calls[2:] == [["scripts.setup_dvc"], ["dvc", "repro", "train_cycle_1"]]


def test_run_blocks_invalid_active_configuration(dashboard, tmp_path, monkeypatch):
    """Stop before pipeline generation if an external edit invalidates the active file."""
    active = tmp_path / "params.yaml"
    active.write_text(active.read_text().replace("n_splits: 3", "n_splits: 0"))

    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("Invalid active configuration must block execution")

    monkeypatch.setattr(subprocess, "Popen", unexpected_subprocess)
    _click(dashboard, "Generate & Run Full Pipeline")
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

dashboard = AppTest.from_file(sys.argv[1]).run()
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

    monkeypatch.setattr(subprocess, "Popen", fixture_process)
    _click(dashboard, "Generate & Run Full Pipeline")
    assert not dashboard.exception
    assert attempted and attempted[0][0] == original
    assert "pipeline run is active" in attempted[0][1]

    draft = dashboard.text_area[0].value.replace("cycles: 3", "cycles: 2")
    dashboard.text_area[0].set_value(draft).run()
    _click(dashboard, "Validate & Preview")
    _click(dashboard, "Apply configuration")
    assert "cycles: 2\n" in active.read_text()
