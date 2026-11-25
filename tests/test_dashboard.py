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
