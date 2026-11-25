"""Regression tests for the dashboard before results are generated."""

from pathlib import Path
from shutil import copyfile

import pytest
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render the dashboard and its empty states in an isolated directory."""
    repo_root = Path(__file__).resolve().parents[1]
    copyfile(repo_root / "params.yaml", tmp_path / "params.yaml")
    monkeypatch.chdir(tmp_path)

    dashboard = AppTest.from_file(str(repo_root / "app.py")).run(timeout=20)

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
