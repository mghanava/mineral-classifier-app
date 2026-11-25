"""Regression tests for isolated DVC pipeline generation."""

from pathlib import Path

import pytest
import yaml

from scripts.setup_dvc import generate_fine_grained_dvc_yaml


@pytest.mark.parametrize(("cycles", "model"), [(1, "gcn"), (2, "gat")])
def test_generate_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cycles: int, model: str
) -> None:
    """Generate configured stages and connect training to prior-cycle data."""
    monkeypatch.chdir(tmp_path)
    config = {"cycles": cycles, "default_model": model}
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    generate_fine_grained_dvc_yaml()

    pipeline = yaml.safe_load((tmp_path / "dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]
    assert "bootstrap" in stages

    for cycle in range(1, cycles + 1):
        train = stages[f"train_cycle_{cycle}"]
        assert train["cmd"].split()[-4:] == ["--cycle", str(cycle), "--model", model]
        producer = "bootstrap" if cycle == 1 else f"prepare_next_cycle_data_{cycle - 1}"
        fold_data = f"results/data/base/cycle_{cycle - 1}/fold_data.pt"
        assert fold_data in stages[producer]["outs"]
        assert fold_data in train["deps"]
        for stage in ("evaluate", "generate_pred_data", "predict", "analyze_drift"):
            assert f"{stage}_cycle_{cycle}" in stages
        assert f"prepare_next_cycle_data_{cycle}" in stages

    assert f"train_cycle_{cycles + 1}" not in stages
    assert ("analyze_cycles_performance" in stages) == (cycles > 1)
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "dvc.lock").exists()
