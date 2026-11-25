"""Regression tests for isolated DVC pipeline generation."""

import shlex
import subprocess
import sys
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

        base = f"results/data/base/cycle_{cycle - 1}"
        training_deps = {
            f"{base}/{kind}.pt"
            for kind in ("base_data", "fold_data", "test_data", "calib_data")
        }
        assert training_deps <= set(train["deps"])
        assert {"data.scaler_type", "data.scaler_params"} <= set(train["params"])
        assert f"results/trained/cycle_{cycle}" in train["outs"]

        preprocess = f"results/trained/cycle_{cycle}/preprocessing.json"
        evaluation = stages[f"evaluate_cycle_{cycle}"]
        assert {preprocess, *training_deps} <= set(evaluation["deps"])
        assert f"results/evaluation/cycle_{cycle}" in evaluation["outs"]
        prediction = stages[f"predict_cycle_{cycle}"]
        calibration = f"results/evaluation/cycle_{cycle}"
        assert {
            preprocess,
            f"{calibration}/calibration_binding.json",
            f"{calibration}/calibrator.pt",
        } <= set(prediction["deps"])

        generation = stages[f"generate_pred_data_cycle_{cycle}"]
        auxiliaries = {
            f"results/data/base/cycle_0/{name}.npy"
            for name in ("hotspots", "prototypes")
        }
        assert auxiliaries <= set(stages["bootstrap"]["outs"])
        assert auxiliaries | {f"{base}/base_data.pt"} <= set(generation["deps"])
        assert not any("results/trained" in dep for dep in generation["deps"])
        assert not any("data.scaler" in parameter for parameter in generation["params"])
        drift = stages[f"analyze_drift_cycle_{cycle}"]
        assert {
            preprocess,
            f"results/trained/cycle_{cycle}/model.pt",
            f"{base}/base_data.pt",
            f"results/data/prediction/cycle_{cycle}/pred_data.pt",
        } <= set(drift["deps"])
        assert f"results/drift_analysis/cycle_{cycle}" in drift["outs"]

    assert f"train_cycle_{cycles + 1}" not in stages
    assert ("analyze_cycles_performance" in stages) == (cycles > 1)
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "dvc.lock").exists()


def test_scaler_changes_invalidate_consumers_and_sidecars_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the generated artifact edges with tiny stage commands in a local cache."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "params.yaml").read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site"))
    monkeypatch.setenv("DVC_NO_ANALYTICS", "1")

    def dvc(*args: str) -> None:
        command = subprocess.run(
            [sys.executable, "-m", "dvc", *args],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert command.returncode == 0, command.stdout + command.stderr

    dvc("init", "--no-scm")
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump(config), encoding="utf-8")
    generate_fine_grained_dvc_yaml()
    pipeline = Path("dvc.yaml")
    spec = yaml.safe_load(pipeline.read_text(encoding="utf-8"))
    assert spec == yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8"))

    # Keep the generated data edges, parameters, and output declarations.
    names = (
        "bootstrap",
        "generate_pred_data_cycle_1",
        "train_cycle_1",
        "evaluate_cycle_1",
        "predict_cycle_1",
        "analyze_drift_cycle_1",
    )
    fixture_stages = {name: spec["stages"][name] for name in names}
    python = shlex.quote(sys.executable)
    for name, stage in fixture_stages.items():
        stage["cmd"] = f"{python} fixture.py {name}"
        stage["deps"] = [
            "fixture.py",
            *(path for path in stage["deps"] if path.startswith("results/")),
        ]
    pipeline.write_text(yaml.safe_dump({"stages": fixture_stages}), encoding="utf-8")
    Path("fixture.py").write_text(
        """import sys
from pathlib import Path

import yaml

stage = sys.argv[1]
spec = yaml.safe_load(Path("dvc.yaml").read_text())["stages"][stage]
with Path("calls.txt").open("a") as log:
    log.write(stage + "\\n")

preprocessing = Path("results/trained/cycle_1/preprocessing.json")
sidecars = {
    "train_cycle_1": ("model.pt", "preprocessing.json"),
    "evaluate_cycle_1": ("calibrator.pt", "calibration_binding.json"),
    "predict_cycle_1": ("predictions.csv",),
    "analyze_drift_cycle_1": ("drift_metadata.json",),
}
for output in map(Path, spec["outs"]):
    if stage not in sidecars:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(stage)
        continue
    output.mkdir(parents=True, exist_ok=True)
    for name in sidecars[stage]:
        value = stage
        if name == "preprocessing.json":
            value = str(yaml.safe_load(Path("params.yaml").read_text())["data"]["scaler_params"])
        elif name != "model.pt":
            value = preprocessing.read_text()
        (output / name).write_text(value)
""",
        encoding="utf-8",
    )

    dvc("dag", "--dot")
    dvc("repro", "--no-run-cache")
    calls = Path("calls.txt")
    assert sorted(calls.read_text(encoding="utf-8").splitlines()) == sorted(names)
    calls.write_text("", encoding="utf-8")
    config["data"]["scaler_params"]["quantile_range"] = [10, 90]
    params.write_text(yaml.safe_dump(config), encoding="utf-8")
    dvc("repro", "--no-run-cache")
    expected = {
        "train_cycle_1",
        "evaluate_cycle_1",
        "predict_cycle_1",
        "analyze_drift_cycle_1",
    }
    assert sorted(calls.read_text(encoding="utf-8").splitlines()) == sorted(expected)

    # Restore only disposable fixture outputs from the temporary DVC cache.
    for path in (
        Path("results/trained/cycle_1/preprocessing.json"),
        Path("results/evaluation/cycle_1/calibration_binding.json"),
        Path("results/drift_analysis/cycle_1/drift_metadata.json"),
    ):
        assert "[10, 90]" in path.read_text(encoding="utf-8")
        path.unlink()
    dvc("checkout")
    for path in (
        Path("results/trained/cycle_1/preprocessing.json"),
        Path("results/evaluation/cycle_1/calibration_binding.json"),
        Path("results/drift_analysis/cycle_1/drift_metadata.json"),
    ):
        assert "[10, 90]" in path.read_text(encoding="utf-8")
