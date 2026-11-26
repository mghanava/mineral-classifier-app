"""Regression tests for isolated DVC pipeline generation."""

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.setup_dvc import generate_fine_grained_dvc_yaml
from src.stages import analyze_cycles_performance

ROOT = Path(__file__).resolve().parents[1]
CYCLE_STAGES = (
    "train_cycle_{}",
    "evaluate_cycle_{}",
    "generate_pred_data_cycle_{}",
    "predict_cycle_{}",
    "analyze_drift_cycle_{}",
    "prepare_next_cycle_data_{}",
)


def dvc(*args: str) -> None:
    """Run DVC in the current temporary repository."""
    command = subprocess.run(
        [sys.executable, "-m", "dvc", *args],
        check=False,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert command.returncode == 0, command.stdout + command.stderr


@pytest.fixture
def isolated_dvc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the workspace, local DVC cache, and site cache."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site"))
    monkeypatch.setenv("DVC_NO_ANALYTICS", "1")
    dvc("init", "--no-scm")
    return tmp_path


@pytest.mark.parametrize(("cycles", "model", "folds"), [(1, "gcn", 2), (2, "gat", 4)])
def test_generate_pipeline(
    isolated_dvc: Path, cycles: int, model: str, folds: int
) -> None:
    """Generate configured stages and connect training to prior-cycle data."""
    config = {
        "cycles": cycles,
        "default_model": model,
        "data": {"base": {"n_splits": folds}},
    }
    Path("params.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    generate_fine_grained_dvc_yaml()

    pipeline = yaml.safe_load(Path("dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]
    expected_names = {"bootstrap"} | {
        pattern.format(cycle)
        for cycle in range(1, cycles + 1)
        for pattern in CYCLE_STAGES
    }
    if cycles > 1:
        expected_names.add("analyze_cycles_performance")
    assert set(stages) == expected_names
    dvc("dag", "--dot")

    outputs = [Path(output) for stage in stages.values() for output in stage["outs"]]
    assert len(set(outputs)) == len(outputs)
    assert all(output not in other.parents for output in outputs for other in outputs)
    assert all(
        (ROOT / dep).is_file()
        for stage in stages.values()
        for dep in stage["deps"]
        if dep.startswith("src/")
    )
    for cycle in range(cycles + 1):
        producer = "bootstrap" if cycle == 0 else f"prepare_next_cycle_data_{cycle}"
        base_outputs = {Path(path).name for path in stages[producer]["outs"]}
        assert {
            name
            for name in base_outputs
            if name.startswith(("train_graph_", "val_graph_"))
        } == {
            f"{role}_graph_{fold}.{ext}"
            for fold in range(1, folds + 1)
            for role in ("train", "val")
            for ext in ("html", "json")
        }

    for cycle in range(1, cycles + 1):
        train = stages[f"train_cycle_{cycle}"]
        assert train["cmd"].split()[-4:] == ["--cycle", str(cycle), "--model", model]
        producer = "bootstrap" if cycle == 1 else f"prepare_next_cycle_data_{cycle - 1}"
        base = f"results/data/base/cycle_{cycle - 1}"
        training_deps = {
            f"{base}/{kind}.pt"
            for kind in ("base_data", "fold_data", "test_data", "calib_data")
        }
        assert training_deps <= set(stages[producer]["outs"])
        assert result_deps(train) == training_deps
        assert {"data.scaler_type", "data.scaler_params"} <= set(train["params"])
        assert "src/utilities/calibration_utils.py" in train["deps"]
        assert f"results/trained/cycle_{cycle}" in train["outs"]

        preprocess = f"results/trained/cycle_{cycle}/preprocessing.json"
        graph_binding = f"results/trained/cycle_{cycle}/graph_binding.json"
        evaluation = stages[f"evaluate_cycle_{cycle}"]
        assert f"results/evaluation/cycle_{cycle}" in evaluation["outs"]
        prediction = stages[f"predict_cycle_{cycle}"]
        for consumer in (evaluation, prediction):
            assert "src/utilities/early_stopping.py" in consumer["deps"]
        calibration = f"results/evaluation/cycle_{cycle}"
        generation = stages[f"generate_pred_data_cycle_{cycle}"]
        auxiliaries = {
            f"results/data/base/cycle_0/{name}.npy"
            for name in ("hotspots", "prototypes")
        }
        assert auxiliaries <= set(stages["bootstrap"]["outs"])
        assert result_deps(generation) == auxiliaries | {f"{base}/base_data.pt"}
        assert not any("data.scaler" in parameter for parameter in generation["params"])
        assert "data.edge_affinity" in generation["params"]
        assert (
            f"results/data/prediction/cycle_{cycle}/prediction_data_graph.json"
            in generation["outs"]
        )
        incoming = f"results/data/prediction/cycle_{cycle}/pred_data.pt"
        assert result_deps(prediction) == {
            incoming,
            preprocess,
            graph_binding,
            f"results/trained/cycle_{cycle}/model.pt",
            f"{calibration}/calibration_binding.json",
            f"{calibration}/calibrator.pt",
        }
        next_base = stages[f"prepare_next_cycle_data_{cycle}"]
        assert result_deps(next_base) == {f"{base}/base_data.pt", incoming}
        assert result_deps(evaluation) == training_deps | {
            preprocess,
            graph_binding,
            f"results/trained/cycle_{cycle}/model.pt",
        }
        drift = stages[f"analyze_drift_cycle_{cycle}"]
        assert {
            preprocess,
            graph_binding,
            f"results/trained/cycle_{cycle}/model.pt",
            f"{base}/base_data.pt",
            f"results/data/prediction/cycle_{cycle}/pred_data.pt",
        } == result_deps(drift)
        assert f"results/drift_analysis/cycle_{cycle}" in drift["outs"]

    if cycles > 1:
        performance = stages["analyze_cycles_performance"]
        assert performance["cmd"].endswith(f"--cycles {cycles}")
        assert result_deps(performance) == {
            f"results/{kind}/cycle_{cycle}/metrics.json"
            for cycle in range(1, cycles + 1)
            for kind in ("evaluation", "prediction")
        }
    assert not (isolated_dvc / "results").exists()
    assert not (isolated_dvc / "dvc.lock").exists()


def result_deps(stage: dict) -> set[str]:
    """Select stage dependencies on produced artifacts."""
    return {dep for dep in stage["deps"] if dep.startswith("results/")}


def required_outputs(cycles: int, folds: int) -> set[Path]:
    """List the artifact files expected by the stages and dashboard."""
    files = {"data/base/cycle_0/hotspots.npy", "data/base/cycle_0/prototypes.npy"}
    for cycle in range(cycles + 1):
        base = f"data/base/cycle_{cycle}/"
        files.update(
            base + name
            for name in (
                "base_data.pt",
                "fold_data.pt",
                "calib_data.pt",
                "test_data.pt",
                "tsne_comparison.png",
            )
        )
        graph_stems = ["base_data_graph", "test_graph", "calib_graph"] + [
            f"{role}_graph_{fold}"
            for fold in range(1, folds + 1)
            for role in ("train", "val")
        ]
        for stem in graph_stems:
            files.update(base + f"{stem}.{ext}" for ext in ("html", "json"))

    for cycle in range(1, cycles + 1):
        files.update(
            f"data/prediction/cycle_{cycle}/{name}"
            for name in (
                "pred_data.pt",
                "prediction_data_graph.html",
                "prediction_data_graph.json",
            )
        )
        for directory, artifacts in bundle_files(cycle, folds).items():
            files.update(f"{directory}/{name}" for name in artifacts)
    files.add("cycles_performance_analysis/cycles_performance.png")
    return {Path("results", file) for file in files}


def bundle_files(cycle: int, folds: int) -> dict[str, tuple[str, ...]]:
    """List the members of directory outputs for the fixture calibration method."""
    return {
        f"trained/cycle_{cycle}": (
            "model.pt",
            "preprocessing.json",
            "graph_binding.json",
            *(f"training_plots_dataset_{fold}.png" for fold in range(1, folds + 1)),
        ),
        f"evaluation/cycle_{cycle}": (
            "calibrator.pt",
            "calibration_binding.json",
            "metrics.json",
            "reliability_diagram.png",
            "confusion_matrix.png",
            "optimum_params.txt",
            "calibration_dirichlet_loss_and_gradient.png",
        ),
        f"prediction/cycle_{cycle}": (
            "predictions.csv",
            "metrics.json",
            "histograms.png",
            "confussion_matrix_.png",
        ),
        f"drift_analysis/cycle_{cycle}": (
            "drift_metadata.json",
            "drift_results.txt",
            "distance_distributions.png",
            "marginal_distributions.png",
            "mutual_information.png",
            "pca_kpca_grid.png",
        ),
    }


@pytest.mark.integration
def test_selective_invalidation_and_artifact_restoration(isolated_dvc: Path) -> None:
    """Exercise the generated DAG and restore complete disposable artifact sets."""
    params = Path("params.yaml")
    params.write_bytes((ROOT / "params.yaml").read_bytes())
    generate_fine_grained_dvc_yaml()
    pipeline = Path("dvc.yaml")
    assert pipeline.read_bytes() == (ROOT / "dvc.yaml").read_bytes()

    config = yaml.safe_load(params.read_text(encoding="utf-8"))
    config["cycles"] = 2
    folds = config["data"]["base"]["n_splits"]
    params.write_text(yaml.safe_dump(config), encoding="utf-8")
    generate_fine_grained_dvc_yaml()
    spec = yaml.safe_load(pipeline.read_text(encoding="utf-8"))
    stages = spec["stages"]
    bundles = {
        f"results/{directory}": files
        for cycle in (1, 2)
        for directory, files in bundle_files(cycle, folds).items()
    }

    # Substitute commands while retaining every declared source and artifact edge.
    python = shlex.quote(sys.executable)
    for name, stage in stages.items():
        stage["cmd"] = f"{python} fixture.py {name}"
        for dep in stage["deps"]:
            if dep.startswith("src/"):
                path = Path(dep)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / dep, path)
    pipeline.write_text(yaml.safe_dump(spec), encoding="utf-8")
    Path("fixture.py").write_text(
        f"bundles = {bundles!r}\n"
        """import hashlib
import sys
from pathlib import Path

import yaml

name = sys.argv[1]
stage = yaml.safe_load(Path("dvc.yaml").read_text())["stages"][name]
config = yaml.safe_load(Path("params.yaml").read_text())
with Path("calls.txt").open("a") as log:
    log.write(name + "\\n")

digest = hashlib.sha256()
for dep in stage["deps"]:
    digest.update(Path(dep).read_bytes())
for key in stage.get("params", []):
    value = config
    for part in key.split("."):
        value = value[part]
    digest.update(yaml.safe_dump(value).encode())

for output in stage["outs"]:
    members = bundles.get(output)
    paths = [Path(output, member) for member in members] if members else [Path(output)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path}: {digest.hexdigest()}\\n")
""",
        encoding="utf-8",
    )

    calls = Path("calls.txt")

    def reproduce(expected: set[str]) -> None:
        calls.write_text("", encoding="utf-8")
        dvc("repro", "--no-run-cache")
        assert sorted(calls.read_text(encoding="utf-8").splitlines()) == sorted(
            expected
        )

    def change(key: str, value: object) -> None:
        *parents, leaf = key.split(".")
        section = config
        for parent in parents:
            section = section[parent]
        section[leaf] = value
        params.write_text(yaml.safe_dump(config), encoding="utf-8")

    def per_cycle(*patterns: str) -> set[str]:
        return {f"{pattern}_cycle_{cycle}" for pattern in patterns for cycle in (1, 2)}

    performance = {"analyze_cycles_performance"}
    model_consumers = per_cycle("train", "evaluate", "predict", "analyze_drift")
    reproduce(set(stages))
    reproduce(set())

    change("data.scaler_params.quantile_range", [10, 90])
    reproduce(model_consumers | performance)
    change("train.lr", 0.001)
    reproduce(model_consumers | performance)
    change("models.gcn.hidden_channels", 32)
    reproduce(model_consumers | performance)
    change("evaluate.n_bins", 12)
    reproduce(per_cycle("evaluate", "predict") | performance)
    change("analyze_drift.n_permutations", 20)
    reproduce(per_cycle("analyze_drift"))
    change("analyze_drift.seed", 2026)
    reproduce(per_cycle("analyze_drift"))
    change("train.seed", 2026)
    reproduce(model_consumers | performance)
    change("evaluate.seed", 2026)
    reproduce(per_cycle("evaluate", "predict") | performance)
    change("data.pred.seed", 2026)
    reproduce(set(stages) - {"bootstrap", "train_cycle_1", "evaluate_cycle_1"})
    change("device", "cpu")
    reproduce(model_consumers | performance)
    change("data.edge_affinity.scale", 461.0)
    reproduce(set(stages))

    for source, expected in (
        ("train_utils", model_consumers | performance),
        ("eval_utils", per_cycle("evaluate", "predict") | performance),
        ("cycles_performance_utils", performance),
        ("calibration_utils", model_consumers | performance),
        ("drift_detection_utils", per_cycle("analyze_drift")),
        ("graph_protocol", set(stages)),
    ):
        with Path(f"src/utilities/{source}.py").open("a", encoding="utf-8") as file:
            file.write("\n# Fixture implementation change\n")
        reproduce(expected)

    expected_files = required_outputs(cycles=2, folds=folds)
    original = {file: file.read_bytes() for file in expected_files}
    shutil.rmtree(isolated_dvc / "results")
    dvc("checkout")
    assert {file: file.read_bytes() for file in expected_files} == original
    reproduce(set())


@pytest.mark.parametrize("explicit_cycles", [True, False])
def test_performance_ignores_legacy_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_cycles: bool
) -> None:
    """Read only the configured range, even when stale metrics remain on disk."""
    monkeypatch.chdir(tmp_path)
    # The generated CLI argument takes precedence over a later config edit.
    Path("params.yaml").write_text(
        f"cycles: {3 if explicit_cycles else 2}\n", encoding="utf-8"
    )
    args = ["--cycles", "2"] if explicit_cycles else []
    monkeypatch.setattr(sys, "argv", ["analyze_cycles_performance.py", *args])
    metrics = {"acc": 0.5, "f1": 0.4, "mcc": 0.3}
    for stage, report in (
        ("evaluation", {"calibrated": metrics}),
        ("prediction", metrics),
    ):
        for cycle in (1, 2, 3):
            path = Path(f"results/{stage}/cycle_{cycle}/metrics.json")
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(report) if cycle <= 2 else "obsolete format",
                encoding="utf-8",
            )

    def plot(cycles_data: dict, output_dir: Path) -> Path:
        expected = [
            {"cycle": cycle, "accuracy": 0.5, "f1": 0.4, "mcc": 0.3} for cycle in (1, 2)
        ]
        assert cycles_data == {"evaluation": expected, "prediction": expected}
        assert output_dir == Path("results/cycles_performance_analysis")
        output = output_dir / "cycles_performance.png"
        output.touch()
        return output

    monkeypatch.setattr(analyze_cycles_performance, "plot_metrics", plot)
    analyze_cycles_performance.main()
    assert Path("results/cycles_performance_analysis/cycles_performance.png").is_file()


def test_resolved_export_generates_identical_dvc_definition(tmp_path, monkeypatch):
    """Preserve all generated commands, tracking and outputs for equal settings."""
    from src.configuration_files import (
        persist_configuration,
        preview_configuration,
        snapshot_configuration,
    )
    from src.configuration_yaml import load_configuration_yaml
    from src.presets import resolve_imported_configuration

    monkeypatch.chdir(tmp_path)
    source = ROOT / "config" / "defaults.yaml"
    active = Path("params.yaml")
    active.write_bytes(source.read_bytes())
    generate_fine_grained_dvc_yaml()
    original = Path("dvc.yaml").read_bytes()
    snapshot = snapshot_configuration(active)
    resolved = resolve_imported_configuration(load_configuration_yaml(active))
    preview = preview_configuration(resolved, snapshot)
    assert not preview.regeneration_paths
    persist_configuration(preview, snapshot, overwrite=True)
    generate_fine_grained_dvc_yaml()
    assert Path("dvc.yaml").read_bytes() == original
