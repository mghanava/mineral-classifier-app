"""Check device selection, portable loads and deterministic stage streams."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.utilities.calibration_utils import CalibratedModel
from src.utilities.device import select_device
from src.utilities.drift_detection_utils import AnalyzeDrift
from src.utilities.general_utils import load_data, load_model_weights
from src.utilities.reproducibility import derive_seed, torch_seed, validate_seed


def test_scoped_seeds_are_stable_and_distinct() -> None:
    """Pin the derivation and distinguish cycles, folds and final refits."""
    scopes = [
        ("train", 1, "cv", 0),
        ("train", 1, "cv", 1),
        ("train", 1, "final_refit"),
        ("train", 2, "cv", 0),
        ("train", 2, "final_refit"),
        ("calibration", 1, "init"),
        ("drift", 1),
    ]
    seeds = [derive_seed(42, *scope) for scope in scopes]
    payload = b'["sha256-v1",42,"train",1,"cv",0]'
    assert seeds[0] == int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    assert len(set(seeds)) == len(scopes)
    assert derive_seed(None, *scopes[0]) is None
    assert validate_seed(None, "train.seed") is None
    assert validate_seed(2**32 - 1, "train.seed") == 2**32 - 1
    assert validate_seed(2**80, "data.pred.seed", maximum=None) == 2**80

    before = torch.random.get_rng_state()
    with torch_seed(seeds[0], torch.device("cpu")):
        sample = torch.rand(6)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    with torch_seed(seeds[0], torch.device("cpu")):
        torch.testing.assert_close(torch.rand(6), sample)


@pytest.mark.parametrize("value", [True, -1, 2**32, "42", 1.0])
def test_invalid_seed_reports_config_key(value: object) -> None:
    """Reject seeds that cannot be used consistently by all backends."""
    with pytest.raises(ValueError, match=r"train\.seed.*integer"):
        validate_seed(value, "train.seed")


def test_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefer visible CUDA automatically and honor explicit CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert select_device("auto") == torch.device("cuda:0")
    assert select_device("cuda") == torch.device("cuda:0")
    assert select_device("cuda:1") == torch.device("cuda:1")
    assert select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match=r"cuda:2.*unavailable.*device: cpu"):
        select_device("cuda:2")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device() == torch.device("cpu")
    with pytest.raises(ValueError, match=r"cuda.*unavailable.*device: cpu"):
        select_device("cuda")
    with pytest.raises(ValueError, match=r"Set device in params\.yaml"):
        select_device("mps")


def test_stage_cli_returns_nonzero_for_invalid_device(tmp_path: Path) -> None:
    """Surface configuration failures before creating any model artifacts."""
    (tmp_path / "params.yaml").write_text("device: mps\ntrain: {}\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "src.stages.train", "--cycle", "1", "--model", "gcn"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode != 0
    assert "Set device in params.yaml" in result.stderr
    assert "Training cycle 1 failed" in result.stdout
    assert not (tmp_path / "results").exists()


def test_graph_and_weights_load_from_cuda_tagged_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read genuine CUDA storage tags on a CPU-only test path."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.serialization, "location_tag", lambda _storage: "cuda:0")
    graph_path, weights_path = tmp_path / "graph.pt", tmp_path / "model.pt"
    tensor = torch.arange(6).reshape(3, 2)
    source, restored = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    torch.save(Data(x=tensor), graph_path)
    torch.save(source.state_dict(), weights_path)
    snapshots = {path: path.read_bytes() for path in (graph_path, weights_path)}
    with pytest.raises(RuntimeError, match="CUDA"):
        torch.load(weights_path, weights_only=True)

    graph = load_data(graph_path, "Graph")
    torch.testing.assert_close(graph.x, tensor)
    load_model_weights(restored, weights_path)
    torch.testing.assert_close(restored.weight, source.weight)
    torch.testing.assert_close(restored.bias, source.bias)
    assert {path: path.read_bytes() for path in snapshots} == snapshots


def test_seeded_calibration_and_portable_calibrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay stochastic calibration and restore its CUDA-tagged state on CPU."""
    labels = torch.arange(36) % 3
    logits = torch.stack((labels, labels.square(), -labels), dim=1).float()
    models = [
        CalibratedModel(method="dirichlet", n_epochs=2, seed=11, cycle_num=cycle)
        for cycle in (1, 1, 2)
    ]
    for model in models:
        model.fit(logits, labels)
    expected = models[0].predict_proba(logits)
    torch.testing.assert_close(
        models[1].predict_proba(logits), expected, rtol=0, atol=0
    )
    assert models[0].initialization_seed != models[2].initialization_seed
    assert not torch.equal(models[2].model.weights, models[0].model.weights)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.serialization, "location_tag", lambda _storage: "cuda:0")
    models[0].save_calibrated_model(str(tmp_path))
    path = tmp_path / "calibrator.pt"
    snapshot = path.read_bytes()
    restored = CalibratedModel.load_calibrated_model(path, torch.device("cpu"))
    torch.testing.assert_close(restored.predict_proba(logits), expected, rtol=0, atol=0)
    assert restored.initialization_seed == models[0].initialization_seed
    assert restored.seed == 11
    assert restored.classes.device.type == "cpu"
    assert path.read_bytes() == snapshot


def test_seeded_drift_replays_permutations_and_sliced_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use reproducible method-specific draws independent of ambient Torch state."""
    reference = torch.arange(24, dtype=torch.float64).reshape(12, 2)
    incoming = reference.flip(1) + 2
    populations = [Data(x=x, unscaled_features=x) for x in (reference, incoming)]
    original = torch.randperm
    orders = []

    def record(*args, **kwargs):
        permutation = original(*args, **kwargs)
        orders.append(permutation.tolist())
        return permutation

    monkeypatch.setattr(torch, "randperm", record)
    with torch.random.fork_rng(devices=[]):
        results = []
        for ambient_seed in (1, 2):
            torch.random.default_generator.manual_seed(ambient_seed)
            analyzer = AnalyzeDrift(
                *populations, seed=314, n_permutations=5, n_projections=4
            )
            results.append(
                (
                    analyzer._perform_permutation_test("mmd"),
                    analyzer._perform_permutation_test("max_sliced_wasserstein"),
                )
            )
    assert results[0] == results[1]
    assert orders[:10] == orders[10:]
    assert orders[:5] != orders[5:10]
