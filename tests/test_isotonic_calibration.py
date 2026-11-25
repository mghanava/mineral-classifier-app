"""Verify class-specific isotonic inference and portable fitted artifacts."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.utilities.calibration_utils import CalibratedModel


@pytest.fixture
def multiclass() -> tuple[CalibratedModel, torch.Tensor, torch.Tensor]:
    """Fit three distinct piecewise linear maps with analytically known knots."""
    model = CalibratedModel(method="isotonic", eps=0.4)
    model.fit(
        torch.tensor([[2.0, -2.0, -5.0], [-2.0, 4.0, -2.0], [0.0, 1.0, 6.0]]),
        torch.tensor([0, 1, 2]),
    )
    # The last row has positive mass below eps, so it must still sum to one.
    logits = torch.tensor([[1.0, 2.5, 2.0], [3.0, -3.0, 2.0], [0.125, 1.1875, -1.5]])
    knots = ((-2, 0, 2), (-2, 1, 4), (-5, -2, 6))
    scores = np.column_stack(
        [
            np.interp(column, x, [0, 0, 1])
            for column, x in zip(logits.T, knots, strict=True)
        ]
    )
    expected = torch.tensor(
        scores / scores.sum(axis=1, keepdims=True), dtype=logits.dtype
    )
    return model, logits, expected


def test_multiclass_isotonic_uses_distinct_fitted_maps(
    multiclass: tuple[CalibratedModel, torch.Tensor, torch.Tensor],
) -> None:
    """Match independent interpolation and normalize each positive row."""
    model, logits, expected = multiclass
    probabilities = model.predict_proba(logits)
    torch.testing.assert_close(probabilities, expected)
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(len(logits)))


def test_multiclass_isotonic_rejects_zero_mass(
    multiclass: tuple[CalibratedModel, torch.Tensor, torch.Tensor],
) -> None:
    """Report all-zero transformed rows rather than emitting invalid probabilities."""
    model, _, _ = multiclass
    with pytest.raises(ValueError, match="zero total probability"):
        model.predict_proba(torch.full((1, 3), -10.0))


def test_multiclass_isotonic_requires_one_map_per_column(
    multiclass: tuple[CalibratedModel, torch.Tensor, torch.Tensor],
) -> None:
    """Reject probabilities with a missing or extra fitted class map."""
    model, logits, _ = multiclass
    with pytest.raises(ValueError, match="one fitted calibrator per logit column"):
        model.predict_proba(logits[:, :2])


def test_binary_isotonic_retains_complementary_probabilities() -> None:
    """Keep the single-logit binary path and its valid zero positive score."""
    model = CalibratedModel(method="isotonic")
    model.fit(torch.tensor([-2.0, 2.0]), torch.tensor([0, 1]))
    probabilities = model.predict_proba(torch.tensor([-3.0, -1.0, 3.0]))
    torch.testing.assert_close(
        probabilities, torch.tensor([[1.0, 0.0], [0.75, 0.25], [0.0, 1.0]])
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_multiclass_isotonic_loads_cuda_tagged_artifacts_on_cpu(
    multiclass: tuple[CalibratedModel, torch.Tensor, torch.Tensor],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    """Restore all fitted thresholds from trusted current and legacy artifacts."""
    model, logits, expected = multiclass
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.serialization, "location_tag", lambda _storage: "cuda:0")
    model.save_calibrated_model(str(tmp_path))
    path = tmp_path / "calibrator.pt"
    if legacy:
        state = torch.load(path, weights_only=False, map_location="cpu")
        del state["seed_provenance"]
        torch.save(state, path)
    snapshot = path.read_bytes()
    restored = CalibratedModel.load_calibrated_model(path, torch.device("cpu"))
    probabilities = restored.predict_proba(logits)
    assert probabilities.device.type == "cpu"
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(probabilities, model.predict_proba(logits))
    assert path.read_bytes() == snapshot
