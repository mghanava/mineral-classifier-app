"""Verify the deterministic CV-informed duration and learning-rate replay."""

from dataclasses import replace

import pytest

from src.utilities.training_schedule import FoldTrainingResult, final_training_plan


@pytest.mark.parametrize(
    ("best_epochs", "duration", "source"),
    [([1, 3, 5], 3, 1), ([1, 4], 3, 1), ([5, 1], 3, 0), ([1, 5], 3, 0), ([6, 6], 6, 0)],
)
def test_median_rounding_and_representative_fold(
    best_epochs: list[int], duration: int, source: int
) -> None:
    """Use the median rounded half up and break equally close ties by order."""
    folds = [
        FoldTrainingResult(0.1 + index, epoch, None, [1 / (index + 1)] * 6)
        for index, epoch in enumerate(best_epochs)
    ]
    plan = final_training_plan(folds, 6)
    assert plan["fold_best_epochs"] == best_epochs
    assert plan["epochs"] == duration
    assert plan["lr_source_fold_index"] == source
    assert plan["learning_rates"] == folds[source].learning_rates[:duration]


def test_short_representative_history_extends_its_last_used_rate() -> None:
    """Distinguish best and stop counts and hold the last rate after early stopping."""
    early = FoldTrainingResult(0.5, 1, 2, [0.01, 0.005])
    full = FoldTrainingResult(0.1, 5, None, [0.01] * 5)
    plan = final_training_plan([early, full], 5)
    assert plan["epochs"] == 3
    assert plan["lr_source_fold_index"] == 0
    assert plan["learning_rates"] == [0.01, 0.005, 0.005]


@pytest.mark.parametrize("epochs", [0, -1, 4, True, 1.5])
def test_invalid_best_epochs_raise(epochs: int) -> None:
    """Reject nonpositive, nonintegral and over-budget validation epochs."""
    with pytest.raises(ValueError, match="Invalid CV epoch counts"):
        final_training_plan([FoldTrainingResult(0.5, epochs, None, [0.1] * 3)], 3)


def test_invalid_budget_or_history_raises() -> None:
    """Reject absent results, invalid budgets and inconsistent stop provenance."""
    valid = FoldTrainingResult(0.5, 2, 3, [0.1] * 3)
    for folds, budget in (([], 4), ([valid], 0), ([valid], True)):
        with pytest.raises(ValueError, match="positive integer epoch budget"):
            final_training_plan(folds, budget)
    for result in (
        replace(valid, learning_rates=[]),
        replace(valid, early_stopping_epoch=2),
        replace(valid, early_stopping_epoch=None),
        replace(valid, learning_rates=[0.1, float("nan"), 0.1]),
        replace(valid, best_validation_loss=float("inf")),
    ):
        with pytest.raises(ValueError, match=r"Invalid CV|must be finite"):
            final_training_plan([result], 4)
