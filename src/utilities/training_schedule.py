"""Derive a fixed development refit schedule from cross-validation results."""

import math
from dataclasses import asdict, dataclass
from statistics import median_high, median_low


@dataclass(frozen=True)
class FoldTrainingResult:
    """Record validation selection and the rates actually used during a CV fold."""

    best_validation_loss: float
    best_validation_epoch: int
    early_stopping_epoch: int | None
    learning_rates: list[float]


def final_training_plan(folds: list[FoldTrainingResult], epoch_budget: int) -> dict:
    """Round the median best epoch half up and replay the nearest fold's rates."""
    if type(epoch_budget) is not int or epoch_budget < 1 or not folds:
        raise ValueError(
            "Final training requires CV results and a positive integer epoch budget."
        )
    best_epochs = [fold.best_validation_epoch for fold in folds]
    for fold in folds:
        epochs = fold.best_validation_epoch
        history = fold.learning_rates
        stop = fold.early_stopping_epoch
        if (
            type(epochs) is not int
            or not 1 <= epochs <= epoch_budget
            or not isinstance(history, list)
            or not epochs <= len(history) <= epoch_budget
            or (stop is None and len(history) != epoch_budget)
            or (stop is not None and (type(stop) is not int or stop != len(history)))
        ):
            raise ValueError(
                "Invalid CV epoch counts or learning-rate history for final training."
            )
        if (
            type(fold.best_validation_loss) not in (int, float)
            or not math.isfinite(fold.best_validation_loss)
            or any(
                type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0
                for rate in history
            )
        ):
            raise ValueError(
                "CV losses and learning rates must be finite, with non-negative rates."
            )

    duration = (median_low(best_epochs) + median_high(best_epochs) + 1) // 2
    source = min(
        range(len(folds)), key=lambda index: abs(best_epochs[index] - duration)
    )
    history = folds[source].learning_rates
    rates = history[:duration]
    rates += [history[-1]] * (duration - len(rates))
    return {
        "epoch_budget": epoch_budget,
        "fold_best_epochs": best_epochs,
        "cv_folds": [asdict(fold) for fold in folds],
        "epochs": duration,
        "learning_rate_strategy": "replay_cv_truncate_or_hold_last",
        "lr_source_fold_index": source,
        "learning_rates": rates,
    }
