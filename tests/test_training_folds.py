"""Regression coverage for isolated fold training and checkpoint selection."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from src.stages import train as training_stage
from src.utilities import train_utils
from src.utilities.general_utils import load_params


class FoldModel(nn.Module):
    """Use trainable weights and controlled validation scores."""

    def __init__(self, fold_idx: int) -> None:
        """Give each fold a distinct starting weight."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(fold_idx)))
        self.fold_idx = fold_idx
        self.validation_weights: list[torch.Tensor] = []
        self.validation_folds: list[int] = []

    def forward(self, data: Data) -> torch.Tensor:
        """Make the first epoch best and identify the validation input."""
        if data.is_validation:
            self.validation_folds.append(data.fold_idx)
            self.validation_weights.append(self.weight.detach().clone())
            margin = 3 if self.fold_idx == 1 else 1
            margin -= len(self.validation_weights) - 1
            return torch.nn.functional.one_hot(data.y, num_classes=2).float() * margin
        return self.weight * data.x


def test_folds_train_independently_and_save_winning_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run two tiny folds and verify metrics and weights come from the correct fold."""
    params = load_params()
    params["data"]["base"]["n_classes"] = 2
    params["train"].update(
        n_epochs=3, patience_early_stopping=1, patience_learning_rate_scheduler=2
    )
    folds = [
        (
            Data(x=torch.eye(2), y=torch.tensor([1, 0]), is_validation=False),
            Data(y=torch.tensor([0, 1]), is_validation=True, fold_idx=fold_idx),
        )
        for fold_idx in (1, 2)
    ]
    torch.save(folds, tmp_path / "fold_data.pt")
    models: list[FoldModel] = []

    def create_model(_name: str, _params: dict) -> FoldModel:
        model = FoldModel(len(models) + 1)
        models.append(model)
        return model

    adam_factory = torch.optim.Adam
    scheduler_factory = torch.optim.lr_scheduler.ReduceLROnPlateau
    stopper_factory = train_utils.EarlyStopping
    optimizers = []
    schedulers = []
    stoppers = []

    def create_optimizer(parameters, **kwargs):
        optimizer = adam_factory(parameters, **kwargs)
        assert not optimizer.state
        optimizers.append(optimizer)
        return optimizer

    def create_scheduler(optimizer, **kwargs):
        scheduler = scheduler_factory(optimizer, **kwargs)
        assert scheduler.best == float("inf")
        schedulers.append(scheduler)
        return scheduler

    def create_stopper(**kwargs):
        stopper = stopper_factory(**kwargs)
        assert stopper.epochs_without_improvement == 0
        assert not stopper.early_stop
        stoppers.append(stopper)
        return stopper

    plot_training = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training_stage, "get_model", create_model)
    monkeypatch.setattr(torch.optim, "Adam", create_optimizer)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", create_scheduler)
    monkeypatch.setattr(train_utils, "EarlyStopping", create_stopper)
    monkeypatch.setattr(train_utils, "plot_training", plot_training)

    training_stage.run_training(
        {"base_data": tmp_path, "output": tmp_path}, params, "gcn", cycle_num=1
    )

    assert len(models) == 2
    assert len(optimizers) == len(schedulers) == len(stoppers) == 2
    assert optimizers[0] is not optimizers[1]
    assert schedulers[0] is not schedulers[1]
    assert stoppers[0] is not stoppers[1]
    assert plot_training.call_count == 2
    for fold_idx, model in enumerate(models, 1):
        optimizer = optimizers[fold_idx - 1]
        assert optimizer.param_groups[0]["params"][0] is model.weight
        assert optimizer.state[model.weight]["step"].item() == 2
        assert schedulers[fold_idx - 1].optimizer is optimizer
        assert stoppers[fold_idx - 1].early_stop
        assert model.validation_folds == [fold_idx, fold_idx]
        first_weight, second_weight = model.validation_weights
        assert first_weight.item() == pytest.approx(fold_idx - params["train"]["lr"])
        assert second_weight.item() < first_weight.item()
        torch.testing.assert_close(model.weight, first_weight)

        plot = plot_training.call_args_list[fold_idx - 1]
        assert (
            Path(plot.kwargs["save_path"]).name
            == f"training_plots_dataset_{fold_idx}.png"
        )
        expected_losses = torch.nn.functional.softplus(
            torch.tensor([-3.0, -2.0] if fold_idx == 1 else [-1.0, 0.0])
        ).tolist()
        assert plot.args[1] == pytest.approx(expected_losses)
        assert all(len(series) == 2 for series in plot.args)

    state = torch.load(tmp_path / "model.pt", weights_only=True)
    torch.testing.assert_close(state["weight"], models[0].validation_weights[0])
    assert "Fold 1 model (validation loss" in capsys.readouterr().out


def test_training_and_validation_modes_each_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the mode and gradient state of every CPU forward over three epochs."""
    model = FoldModel(fold_idx=1).eval()
    train_data = Data(x=torch.eye(2), y=torch.tensor([1, 0]), is_validation=False)
    val_data = Data(y=torch.tensor([0, 1]), is_validation=True, fold_idx=1)
    forwards: list[tuple[bool, bool, bool]] = []

    def record_forward(module: nn.Module, inputs: tuple[Data]) -> None:
        (data,) = inputs
        forwards.append((data.is_validation, module.training, torch.is_grad_enabled()))

    model.register_forward_pre_hook(record_forward)
    monkeypatch.setattr(train_utils, "plot_training", Mock())

    train_utils.train(
        (train_data, val_data),
        model,
        n_classes=2,
        n_epochs=3,
        lr=0.01,
        max_grad_norm=1.0,
        weight_decay=0.0,
        factor_learning_rate_scheduler=0.5,
        patience_learning_rate_scheduler=3,
        patience_early_stopping=3,
        min_delta_early_stopping=0.0,
        dataset_idx=1,
        save_path=str(tmp_path),
    )

    training_forward = (False, True, True)
    validation_forward = (True, False, False)
    assert forwards == [training_forward, validation_forward] * 3
