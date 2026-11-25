"""Utility functions for training and evaluating neural network models.

This module provides:
- train: Main training function with early stopping and learning rate scheduling
- plot_training: Visualization function for training metrics and model performance
"""

import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, fbeta_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from src.utilities.calibration_utils import CalibrationMetrics
from src.utilities.early_stopping import EarlyStopping
from src.utilities.training_schedule import FoldTrainingResult


def _weighted_loss(labels, n_classes: int) -> nn.CrossEntropyLoss:
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(n_classes), y=labels.cpu().numpy()
    )
    return nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=labels.device)
    )


def _training_step(data, model, optimizer, criterion, max_grad_norm: float):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(data), data.y)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item(), grad_norm.item()


def train_final(
    data,
    model: nn.Module,
    n_classes: int,
    learning_rates: list[float],
    max_grad_norm: float,
    weight_decay: float,
) -> None:
    """Train the fresh development model for exactly the replay schedule length."""
    if not learning_rates:
        raise ValueError("Final training requires at least one scheduled epoch.")
    data = data.to(next(model.parameters()).device)
    criterion = _weighted_loss(data.y, n_classes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rates[0], weight_decay=weight_decay
    )
    for rate in learning_rates:
        optimizer.param_groups[0]["lr"] = rate
        _training_step(data, model, optimizer, criterion, max_grad_norm)


def train(
    data,
    model,
    n_classes: int,
    n_epochs: int,
    lr: float,
    max_grad_norm: float,
    weight_decay: float,
    factor_learning_rate_scheduler: float,
    patience_learning_rate_scheduler: int,
    patience_early_stopping: int,
    min_delta_early_stopping: float,
    dataset_idx: int,
    save_path: str,
) -> FoldTrainingResult:
    """Train a Graph Neural Network model with early stopping and learning rate scheduling.

    This function implements a training loop for a GNN model with various features including
    gradient clipping, class weight balancing, and multiple evaluation metrics.

    Args:
        data: The input graph data object containing features, edges, and masks for train/val splits
        model: The GNN model to be trained
        n_classes (int): The total number of classes in the dataset.
        n_epochs (int): Maximum number of training epochs
        lr (float): Initial learning rate
        max_grad_norm (float): Maximum gradient norm for gradient clipping
        weight_decay (float): L2 regularization factor
        factor_learning_rate_scheduler (float): Factor by which learning rate is reduced
        patience_learning_rate_scheduler (float): Number of epochs to wait before reducing lr
        patience_early_stopping (float): Number of epochs to wait before early stopping
        min_delta_early_stopping (float): Minimum change in validation loss to be considered as improvement
        dataset_idx (int): Index of the current dataset for saving plots
        save_path (str): Directory path where training plots will be saved

    Returns:
        Fold diagnostics with one-based completed epoch counts. The model is
        restored in place to its best validation epoch.

    The function tracks multiple metrics during training:
        - Training and validation losses
        - Validation accuracy, F1 score, and Matthews correlation coefficient
        - Expected and Maximum Calibration Error
        - Gradient norms

    Features:
        - Class-weighted loss computation for imbalanced datasets
        - Learning rate scheduling based on validation loss
        - Early stopping when validation loss stops improving
        - Gradient clipping to prevent exploding gradients
        - Comprehensive training visualization plots

    """
    if type(n_epochs) is not int or n_epochs < 1:
        raise ValueError("Cross-validation requires a positive integer epoch budget.")
    device = next(model.parameters()).device

    train_data, val_data = data
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    y_true_train = train_data.y
    y_true_val = val_data.y

    criterion_train = _weighted_loss(y_true_train, n_classes)
    criterion_val = _weighted_loss(y_true_val, n_classes)

    sample_weights_val = compute_sample_weight("balanced", y_true_val.cpu())

    best_model_state = None
    best_loss_val = float("inf")
    best_validation_epoch = None
    early_stopping_epoch = None
    learning_rates = []
    initial_lr = lr

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay
    )

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor_learning_rate_scheduler,
        patience=patience_learning_rate_scheduler,
    )
    # Initialize metrics tracking
    training_losses = []
    grad_norms = []
    validation_losses = []
    validation_accuracies = []
    validation_f1_scores = []
    validation_ece = []
    validation_mce = []
    validation_mcc = []

    last_lr = initial_lr  # Track the last learning rate
    early_stopping = EarlyStopping(
        patience=patience_early_stopping, min_delta=min_delta_early_stopping
    )
    # Training phase
    for epoch in range(n_epochs):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        loss, grad_norm = _training_step(
            train_data, model, optimizer, criterion_train, max_grad_norm
        )
        grad_norms.append(grad_norm)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            logits_val = model(val_data)
            y_prob_val = torch.softmax(logits_val, dim=1)
            loss_val = criterion_val(logits_val, y_true_val)

            cal_metrics = CalibrationMetrics()
            cal_metrics_stats = cal_metrics.calculate_metrics(
                y_prob_val,
                y_true_val,
                torch.tensor(sample_weights_val, dtype=torch.float32).to(device),
            )
            ece, mce = cal_metrics_stats.ece, cal_metrics_stats.mce
            y_pred_val = logits_val.argmax(1)

            acc = accuracy_score(
                y_true=y_true_val.cpu().numpy(),
                y_pred=y_pred_val.cpu().numpy(),
                sample_weight=sample_weights_val,
            )
            f1 = fbeta_score(
                y_true=y_true_val.cpu().numpy(),
                y_pred=y_pred_val.cpu().numpy(),
                sample_weight=sample_weights_val,
                beta=0.5,
                average="macro",
            )

            mcc = matthews_corrcoef(
                y_true=y_true_val.cpu().numpy(),
                y_pred=y_pred_val.cpu().numpy(),
                sample_weight=sample_weights_val,
            )

        validation_loss = loss_val.item()
        if np.isfinite(validation_loss) and validation_loss < best_loss_val:
            best_loss_val = validation_loss
            best_validation_epoch = epoch + 1
            best_model_state = {
                name: value.detach().to(device="cpu", copy=True)
                for name, value in model.state_dict().items()
            }

        training_losses.append(loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(acc)
        validation_f1_scores.append(f1)
        validation_mcc.append(mcc)
        validation_ece.append(ece)
        validation_mce.append(mce)

        # Update learning rate based on validation loss
        scheduler.step(loss_val)
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        # Print message if learning rate has changed
        if current_lr != last_lr:
            print(f"Learning rate decreased from {last_lr:.5e} to {current_lr:.5e}")
            last_lr = current_lr
            early_stopping.reset_counter()
        # Check early stopping
        early_stopping(loss_val)
        if early_stopping.early_stop:
            early_stopping_epoch = epoch + 1
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            print(
                f"Best validation loss {best_loss_val:.4f} at epoch {best_validation_epoch}."
            )
            break

    training_plots_path = os.path.join(
        save_path, f"training_plots_dataset_{dataset_idx}.png"
    )
    print(
        f"Saving training plots for dataset {dataset_idx} to {training_plots_path}..."
    )
    plot_training(
        training_losses,
        validation_losses,
        validation_accuracies,
        validation_f1_scores,
        validation_mcc,
        grad_norms,
        validation_ece,
        validation_mce,
        title=f"Training Plot\n dataset {dataset_idx}",
        save_path=training_plots_path,
    )

    if best_model_state is None or best_validation_epoch is None:
        raise ValueError(f"CV fold {dataset_idx} has no finite best validation epoch.")
    model.load_state_dict(best_model_state)
    return FoldTrainingResult(
        best_loss_val, best_validation_epoch, early_stopping_epoch, learning_rates
    )


def plot_training(
    train_losses,
    val_losses,
    val_accuracies,
    val_f1_scores,
    val_mcc,
    train_gradients,
    val_ece,
    val_mce,
    title="Training Plot",
    save_path: str | None = None,
):
    """Plot training and validation metrics including losses, accuracies, and calibration errors.

    Args:
        train_losses (list): Training losses over epochs
        val_losses (list): Validation losses over epochs
        val_accuracies (list): Validation accuracies over epochs
        val_f1_scores (list): Validation F1 scores (beta=0.5) over epochs
        val_mcc (list): Validation Matthews Correlation Coefficient over epochs
        train_gradients (list): Training gradient norms over epochs
        val_ece (list): Validation Expected Calibration Error over epochs
        val_mce (list): Validation Maximum Calibration Error over epochs
        title (str): Plot title
        save_path (str | None): Path to save the plot. If None, plot is not saved.

    """
    fig, axs = pyplot.subplots(2, 2, figsize=(14, 12))
    ax = axs.ravel()
    ax[0].plot(train_losses, label="Train Loss", color="blue")
    ax[0].plot(val_losses, label="Validation Loss", color="red")
    ax[0].set_title("Losses")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(val_f1_scores, label="Validation F1 scores (beta=0.5)", color="blue")
    ax[1].plot(val_accuracies, label="Validation Accuracy", color="green")
    ax[1].plot(val_mcc, label="Validation MatthewsCorrCoef", color="red")
    ax[1].set_title("Performance Metrics")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Metric")
    ax[1].legend()

    ax[2].plot(train_gradients, label="Train Gradients", color="blue")
    ax[2].set_title("Gradients")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Gradient")
    ax[2].set_yscale("log")
    ax[2].grid(True)
    ax[2].legend()

    ax[3].plot(val_ece, label="Validation Expected Calibration Error", color="blue")
    ax[3].plot(val_mce, label="Validation Maximum Calibration Error", color="red")
    ax[3].set_title("Calibration Error")
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Calibration Error")
    ax[3].legend()

    pyplot.suptitle(title)
    if save_path:
        pyplot.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig
