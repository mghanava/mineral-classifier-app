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


def train(
    data,
    model,
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
):
    """Train a Graph Neural Network model with early stopping and learning rate scheduling.

    This function implements a training loop for a GNN model with various features including
    gradient clipping, class weight balancing, and multiple evaluation metrics.

    Args:
        data: The input graph data object containing features, edges, and masks for train/val splits
        model: The GNN model to be trained
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
        tuple: (trained_model, best_validation_loss)
            - trained_model: The trained GNN model
            - best_validation_loss: The lowest validation loss achieved during training

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
    y_true_train = data.y[data.train_mask]
    classes_train = np.unique(y_true_train.cpu().numpy())

    device = y_true_train.device

    class_weights_train = compute_class_weight(
        class_weight="balanced",
        classes=classes_train,
        y=y_true_train.cpu().numpy(),
    )
    criterion_train = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights_train, dtype=torch.float32).to(device)
    )
    y_true_val = data.y[data.val_mask]
    classes_val = np.unique(y_true_val.cpu().numpy())
    class_weights_val = compute_class_weight(
        class_weight="balanced",
        classes=classes_val,
        y=y_true_val.cpu().numpy(),
    )
    criterion_val = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights_val, dtype=torch.float32).to(device)
    )

    sample_weights_val = compute_sample_weight("balanced", y_true_val.cpu())

    best_model_state = None
    best_loss_val = float("inf")
    initial_lr = lr
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

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        logits_train = logits[data.train_mask]
        loss = criterion_train(logits_train, y_true_train)
        loss.backward()

        # Calculate the total gradient norm (magnitude) across all parameters in
        # the model before clipping to see if gradients are exploding or vanishing
        total_norm = 0.0
        # Iterate through all parameters in the model
        for p in model.parameters():
            # Check if parameter has gradients (some layers might not)
            if p.grad is not None:
                # Calculate L2 norm (Euclidean norm) of gradients for this parameter
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        grad_norms.append(total_norm)
        # apply Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            logits = model(data)
            logits_val = logits[data.val_mask]
            loss_val = criterion_val(logits_val, y_true_val)

            cal_metrics = CalibrationMetrics()
            cal_metrics_stats = cal_metrics.calculate_metrics(
                logits_val,
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

        # Save best model state
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

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
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            print(f"Best validation loss achieved: {early_stopping.best_score:.4f}\n")
            break

        training_losses.append(loss.item() if loss else 0)
        validation_losses.append(loss_val.item() if loss else 0)
        validation_accuracies.append(acc)
        validation_f1_scores.append(f1)
        validation_mcc.append(mcc)
        validation_ece.append(ece)
        validation_mce.append(mce)

    training_plots_path = os.path.join(
        save_path, f"training_plots_dataset_{dataset_idx}.png"
    )
    print(f"Saving training plots for dataset {dataset_idx} to {training_plots_path}!")
    plot_training(
        training_losses,
        validation_losses,
        validation_accuracies,
        validation_f1_scores,
        validation_mcc,
        grad_norms,
        validation_ece,
        validation_mce,
        title="Training Plot",
        save_path=training_plots_path,
    )

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        return model, best_loss_val

    return model, min(validation_losses)


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
