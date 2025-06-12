"""Training module for Graph Neural Network models.

This module provides functions for training GNN models with features including:
- Early stopping and learning rate scheduling
- Gradient clipping and class weight balancing
- Multiple evaluation metrics tracking
- Training visualization and model checkpointing
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, fbeta_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from src.models import get_model
from src.utilities.utils import (
    CalibrationMetrics,
    EarlyStopping,
    LogTime,
    plot_training,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.default_rng(seed)  # Set NumPy's random seed directly
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def main():
    """Execute the main training pipeline for the GNN model.

    This function loads the dataset, initializes the model with specified parameters,
    trains the model across multiple folds, and saves the best performing model.
    """
    dataset_path = "results/data"
    model_trained_path = "results/trained"
    os.makedirs(model_trained_path, exist_ok=True)

    fold_data = torch.load(
        os.path.join(dataset_path, "fold_data.pt"), weights_only=False
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # Get model-specific parameters
    model_params = params["models"][args.model]
    # Initialize the model
    model = get_model(args.model, model_params)

    N_EPOCHS = params["train"]["n_epochs"]
    LEARNING_RATE = params["train"]["lr"]
    WEIGHT_DEACY = params["train"]["weight_decay_adam_optimizer"]
    MAX_GRAD_NORM = params["train"]["max_grad_norm"]
    FACTOR = params["train"]["factor_learning_rate_scheduler"]
    PATIENCE_LEARNING_RATE_SCHEDULER = params["train"][
        "patience_learning_rate_scheduler"
    ]
    PATIENCE_EARLY_STOPPING = params["train"]["patience_early_stopping"]
    MIN_DELTA_EARLY_STOPPING = params["train"]["min_delta_early_stopping"]
    SEED = params["train"]["seed"]
    set_seed(seed=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for training")
    fold_results = []
    dataset_idx = 0
    with LogTime(task_name="\nTraining"):
        for graph in fold_data:
            print(f"\nTraining model on kfold {graph.fold + 1} ...\n")
            model_trained, best_loss_val = train(
                graph.to(device),
                model.to(device),
                n_epochs=N_EPOCHS,
                lr=LEARNING_RATE,
                max_grad_norm=MAX_GRAD_NORM,
                weight_decay=WEIGHT_DEACY,
                factor_learning_rate_scheduler=FACTOR,
                patience_learning_rate_scheduler=PATIENCE_LEARNING_RATE_SCHEDULER,
                patience_early_stopping=PATIENCE_EARLY_STOPPING,
                min_delta_early_stopping=MIN_DELTA_EARLY_STOPPING,
                save_path=model_trained_path,
                dataset_idx=dataset_idx + 1,
            )
            fold_results.append((model_trained, best_loss_val))
            dataset_idx += 1

    # Select the best trained model and save it
    sorted_models = sorted(fold_results, key=lambda x: x[1], reverse=True)
    model_trained_best = sorted_models[0][0]
    torch.save(
        model_trained_best.state_dict(), f"{model_trained_path}/{args.model}.pkl"
    )
    print(f"Model saved to {model_trained_path}/{args.model}.pkl\n")


if __name__ == "__main__":
    main()
