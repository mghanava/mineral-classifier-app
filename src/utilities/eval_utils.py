"""Utility module for model evaluation, calibration, and visualization.

This module provides:
- CalibrationMetrics: Class for computing model calibration metrics
- CalibratedModel: Base class for various calibration methods
- CalibrationPipeline: High-level pipeline for model calibration
- Visualization functions for confusion matrices, ROC curves, and reliability diagrams

The module supports multiple calibration methods including Temperature Scaling,
Isotonic Regression, Platt Scaling, Beta Calibration, and Dirichlet Calibration.
"""

import json
import os
from typing import Literal

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.utilities.calibration_utils import (
    CalibrationMetrics,
    CalibrationPipeline,
    CalibrationStats,
)


def evaluate_with_calibration(
    test_data,
    calibration_data,
    model,
    calibration_method: Literal[
        "temperature", "isotonic", "platt", "beta", "dirichlet"
    ],
    initial_temperature: float,
    n_bins: int,
    class_names: list,
    lr: float,
    n_epochs: int,
    weight_decay_adam_optimizer: float,
    reg_lambda: float,
    reg_mu: float,
    eps: float,
    factor_learning_rate_scheduler: float,
    patience_learning_rate_scheduler: int,
    patience_early_stopping: int,
    min_delta_early_stopping: float,
    verbose: bool = False,
    seed: int = 42,
    save_path: str | None = None,
):
    """Evaluate a model's performance with and without calibration, and save results.

    This function performs model evaluation, applies calibration, and computes various
    metrics before and after calibration. It also generates and saves visualization plots
    and metrics to disk.

    Args:
        data: PyTorch Geometric data object containing the graph data and masks
        model: PyTorch model to evaluate
        calibration_method (str): Method used for calibration ('temperature_scaling', 'vector_scaling', etc.)
        initial_temperature (float): Initial temperature parameter for temperature scaling
        n_bins (int): Number of bins to use for reliability diagram and calibration metrics
        class_names (list): List of class names for confusion matrix plotting
        lr (float): Learning rate for calibration optimization
        n_epochs (int): Number of epochs to train calibration
        weight_decay_adam_optimizer (float): Weight decay parameter for Adam optimizer during calibration
        device (torch.device): Device to run computations on
        reg_lambda (float): Regularization parameter for calibration
        reg_mu (float): Regularization parameter for calibration
        eps (float): Small value to avoid division by zero in calibration
        factor_learning_rate_scheduler (float): Factor to reduce learning rate in scheduler
        patience_learning_rate_scheduler (float): Patience for learning rate scheduler
        patience_early_stopping (float): Patience for early stopping
        min_delta_early_stopping (float): Minimum change in loss for early stopping
        verbose (bool): Whether to print progress during calibration
        save_path (str): Directory path to save outputs
        seed (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing metrics before and after calibration:
            {
                'uncalibrated': {
                    'acc': float,  # Accuracy
                    'f1': float,   # F1 score (beta=0.5)
                    'mcc': float,  # Matthews correlation coefficient
                    'ece': float,  # Expected calibration error
                    'mce': float   # Maximum calibration error
                },
                'calibrated': {
                    'acc': float,
                    'f1': float,
                    'mcc': float,
                    'ece': float,
                    'mce': float

    Side Effects:
        - Saves reliability diagram plot to '{save_path}/reliability_diagram.png'
        - Saves confusion matrix plot to '{save_path}/confusion_matrix.png'
        - Saves metrics to '{save_path}/metrics.json'
        - Saves calibrator model to '{save_path}/calibrator.pkl'

    """
    device = next(model.parameters()).device
    y_true_test = test_data.y
    y_tru_calib = calibration_data.y
    sample_weights_test = compute_sample_weight("balanced", y_true_test.cpu())
    sample_weights_test = torch.tensor(sample_weights_test, dtype=torch.float32).to(
        device
    )
    sample_weights_calib = compute_sample_weight("balanced", y_tru_calib.cpu())
    sample_weights_calib = torch.tensor(sample_weights_calib, dtype=torch.float32).to(
        device
    )
    # Get base model predictions
    model.eval()
    test_logits = model(test_data)
    calib_logits = model(calibration_data)
    # Calculate metrics before calibration
    uncalibrated_probs = F.softmax(test_logits, dim=1)
    uncalibrated_pred = uncalibrated_probs.argmax(dim=1)

    cal_metrics = CalibrationMetrics(n_bins=n_bins)
    cal_metrics_uncalibrated = cal_metrics.calculate_metrics(
        uncalibrated_probs, y_true_test, sample_weights_test
    )
    uncal_metrics = {
        "acc": accuracy_score(
            y_true_test.cpu().numpy(),
            uncalibrated_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
        ),
        "f1": fbeta_score(
            y_true_test.cpu().numpy(),
            uncalibrated_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
            beta=0.5,
            average="macro",
        ),
        "mcc": matthews_corrcoef(
            y_true_test.cpu().numpy(),
            uncalibrated_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
        ),
        "ece": cal_metrics_uncalibrated.ece,
        "mce": cal_metrics_uncalibrated.mce,
    }
    if save_path is None:
        save_path = "results/evaluation"
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving results to {save_path} ...\n")
    # Initialize calibration pipeline
    pipeline = CalibrationPipeline(base_model=model, device=device)
    pipeline.calibrate(
        calib_logits,
        y_tru_calib,
        sample_weights_calib,
        method=calibration_method,
        lr=lr,
        weight_decay_adam_optimizer=weight_decay_adam_optimizer,
        n_epochs=n_epochs,
        reg_lambda=reg_lambda,
        reg_mu=reg_mu,
        eps=eps,
        initial_temperature=initial_temperature,
        verbose=verbose,
        factor_learning_rate_scheduler=factor_learning_rate_scheduler,
        patience_learning_rate_scheduler=patience_learning_rate_scheduler,
        patience_early_stopping=patience_early_stopping,
        min_delta_early_stopping=min_delta_early_stopping,
        seed=seed,
        save_path=save_path,
    ).save(filepath=save_path)
    calibrated_probs = pipeline.predict_from_logits(test_logits)
    calibrated_pred = calibrated_probs.argmax(dim=1)
    cal_metrics_calibrated = cal_metrics.calculate_metrics(
        calibrated_probs, y_true_test, sample_weights_test, verbose=False
    )
    cal_mcc = matthews_corrcoef(
        y_true_test.cpu().numpy(),
        calibrated_pred.cpu().numpy(),
        sample_weight=sample_weights_test.cpu().numpy(),
    )
    cal_metrics = {
        "acc": accuracy_score(
            y_true_test.cpu().numpy(),
            calibrated_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
        ),
        "f1": fbeta_score(
            y_true_test.cpu().numpy(),
            calibrated_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
            beta=0.5,
            average="macro",
        ),
        "mcc": cal_mcc,
        "ece": cal_metrics_calibrated.ece,
        "mce": cal_metrics_calibrated.mce,
    }

    # Save plots to disk
    reliability_path = os.path.join(save_path, "reliability_diagram.png")
    confusion_path = os.path.join(save_path, "confusion_matrix.png")

    print(f"Saving reliability diagram to {reliability_path} ...")
    plot_reliability_diagram(
        uncalibrated_stats=cal_metrics_uncalibrated,
        calibrated_stats=cal_metrics_calibrated,
        title=f"Reliability Diagram for {calibration_method}",
        save_path=reliability_path,
    )
    print(f"Saving confusion matrix to {confusion_path} ...")
    plot_confusion_matrix(
        y_true_test.cpu().numpy(),
        calibrated_pred.cpu().numpy(),
        class_names,
        title=f"Matthews correlation coefficient {cal_mcc:.3f}",
        save_path=confusion_path,
    )
    # Save metrics as JSON file
    metrics_path = os.path.join(save_path, "metrics.json")
    print(f"Saving metrics to {metrics_path} ...")
    metrics = {"uncalibrated": uncal_metrics, "calibrated": cal_metrics}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
):
    """Plot confusion matrix using seaborn.

    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        classes (list): Class names
        title (str): Title for the confusion matrix plot
        save_path (str | None): Path to save the plot. If None, plot is not saved.

    """
    cm = confusion_matrix(y_true, y_pred)
    cm = np.round(cm).astype(int)
    pyplot.figure(figsize=(10, 8))
    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    pyplot.title(f"{title}")
    pyplot.ylabel("True Label")
    pyplot.xlabel("Predicted Label")
    pyplot.title(title)
    if save_path:
        pyplot.savefig(save_path, bbox_inches="tight", dpi=300)


def plot_reliability_diagram(
    uncalibrated_stats: CalibrationStats,
    calibrated_stats: CalibrationStats | None = None,
    title: str = "Reliability Diagram",
    save_path: str | None = None,
    figsize: tuple[int, int] = (7, 7),
):
    """Plot reliability diagram comparing uncalibrated and optionally calibrated model statistics.

    Args:
        uncalibrated_stats: CalibrationStats for the uncalibrated model
        calibrated_stats: Optional CalibrationStats for the calibrated model
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)

    """
    fig, ax1 = pyplot.subplots(figsize=figsize)
    ax1 = pyplot.gca()
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)

    # Calculate bin centers for plotting
    bin_centers = (
        uncalibrated_stats.bin_edges[:-1] + uncalibrated_stats.bin_edges[1:]
    ) / 2

    legend = f"ECE/MCE: {uncalibrated_stats.ece:.3f}/{uncalibrated_stats.mce:.3f}"
    # Plot uncalibrated statistics
    ax1.plot(
        uncalibrated_stats.bin_confidences.cpu(),
        uncalibrated_stats.bin_accuracies.cpu(),
        "ro-",
        label=f"Uncalibrated ({legend})",
    )

    # Plot calibrated statistics if provided
    if calibrated_stats is not None:
        legend = f"ECE/MCE: {calibrated_stats.ece:.3f}/{calibrated_stats.mce:.3f}"
        ax1.plot(
            calibrated_stats.bin_confidences.cpu(),
            calibrated_stats.bin_accuracies.cpu(),
            "bo-",
            label=f"Calibrated ({legend})",
        )
        # Add histogram of confidence distribution
        ax1 = pyplot.gca()
        ax2 = ax1.twinx()
        # Plot histograms with low alpha for visibility
        ax2.bar(
            bin_centers.cpu(),
            uncalibrated_stats.bin_weights.cpu(),
            width=1
            / len(uncalibrated_stats.bin_counts),  # assure no overlap among bins
            alpha=0.1,
            color="red",
            label="Uncalibrated samples",
        )
        bin_centers = (
            calibrated_stats.bin_edges[:-1] + calibrated_stats.bin_edges[1:]
        ) / 2
        ax2.bar(
            bin_centers.cpu(),
            calibrated_stats.bin_weights.cpu(),
            width=1 / len(calibrated_stats.bin_counts),  # assure no overlap among bins
            alpha=0.1,
            color="blue",
            label="Calibrated samples",
        )
    else:
        # Add histogram of confidence distribution
        ax1 = pyplot.gca()
        ax2 = ax1.twinx()
        # Plot histograms
        ax2.bar(
            bin_centers.cpu(),
            uncalibrated_stats.bin_weights.cpu(),
            width=1 / len(uncalibrated_stats.bin_counts),
            alpha=0.1,
            color="red",
            label="Uncalibrated samples",
        )

    pyplot.title(f"{title}")

    # Customize plot
    ax1.set_xlabel("Confidence\n(Mean Predicted Probability)")
    ax1.set_ylabel("Accuracy\n(Mean Observed Probability)")
    ax2.set_ylabel("Sample Count")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.text(0.7, 0.3, "Under-confident", color="red", fontsize=12, ha="center")
    ax1.text(0.3, 0.7, "Overconfident", color="red", fontsize=12, ha="center")
    ax1.grid(True, alpha=0.3)
    pyplot.tight_layout()

    if save_path:
        pyplot.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_roc_curve(ax, y_true, y_prob, sample_weights, title="ROC Curve"):
    """Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on
        y_true (array-like): True binary labels
        y_prob (array-like): Target scores (probability estimates)
        sample_weights (array-like): Sample weights
        title (str, optional): Title for the plot. Defaults to "ROC Curve"

    Returns:
        None

    """
    auc_score = roc_auc_score(y_true, y_prob, sample_weight=sample_weights)
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=sample_weights)
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="no-skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
