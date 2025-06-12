"""Evaluation module for assessing model performance and calibration.

This module provides functionality to evaluate trained models, including:
- Model calibration assessment
- Metrics calculation (accuracy, F1, MCC)
- Generation of reliability diagrams and confusion matrices
- Saving of evaluation results and calibrated models
"""

import argparse
import json
import os
from typing import Literal, cast

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, fbeta_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight

from src.models import get_model
from src.utilities.utils import (
    CalibrationMetrics,
    LogTime,
    ModelCalibration,
    plot_confusion_matrix,
    plot_reliability_diagram,
)


def evaluate_with_calibration(
    data,
    model,
    calibration_method: str,
    initial_temperature: float,
    n_bins: int,
    class_names: list,
    lr: float,
    n_epochs: int,
    weight_decay: float,
    factor_learning_rate_scheduler: float,
    patience_learning_rate_scheduler: float,
    patience_early_stopping: float,
    min_delta_early_stopping: float,
    verbose: bool,
    save_path: str,
    device: torch.device,
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
        weight_decay (float): Weight decay parameter for Adam optimizer during calibration
        factor_learning_rate_scheduler (float): Factor to reduce learning rate in scheduler
        patience_learning_rate_scheduler (float): Patience for learning rate scheduler
        patience_early_stopping (float): Patience for early stopping
        min_delta_early_stopping (float): Minimum change in loss for early stopping
        verbose (bool): Whether to print progress during calibration
        save_path (str): Directory path to save outputs
        device (torch.device): Device to run computations on

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
    y_true_test = data.y[data.test_mask]
    y_tru_calib = data.y[data.calib_mask]
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
    logits = model(data)
    base_probs = F.softmax(logits, dim=1)
    # Calculate metrics before calibration
    uncal_probs = base_probs[data.test_mask]
    uncal_pred = uncal_probs.argmax(dim=1)

    cal_metrics = CalibrationMetrics(n_bins=n_bins)
    cal_metrics_uncalibrated = cal_metrics.calculate_metrics(
        uncal_probs, y_true_test, sample_weights_test
    )
    uncal_metrics = {
        "acc": accuracy_score(
            y_true_test.cpu().numpy(),
            uncal_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
        ),
        "f1": fbeta_score(
            y_true_test.cpu().numpy(),
            uncal_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
            beta=0.5,
            average="macro",
        ),
        "mcc": matthews_corrcoef(
            y_true_test.cpu().numpy(),
            uncal_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
        ),
        "ece": cal_metrics_uncalibrated.ece,
        "mce": cal_metrics_uncalibrated.mce,
    }

    calib_logits = logits[data.calib_mask]
    calibrator = ModelCalibration(
        method=cast(
            Literal["temperature", "isotonic", "platt", "beta", "dirichlet"],
            calibration_method,
        ),
        initial_temperature=initial_temperature,
        device=device,
        lr=lr,
        weight_decay_adam_optimizer=weight_decay,
        n_epochs=n_epochs,
        verbose=verbose,
        patience_early_stopping=patience_early_stopping,
        factor_learning_rate_scheduler=factor_learning_rate_scheduler,
        patience_learning_rate_scheduler=patience_learning_rate_scheduler,
        min_delta_early_stopping=min_delta_early_stopping,
        save_path=save_path,
    )
    calibrator.fit(calib_logits, y_tru_calib, sample_weights_calib)
    cal_probs = calibrator.predict_probability(logits[data.test_mask])
    cal_pred = cal_probs.argmax(dim=1)
    cal_metrics_calibrated = cal_metrics.calculate_metrics(
        cal_probs, y_true_test, sample_weights_test, verbose=False
    )

    cal_metrics = {
        "acc": accuracy_score(
            y_true_test.cpu().numpy(),
            cal_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
        ),
        "f1": fbeta_score(
            y_true_test.cpu().numpy(),
            cal_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
            beta=0.5,
            average="macro",
        ),
        "mcc": matthews_corrcoef(
            y_true_test.cpu().numpy(),
            cal_pred.cpu().numpy(),
            sample_weight=sample_weights_test.cpu().numpy(),
        ),
        "ece": cal_metrics_calibrated.ece,
        "mce": cal_metrics_calibrated.mce,
    }

    # Save plots to disk
    reliability_path = os.path.join(save_path, "reliability_diagram.png")
    confusion_path = os.path.join(save_path, "confusion_matrix.png")

    print(f"Saving reliability diagram to {reliability_path}!")
    plot_reliability_diagram(
        uncalibrated_stats=cal_metrics_uncalibrated,
        calibrated_stats=cal_metrics_calibrated,
        title=f"Reliability Diagram for {calibrator.method}",
        save_path=reliability_path,
    )
    print(f"Saving confusion matrix to {confusion_path}!")
    plot_confusion_matrix(
        y_true_test.cpu().numpy(),
        cal_pred.cpu().numpy(),
        sample_weights_test.cpu().numpy(),
        class_names,
        save_path=confusion_path,
    )
    # Save metrics as JSON file
    metrics_path = os.path.join(save_path, "metrics.json")
    print(f"Saving metrics to {metrics_path}")
    metrics = {"uncalibrated": uncal_metrics, "calibrated": cal_metrics}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    # Save the calibrator
    file_name = f"{save_path}/calibrator.pkl"
    print(f"Saving calibrated model to {file_name}\n")
    calibrator.save(file_name)

    return metrics


def main():
    """Run evaluation script.

    This function parses command-line arguments, loads the model and evaluation data,
    performs model evaluation with calibration, saves evaluation results, and prints progress.
    """
    dataset_path = "results/data"
    model_trained_path = "results/trained"
    evaluation_path = "results/evaluation"
    os.makedirs(evaluation_path, exist_ok=True)

    test_data = torch.load(
        os.path.join(dataset_path, "test_data.pt"), weights_only=False
    )

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    # Get model-specific parameters
    model_params = params["models"][args.model]
    # Initialize a new model instance
    model = get_model(args.model, model_params)
    model.load_state_dict(
        torch.load(f"{model_trained_path}/{args.model}.pkl", weights_only=True)
    )

    INITIAL_TEMPERATURE = params["evaluate"]["initial_temperature"]
    CALIBRATION_METHOD = params["evaluate"]["calibration_method"]
    N_BINS = params["evaluate"]["n_bins"]
    CLASS_NAMES = params["evaluate"]["class_names"]
    N_EPOCHS = params["evaluate"]["n_epochs"]
    LEARNING_RATE = params["evaluate"]["lr"]
    WEIGHT_DEACY = params["evaluate"]["weight_decay_adam_optimizer"]
    FACTOR = params["evaluate"]["factor_learning_rate_scheduler"]
    PATIENCE_LEARNING_RATE_SCHEDULER = params["evaluate"][
        "patience_learning_rate_scheduler"
    ]
    PATIENCE_EARLY_STOPPING = params["evaluate"]["patience_early_stopping"]
    MIN_DELTA_EARLY_STOPPING = params["evaluate"]["min_delta_early_stopping"]
    VERBOSE = params["evaluate"]["verbose"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for evaluation\n")
    with LogTime(task_name="\nEvaluation"):
        print("Assessing test dataset ...\n")
        evaluate_with_calibration(
            data=test_data.to(device),
            model=model.to(device),
            calibration_method=CALIBRATION_METHOD,
            initial_temperature=INITIAL_TEMPERATURE,
            n_bins=N_BINS,
            class_names=CLASS_NAMES,
            lr=LEARNING_RATE,
            n_epochs=N_EPOCHS,
            weight_decay=WEIGHT_DEACY,
            factor_learning_rate_scheduler=FACTOR,
            patience_learning_rate_scheduler=PATIENCE_LEARNING_RATE_SCHEDULER,
            patience_early_stopping=PATIENCE_EARLY_STOPPING,
            min_delta_early_stopping=MIN_DELTA_EARLY_STOPPING,
            save_path=evaluation_path,
            verbose=VERBOSE,
            device=device,
        )


if __name__ == "__main__":
    main()
