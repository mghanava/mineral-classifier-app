"""Utility functions and classes for geospatial data processing, model calibration, and visualization.

This module provides:
- Data generation and preprocessing utilities for geospatial datasets.
- Calibration metrics and model calibration classes (Temperature Scaling, Platt, Beta, Dirichlet, Isotonic).
- Graph construction and visualization helpers for PyTorch Geometric.
- Plotting functions for training, calibration, and evaluation metrics.
"""

import os
import time
from typing import Any, ClassVar, Literal, NamedTuple

import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot
from scipy.spatial import distance_matrix
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data


class LogTime:
    """A context manager for measuring execution time of code blocks and functions.

    Usage as a context manager:
    ```
    with ExecutionTimer(name="My Task"):
        # code to measure
        time.sleep(1)
    ```

    """

    def __init__(self, task_name: str):
        """Initialize the LogTime context manager with a task name."""
        self.task_name = task_name

    def __enter__(self):
        """Start timing the execution of the code block."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """End timing and print the execution time of the code block."""
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print(f"{self.task_name} executed in {self.execution_time:.3f} seconds.\n")
        return False  # Propagate any exceptions


class CalibrationStats(NamedTuple):
    """Container for calibration metrics and reliability diagram statistics."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    bin_unweighted_calibration_errors: (
        torch.Tensor
    )  # Unweighted Calibration errors for each bin
    bin_confidences: torch.Tensor  # Confidence values for each bin
    bin_accuracies: torch.Tensor  # Accuracy values for each bin
    bin_weights: torch.Tensor  # Sample weights for each bin
    bin_edges: torch.Tensor  # Bin edge values
    bin_counts: torch.Tensor  # Number of samples in each bin


class CalibrationMetrics:
    """Calculate calibration metrics for model predictions.

    Measures how often the model is correct when it makes predictions at all confidences.
        Group Predictions: Create groups of predictions based on predicted
            probabilities.
        Calculate Accuracy: For each group, calculate the accuracy
            (fraction of correct predictions).
        Calculate Confidence: For each group, calculate the average
            predicted probability.
        Calculate Absolute Difference: For each group, calculate the absolute
            difference between accuracy and confidence.
        Weight by group frequency or weight: Weight these differences by
            the number of predictions in each group or their weights.
        ECE: Sum up the weighted differences
        MCE: Get maximum of unweighted differences

    Args:
        n_bins (int): Number of bins for binning confidence scores
        binary_case_prob_threshold (float): Threshold for binary classification
        min_samples_per_bin: Minimum samples required per bin

    """

    def __init__(
        self,
        n_bins: int = 10,
        binary_case_prob_threshold: float = 0.5,
        min_samples_per_bin: int = 10,
    ):
        """Initialize the CalibrationMetrics class.

        Args:
            n_bins (int): Number of bins for binning confidence scores.
            binary_case_prob_threshold (float): Threshold for binary classification.
            min_samples_per_bin (int): Minimum samples required per bin.

        """
        if n_bins < 1:
            raise ValueError("Number of bins must be positive")
        if not 0 < binary_case_prob_threshold < 1:
            raise ValueError(
                "Binary case probability threshold must be between 0 and 1"
            )

        self.n_bins = n_bins
        self.binary_case_prob_threshold = binary_case_prob_threshold
        self.min_samples_per_bin = min_samples_per_bin

    def _adaptive_binning(self, max_probs: torch.Tensor, n_bins: int) -> torch.Tensor:
        """Create adaptive bin edges based on prediction distribution.

        Args:
            max_probs: Prediction probabilities
            n_bins: Target number of bins

        Returns:
            torch.Tensor: Bin edges

        """
        # Sort probabilities
        sorted_probs = torch.sort(max_probs)[0]

        # Calculate target samples per bin
        target_bin_size = len(max_probs) // n_bins

        if target_bin_size < self.min_samples_per_bin:
            # Reduce number of bins if not enough samples
            n_bins = max(1, len(max_probs) // self.min_samples_per_bin)
            target_bin_size = len(max_probs) // n_bins

        # Create adaptive bin edges
        bin_edges = [0.0]
        for i in range(1, n_bins):
            idx = i * target_bin_size
            if idx < len(sorted_probs):
                edge = sorted_probs[idx].item()
                if edge > bin_edges[-1] and edge < 1.0:
                    bin_edges.append(edge)
        bin_edges.append(1.0)

        return torch.tensor(bin_edges, device=max_probs.device)

    def calculate_metrics(
        self,
        y_prob: torch.Tensor,
        y_true: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        adaptive_binning: bool = False,
        verbose: bool = False,
    ) -> CalibrationStats:
        """Compute calibration metrics.

        Args:
            y_prob (torch.Tensor): Predicted probabilities
            y_true (torch.Tensor): True labels
            sample_weights (torch.Tensor, optional): Sample weights
            adaptive_binning: Whether to use adaptive binning
            verbose: Whether to print binning statistics

        Returns:
            CalibrationStats: Calibration metrics and reliability diagram statistics

        """
        # Handle binary and multi-class cases
        is_binary = y_prob.dim() == 1 or y_prob.shape[1] == 1
        if is_binary:
            # ensure consistent handling of binary classification probabilities
            y_prob = y_prob.view(-1)
            pred_classes = (y_prob >= self.binary_case_prob_threshold).long()
            # confidence or probability of class correctness
            max_probs = torch.where(pred_classes == 1, y_prob, 1 - y_prob)
        else:
            # Multi-class case
            # assure sum of probabilities for each sample is 1 otherwise treat tensor
            # as logits and apply softmax to normalize
            floating_point_tolerance = 1e-3
            if torch.any(torch.abs(y_prob.sum(dim=1) - 1) > floating_point_tolerance):
                y_prob = F.softmax(y_prob, dim=1)
            max_probs, pred_classes = torch.max(y_prob, dim=1)

        max_probs = max_probs.to(dtype=torch.float32)

        # Default sample weights if not provided
        if sample_weights is None:
            sample_weights = torch.ones_like(y_true, dtype=torch.float32)
        if torch.any(sample_weights < 0):
            raise ValueError("Sample weights must be non-negative")
        # assure same device
        sample_weights = sample_weights.to(y_true.device)
        # Normalize weights
        total_weight = sample_weights.sum()
        if total_weight == 0:
            raise ValueError("Sum of sample weights must be positive")

        # Create boolean mask for correct predictions
        correct = (pred_classes == y_true).to(dtype=torch.float32)

        # Get bin edges (either adaptive or uniform)
        if adaptive_binning:
            bin_edges = self._adaptive_binning(max_probs, self.n_bins)
        else:
            bin_edges = torch.linspace(0, 1, self.n_bins + 1, device=y_prob.device)
        # initialize bin counts to store number of samples in each bin
        bin_counts = torch.zeros(self.n_bins, device=y_prob.device)
        # Compute binning metrics for non-empty bins
        bin_confidences_list = []
        bin_accuracies_list = []
        bin_weights_list = []

        # Compute binning metrics
        for i in range(self.n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            # Handle edge cases for first bin to include exact 0
            if i == 0:
                bin_mask = torch.logical_and(
                    max_probs >= bin_lower, max_probs <= bin_upper
                )
            else:
                bin_mask = torch.logical_and(
                    max_probs > bin_lower, max_probs <= bin_upper
                )

            bin_counts[i] = bin_mask.sum()
            if bin_mask.sum() > 0:
                # Bin weighted count
                bin_weight = sample_weights[bin_mask].sum()
                # Bin accuracy (weighted proportion of correct predictions in this bin)
                bin_acc = (correct[bin_mask] @ sample_weights[bin_mask]) / bin_weight
                # Bin confidence (weighted average of max probabilities in this bin)
                bin_conf = (max_probs[bin_mask] @ sample_weights[bin_mask]) / bin_weight

                # append stats of each bin
                bin_weights_list.append(bin_weight)
                bin_accuracies_list.append(bin_acc)
                bin_confidences_list.append(bin_conf)

        # Handle case where all predictions fall into a single bin
        if len(bin_weights_list) == 0:
            raise ValueError("No valid predictions found in any bin")

        # Convert lists to tensors
        bin_weights = torch.tensor(bin_weights_list, device=y_prob.device)
        bin_accuracies = torch.tensor(bin_accuracies_list, device=y_prob.device)
        bin_confidences = torch.tensor(bin_confidences_list, device=y_prob.device)
        # find single metric calibration error from calibration errors of all bins
        bin_errors = bin_accuracies - bin_confidences
        # ece is the weighted average of the bins’ accuracy/confidence absolute difference
        ece = (torch.abs(bin_errors) @ bin_weights / total_weight).item()
        # mce is the maximum of the bins’ accuracy/confidence absolute difference
        mce = torch.max(torch.abs(bin_errors)).item()

        # Filter bin edges to only include edges of non-empty bins (redundant for adaptive binning)
        non_empty_bins = torch.where(bin_counts > 0)[0]
        filtered_edges = torch.cat(
            [bin_edges[non_empty_bins], bin_edges[non_empty_bins[-1] + 1].unsqueeze(0)]
        )
        if verbose:
            # Add diagnostic information
            print("\nCalibration Diagnostics:")
            print(f"Number of bins used: {len(bin_edges) - 1}")
            print(
                f"Average samples per bin: {len(max_probs) / (len(bin_edges) - 1):.1f}"
            )
            print(f"Prediction range: [{max_probs.min():.3f}, {max_probs.max():.3f}]")
            print(f"Empty bins: {(bin_counts == 0).sum().item()}")

        # Create and return CalibrationStats object
        return CalibrationStats(
            ece=ece,
            mce=mce,
            bin_unweighted_calibration_errors=bin_errors,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_weights=bin_weights,
            bin_edges=filtered_edges,
            bin_counts=bin_counts,
        )


class EarlyStopping:
    """Implements early stopping to terminate training when a monitored metric stops improving.

    Attributes
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    best_loss : float
        Best score seen so far.
    epochs_without_improvement : int
        Number of epochs since last improvement.
    early_stop : bool
        Whether early stopping has been triggered.
    decreasing_score : bool
        Whether a lower score is considered better.

    Methods
    -------
    __call__(loss_val)
        Call to update early stopping state with new loss value.
    reset_counter()
        Reset the counter for epochs without improvement.
    best_score
        Property to get the best score seen so far.

    """

    def __init__(self, patience=50, min_delta=0.001, decreasing_score: bool = True):
        """Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as an improvement.
            decreasing_score (bool): Whether a lower score is considered better.

        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf") if decreasing_score else -float("inf")
        self.epochs_without_improvement = 0
        self.early_stop = False
        self.decreasing_score = decreasing_score

    def __call__(self, loss_val):
        """Update early stopping state with new loss value.

        Args:
            loss_val (float): The current loss value to evaluate for early stopping.

        """
        stop_check_condition = (
            loss_val < self.best_loss - self.min_delta
            if self.decreasing_score
            else loss_val > self.best_loss + self.min_delta
        )
        if stop_check_condition:
            self.best_loss = loss_val
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.early_stop = True

    def reset_counter(self):
        """Reset the counter when learning rate changes."""
        self.epochs_without_improvement = 0

    @property
    def best_score(self):
        """Return the best score seen so far."""
        return self.best_loss


class TemperatureScaling(nn.Module):
    """Applies temperature scaling to logits for model calibration.

    Temperature scaling is a post-processing technique for calibrating the confidence of model predictions.
    It rescales the logits by a learned temperature parameter to improve the reliability of predicted probabilities.

    Args:
        device: The device on which to store the temperature parameter.
        initial_temperature (float): Initial value for the temperature parameter.

    Methods:
        forward(logits): Applies temperature scaling to the input logits.

    """

    def __init__(self, device, initial_temperature: float = 0.9):
        """Initialize the TemperatureScaling module with a temperature parameter.

        Args:
            device: The device on which to store the temperature parameter.
            initial_temperature (float): Initial value for the temperature parameter.

        """
        super().__init__()
        self.T = nn.Parameter(torch.tensor(initial_temperature, device=device))

    def forward(self, logits):
        """Apply Beta calibration to the input logits.

        Args:
            logits (torch.Tensor): First component of the logits transformation.
                Shape: [batch_size, num_classes] for multi-class or [batch_size] for binary.

        Returns:
            torch.Tensor: Calibrated logits using the bivariate logistic regression model.
                Shape matches input logits.

        """
        return logits / self.T


class PlattScaling(nn.Module):
    """Applies Platt scaling to logits for model calibration.

    Platt scaling is a post-processing technique for calibrating the confidence of model predictions.
    It applies a class-specific linear transformation to the logits to improve the reliability of predicted probabilities.

    Args:
        n_classes (int): Number of classes.
        device: The device on which to store the parameters.

    Methods:
        forward(s_prime, s_double_prime): Applies Platt scaling to the input logits.

    """

    def __init__(self, n_classes, device):
        """Initialize Platt scaling parameters for model calibration.

        Args:
            n_classes (int): Number of classes in the classification problem.
            device: The device (CPU/GPU) on which to store and compute the parameters.

        The initialization creates two learnable parameters for each class:
        - a_raw: Raw slope parameter (will be passed through ReLU to ensure positive)
        - b: Intercept parameter

        """
        super().__init__()
        # Parameters for Platt Scaling: a (slope) and b (intercept) for each class
        # Initialize slopes to 1.0 + small noise
        self.a_raw = nn.Parameter(
            torch.ones(n_classes, device=device)
            + 0.01 * torch.randn(n_classes, device=device)
            if n_classes > 2
            else torch.tensor(1.0, device=device)
        )
        # Initialize intercepts to 0.0
        self.b = nn.Parameter(
            torch.zeros(n_classes, device=device)
            if n_classes > 2
            else torch.tensor(0.0, device=device)
        )

    @property
    def a(self):
        """Apply ReLU to ensure a >= 0 for all classes."""
        return F.relu(self.a_raw)

    def forward(self, s_prime, s_double_prime):
        """Apply Platt scaling to each class.

        Args:
            s_prime: Tensor of shape [batch_size, num_classes]
            s_double_prime: Tensor of shape [batch_size, num_classes]

        Returns:
            Calibrated logits of shape [batch_size, num_classes]

        """
        logits = s_prime + s_double_prime
        # Apply class-specific scaling parameters
        calibrated_logits = self.a * logits + self.b
        return calibrated_logits


class BetaCalibration(nn.Module):
    """Applies Beta calibration to logits for model calibration.

    Beta calibration extends Platt scaling by using three parameters (a, b, c) per class to calibrate model outputs. The calibration function is:
    β(p) = sigmoid(c + a*log(p) + b*log(1-p))

    Args:
        n_classes (int): Number of classes.
        device: The device on which to store the parameters.

    Methods:
        forward(s_prime, s_double_prime): Applies Beta calibration to the input logits.

    """

    def __init__(self, n_classes, device):
        """Initialize the BetaCalibration module with parameters for each class.

        Beta calibration extends Platt scaling by using three parameters (a, b, c) per class to calibrate the confidence scores. The parameters are learned through optimization.

        Args:
            n_classes (int): Number of classes in the classification problem.
            device: The device (CPU/GPU) on which to store and compute the parameters.

        """
        super().__init__()
        # Raw parameters (unconstrained)
        self.a_raw = nn.Parameter(
            torch.ones(n_classes, device=device)
            + 0.01 * torch.randn(n_classes, device=device)
            if n_classes > 2
            else torch.tensor(1.0, device=device)
        )
        self.b_raw = nn.Parameter(
            torch.ones(n_classes, device=device)
            + 0.01 * torch.randn(n_classes, device=device)
            if n_classes > 2
            else torch.tensor(1.0, device=device)
        )
        self.c = nn.Parameter(
            torch.zeros(n_classes, device=device)
            if n_classes > 2
            else torch.tensor(0.0, device=device)
        )

    @property
    def a(self):
        """Apply ReLU to ensure a >= 0."""
        return F.relu(self.a_raw)

    @property
    def b(self):
        """Apply ReLU to ensure b >= 0."""
        return F.relu(self.b_raw)

    def forward(self, s_prime, s_double_prime):
        """Apply Beta calibration to the input logits.

        Args:
            s_prime (torch.Tensor): First component of the logits transformation
            s_double_prime (torch.Tensor): Second component of the logits transformation

        Returns:
            torch.Tensor: Calibrated logits using the bivariate logistic regression model

        """
        # Bivariate logistic regression model: c + a * s' + b * s''
        calibrated_logits = self.c + self.a * s_prime + self.b * s_double_prime
        return calibrated_logits


class DirichletCalibration(nn.Module):
    """Applies Dirichlet calibration to logits for model calibration.

    Dirichlet calibration learns a transformation matrix W and bias vector b that maps from the log probabilities (log p) of the model's predictions to calibrated logits:
    calibrated_logits = W * log(p) + b

    Args:
        n_classes (int): Number of classes in the classification problem.
        device: The device on which to store and compute the parameters.

    Methods:
        forward(logits): Applies Dirichlet calibration to the input logits.

    """

    def __init__(self, n_classes, device):
        """Initialize the DirichletCalibration module.

        Dirichlet calibration learns a transformation matrix W and bias vector b that maps from the log probabilities of the model's uncalibrated predictions to calibrated logits:
        calibrated_logits = W * log(p) + b

        Args:
            n_classes: Number of classes in the classification problem
            device: The device (CPU/GPU) on which to store and compute parameters

        The parameters are initialized as:
            weights: Near-identity matrix + small random noise
            biases: Zero vector of length n_classes

        """
        super().__init__()
        self.n_classes = n_classes
        # Transformation parameters
        # Initialize to near-identity matrix + small noise
        # (The noise ensures small, random differences in the initial weights across classes. Now, gradients will
        # update each weight slightly differently, allowing the model to learn class-specific adjustments faster.)
        self.weights = nn.Parameter(
            torch.eye(n_classes, device=device)
            + 0.01 * torch.randn(n_classes, n_classes, device=device)
        )
        # Initialize to zero vector
        self.biases = nn.Parameter(torch.zeros(n_classes, device=device))

    def forward(self, logits):
        """Apply forward pass through the Dirichlet calibrator.

        Converts raw logits from a classifier into calibrated logits by first applying
        softmax to get probabilities, taking log, then applying a linear transformation.

        Args:
            logits (torch.Tensor): Raw logits from classifier model, shape (batch_size, num_classes)

        Returns:
            torch.Tensor: Calibrated logits after Dirichlet transformation, shape (batch_size, num_classes)

        """
        # convert classifier raw logits into log probabilities (processed logits)
        log_p = torch.log_softmax(logits, dim=1)
        # Apply the linear transformation on the processed logits: Dirichlet map (q; W, b) = (W * log(q) + b)
        calibrated_logits = torch.matmul(log_p, self.weights) + self.biases
        return calibrated_logits


class IsotonicRegressionCalibration:
    """Applies isotonic regression calibration to model predictions.

    Isotonic regression finds a non-decreasing function that best fits the relationship
    between predicted probabilities and true probabilities. This calibration method is
    non-parametric and can capture any monotonic miscalibration pattern.

    Attributes:
        device: The device on which to perform computations.

    Methods:
        _fit_isotonic_calibrator: Fits isotonic regression for a single class.
        _predict_isotonic_calibration: Applies calibration to new predictions.
        _isotonic_regression: Performs isotonic regression using Pool Adjacent Violators algorithm.

    """

    def __init__(self, device):
        """Initialize the utility class.

        Args:
            device: The device to be used for processing (e.g., 'cpu' or 'cuda').

        Returns:
            None

        """
        super().__init__()
        self.device = device

    def _fit_isotonic_calibrator(self, logits, binary_targets, sample_weight=None):
        """Fit isotonic regression for a single class.

        Args:
            logits (torch.Tensor): Model scores for the class
            binary_targets (torch.Tensor): Binary targets (1 for class, 0 otherwise)
            sample_weight (torch.Tensor, optional): Sample weights

        Returns:
            dict: Calibrator parameters

        """
        # Sort logits and correspondingly reorder targets and weights
        indices = torch.argsort(logits)
        logits_sorted = logits[indices]
        targets_sorted = binary_targets[indices]

        if sample_weight is not None:
            weights_sorted = sample_weight[indices]
        else:
            weights_sorted = torch.ones_like(logits_sorted, device=self.device)

        # Apply isotonic regression using PAV algorithm
        fitted_values = self._isotonic_regression(targets_sorted, weights_sorted)

        # Find unique logit thresholds to store
        # We want to find indices where logits change
        change_indices = torch.cat(
            [
                torch.tensor([0], dtype=torch.long, device=self.device),
                torch.where(logits_sorted[1:] != logits_sorted[:-1])[0] + 1,
                torch.tensor(
                    [len(logits_sorted)], dtype=torch.long, device=self.device
                ),
            ]
        )

        # Store unique thresholds and their corresponding calibrated values
        x_thresholds = logits_sorted[change_indices[:-1]]
        y_thresholds = fitted_values[change_indices[:-1]]

        return {"X_thresholds": x_thresholds, "y_thresholds": y_thresholds}

    def _predict_isotonic_calibration(self, calibrator, logits):
        """Apply calibration to new logits.

        Args:
            calibrator (dict): Calibrator parameters
            logits (torch.Tensor): New logits to calibrate

        Returns:
            torch.Tensor: Calibrated probabilities

        """
        X_thresholds = calibrator["X_thresholds"]
        y_thresholds = calibrator["y_thresholds"]

        # Prepare output array
        calibrated = torch.zeros_like(logits, device=self.device)

        # For values below the first threshold
        mask_below = logits < X_thresholds[0]
        calibrated[mask_below] = y_thresholds[0]

        # For values above the last threshold
        mask_above = logits >= X_thresholds[-1]
        calibrated[mask_above] = y_thresholds[-1]

        # For values in the range of thresholds
        mask_between = ~(mask_below | mask_above)
        logits_between = logits[mask_between]

        if len(logits_between) > 0:
            # Find the positions where logits_between would be inserted to maintain order
            indices = torch.searchsorted(X_thresholds, logits_between) - 1

            # Clip indices to valid range
            indices = torch.clamp(indices, 0, len(X_thresholds) - 2)

            # Perform linear interpolation
            x_low = X_thresholds[indices]
            x_high = X_thresholds[indices + 1]
            y_low = y_thresholds[indices]
            y_high = y_thresholds[indices + 1]

            # Handle cases where x_high == x_low to avoid division by zero
            same_x = x_high == x_low
            diff_x = ~same_x

            # For same x values, just use y_low
            calibrated_between = torch.zeros_like(logits_between)
            calibrated_between[same_x] = y_low[same_x]

            # For different x values, interpolate
            slope = torch.zeros_like(logits_between)
            slope[diff_x] = (y_high[diff_x] - y_low[diff_x]) / (
                x_high[diff_x] - x_low[diff_x]
            )
            calibrated_between[diff_x] = y_low[diff_x] + slope[diff_x] * (
                logits_between[diff_x] - x_low[diff_x]
            )

            # Assign interpolated values back to the full array
            calibrated[mask_between] = calibrated_between

        # Ensure probabilities are in [0, 1]
        calibrated = torch.clamp(calibrated, 0, 1)

        return calibrated

    def _isotonic_regression(self, y, sample_weight):
        """Perform isotonic regression using the Pool Adjacent Violators algorithm.

        Args:
            y (torch.Tensor): Target values
            sample_weight (torch.Tensor): Sample weights

        Returns:
            torch.Tensor: Fitted non-decreasing sequence

        """
        n_samples = len(y)
        result = torch.zeros_like(y, device=self.device)

        # Create initial solution blocks: (start_idx, end_idx, value, weight)
        solution_blocks = [(i, i, y[i], sample_weight[i]) for i in range(n_samples)]

        # Pool Adjacent Violators algorithm
        i = 0
        while i < len(solution_blocks) - 1:
            current_block = solution_blocks[i]
            next_block = solution_blocks[i + 1]

            # Check if monotonicity is violated
            if current_block[2] > next_block[2]:  # Current value > next value
                # Merge blocks
                start_idx = current_block[0]
                end_idx = next_block[1]

                # Calculate weighted average
                total_weight = current_block[3] + next_block[3]
                weighted_avg = (
                    current_block[2] * current_block[3] + next_block[2] * next_block[3]
                ) / total_weight

                # Create merged block
                merged_block = (start_idx, end_idx, weighted_avg, total_weight)

                # Replace the two blocks with the merged one
                solution_blocks[i] = merged_block
                solution_blocks.pop(i + 1)

                # Go back if possible to check if the merged block violates monotonicity with previous blocks
                if i > 0:
                    i -= 1
            else:
                # Move to the next block if no violation
                i += 1

        # Fill in the result array based on solution blocks
        for block in solution_blocks:
            start_idx, end_idx, value, _ = block
            result[start_idx : end_idx + 1] = value

        return result


class CalibratedModel:
    """A class for calibrating machine learning model predictions.

    This class implements various calibration methods to improve the reliability of model
    probability estimates. Supported methods include Temperature Scaling, Isotonic Regression,
    Platt Scaling, Beta Calibration, and Dirichlet Calibration.

    Attributes
    ----------
    VALID_METHODS : set
        Set of supported calibration methods
    method : str
        The chosen calibration method
    device : torch.device
        Device for computation (CPU/GPU)
    lr : float
        Learning rate for optimization
    n_epochs : int
        Maximum number of training epochs
    reg_lambda : float
        Regularization strength
    reg_mu : float
        Regularization strength for bias terms
    eps : float
        Small value for numerical stability
    initial_temperature : float
        Initial temperature for scaling
    verbose : bool
        Whether to print detailed information
    calibrated : bool
        Whether the model has been calibrated
    model : torch.nn.Module
        The calibration model

    """

    # VALID_METHODS = {"temperature", "isotonic", "platt", "beta", "dirichlet"}
    VALID_METHODS: ClassVar[list[str]] = [
        "temperature",
        "isotonic",
        "platt",
        "beta",
        "dirichlet",
    ]
    model: (
        TemperatureScaling
        | PlattScaling
        | BetaCalibration
        | DirichletCalibration
        | None
    )

    iso_calibrators: list | None

    def __init__(
        self,
        base_model=None,
        method: Literal[
            "temperature", "isotonic", "platt", "beta", "dirichlet"
        ] = "temperature",
        device: torch.device = torch.device("cuda"),
        lr: float = 1e-3,
        weight_decay_adam_optimizer: float = 5e-4,
        n_epochs: int = 1000,
        reg_lambda: float = 0.01,
        reg_mu: float = 0.01,
        eps: float = 1e-8,
        initial_temperature: float = 0.9,
        verbose: bool = False,
        factor_learning_rate_scheduler: float = 0.1,
        patience_learning_rate_scheduler: int = 20,
        patience_early_stopping: int = 50,
        min_delta_early_stopping: float = 0.001,
        save_path: str | None = None,
        seed: int = 42,
    ):
        """Initialize the calibration model.

        Args:
            method (str): Calibration method. Options: "beta" (Beta Calibration)
                or "temperature" (Temperature Scaling).
            device (torch.device): Device for computation
            lr (float): Learning rate for optimization.
            weight_decay_adam_optimizer (float): Weight decay parameter for Adam optimizer.
            n_epochs (int): Maximum number of iterations for optimization.
            reg_lambda (float): Regularization strength for platt scaling, beta calibration or for off-diagonal elements
                in dirichlet calibration.
            reg_mu (float): Regularization strength for intercept (bias) terms in dirichlet calibration.
            eps (float): Minimum value when calculating logarithm of probabilities (p > eps).
            initial_temperature (float): Initial temperature value for temperature scaling.
            verbose (bool): Whether to print detailed information during training.
            factor_learning_rate_scheduler (float): Factor by which to reduce learning rate on plateau.
            patience_learning_rate_scheduler (int): Number of epochs with no improvement before reducing learning rate.
            patience_early_stopping (int): Number of epochs with no improvement before early stopping.
            min_delta_early_stopping (float): Minimum change in monitored quantity to qualify as an improvement.
            save_path (str, optional): Directory path to save training plots and model parameters.

        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Method must be one of {self.VALID_METHODS}")

        self.base_model = base_model  # Original model that outputs logits
        self.method = method
        self.device = device
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay_adam_optimizer = weight_decay_adam_optimizer
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.eps = eps
        self.initial_temperature = initial_temperature
        self.verbose = verbose
        self.calibrated = False
        self.use_binary_calibration = None
        self.classes = None
        self.n_classes = None
        self.model = None
        self.iso_calibrators = None
        self.factor_learning_rate_scheduler = factor_learning_rate_scheduler
        self.patience_learning_rate_scheduler = patience_learning_rate_scheduler
        self.patience_early_stopping = patience_early_stopping
        self.min_delta_early_stopping = min_delta_early_stopping
        self.save_path = save_path
        self.seed = seed

    def fit(
        self,
        logits_train,
        labels_train,
        sample_weights_train=None,
    ):
        """Fit the calibration model to the training data.

        Args:
            logits_train (torch.Tensor): Logits from the model (shape: [n_samples] for binary,
                [n_samples, n_classes] for multi-class).
            labels_train (torch.Tensor): Ground truth labels (shape: [n_samples]).
            sample_weights_train (torch.Tensor): Weights of each sample (shape: [n_samples]).

        """
        if sample_weights_train is None:
            sample_weights_train = torch.ones_like(
                labels_train, dtype=torch.float32, device=self.device
            )

        self.classes = torch.unique(labels_train)
        self.n_classes = len(self.classes)

        single_logit = logits_train.dim() == 1 or (
            logits_train.dim() == 2 and logits_train.shape[1] == 1
        )
        # Check if binary or multi-class
        if self.n_classes <= 2 and single_logit:
            self.use_binary_calibration = True

            if self.method == "dirichlet":
                raise ValueError(
                    f"{single_logit * 1} logit does not match minimum number of 2 logits for Dirichlet calibration!"
                )
            # Binary classification
            logits_train = logits_train.reshape(labels_train.shape)
            self._fit_binary(logits_train, labels_train, sample_weights_train)

        else:
            self.use_binary_calibration = False

            # If necessary, expand logits to match class count
            if logits_train.shape[1] != self.n_classes:
                raise ValueError(
                    f"Number of logits columns ({logits_train.shape[1]}) does not match "
                    f"number of classes ({self.n_classes})"
                )
            # Multi-class classification (including binary classification with
            # two explicit logits value for each class)
            self._fit_multi_class(logits_train, labels_train, sample_weights_train)

    def _fit_binary(self, logits_train, labels_train, sample_weights_train):
        """Fit calibration model for binary classification."""
        labels_train = labels_train.float()

        if self.method != "isotonic":
            if self.method == "temperature":
                self.model = TemperatureScaling(
                    device=self.device, initial_temperature=self.initial_temperature
                )
            elif self.method == "platt":
                self.model = PlattScaling(n_classes=2, device=self.device)
            elif self.method == "beta":
                self.model = BetaCalibration(n_classes=2, device=self.device)

            # split the calibration data into training and validation
            labels_cpu = (
                labels_train.cpu().numpy()
                if labels_train.is_cuda
                else labels_train.numpy()
            )
            # Stratified split
            indices = torch.arange(len(logits_train))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.25, stratify=labels_cpu, random_state=self.seed
            )
            # Apply the split (first val then train as train is updated inplace)
            logits_val = logits_train[val_idx]
            labels_val = labels_train[val_idx]
            sample_weights_val = sample_weights_train[val_idx]
            logits_train = logits_train[train_idx]
            labels_train = labels_train[train_idx]

            # Initialize with default values
            s_prime = torch.empty(0, device=self.device)  # Initialize with empty tensor
            s_double_prime = torch.empty(0, device=self.device)
            val_s_prime = torch.empty(0, device=self.device)
            val_s_double_prime = torch.empty(0, device=self.device)

            if self.method in ["platt", "beta"]:
                # Calculate s_prime and s_double_prime locally
                probs_train = torch.sigmoid(logits_train).clamp(self.eps, 1 - self.eps)
                s_prime = torch.log(probs_train)
                s_double_prime = -torch.log(1 - probs_train)
                probs_val = torch.sigmoid(logits_val).clamp(self.eps, 1 - self.eps)
                val_s_prime = torch.log(probs_val)
                val_s_double_prime = -torch.log(1 - probs_val)

            pos_weight = labels_train.size(0) / (2 * labels_train.sum())

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay_adam_optimizer,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.factor_learning_rate_scheduler,
                patience=self.patience_learning_rate_scheduler,
            )

            last_lr = self.lr
            early_stopping = EarlyStopping(
                patience=self.patience_early_stopping,
                min_delta=self.min_delta_early_stopping,
            )

            track_losses = []
            track_gradients = []
            for epoch in range(self.n_epochs):
                optimizer.zero_grad()
                if self.method == "temperature":
                    calibrated_logits = self.model(logits_train)
                else:
                    assert len(s_prime) > 0 and len(s_double_prime) > 0, (
                        "Variables not initialized for this method"
                    )
                    calibrated_logits = self.model(s_prime, s_double_prime)
                # For Temperature Scaling, use logits directly
                loss = criterion(calibrated_logits, labels_train)
                # Add L2 regularization for smoothness
                loss += self.reg_lambda * sum([p**2 for p in self.model.parameters()])
                loss.backward()
                optimizer.step()
                track_losses.append(loss.item())
                track_gradients.append(self._calculate_parameter_gradients(self.model))
                # Evaluation on validation set
                with torch.no_grad():
                    calibrated_logits_val = (
                        self.model(logits_val)
                        if self.method == "temperature"
                        else self.model(val_s_prime, val_s_double_prime)
                    )
                    calibrated_probs_val = torch.sigmoid(calibrated_logits_val)
                    # Compute ECE on validation set
                    ece_val = self._compute_weighted_ece(
                        labels_val, calibrated_probs_val, sample_weights_val
                    )
                # Update learning rate
                scheduler.step(ece_val)
                # Get current learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                # Print message if learning rate has changed
                if current_lr != last_lr:
                    print(
                        f"Learning rate decreased from {last_lr:.5e} to {current_lr:.5e}"
                    )
                    last_lr = current_lr
                    early_stopping.reset_counter()
                # Check for early stopping
                early_stopping(ece_val)
                if early_stopping.early_stop:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!\n")
                    break
            if self.verbose:
                # print optimum parameters based on model type
                self._plot_loss_and_gradients(self.model, track_losses, track_gradients)
        else:
            model = IsotonicRegressionCalibration(device=self.device)
            if self.classes is None:
                raise RuntimeError("Classes were not properly found during fitting!")
            self.iso_calibrators = [
                model._fit_isotonic_calibrator(
                    logits_train,
                    (labels_train == self.classes[1]).float(),
                    sample_weights_train,
                )
            ]

        self.calibrated = True

    def _fit_multi_class(self, logits_train, labels_train, sample_weights_train):
        """Fit calibration model for multi-class classification."""
        if self.method != "isotonic":
            if self.method == "temperature":
                self.model = TemperatureScaling(
                    device=self.device, initial_temperature=self.initial_temperature
                )
            elif self.method == "platt":
                self.model = PlattScaling(n_classes=self.n_classes, device=self.device)
            elif self.method == "beta":
                self.model = BetaCalibration(
                    n_classes=self.n_classes, device=self.device
                )
            elif self.method == "dirichlet":
                self.model = DirichletCalibration(
                    n_classes=self.n_classes, device=self.device
                )
            # split the calibration data into training and validation
            labels_cpu = (
                labels_train.cpu().numpy()
                if labels_train.is_cuda
                else labels_train.numpy()
            )
            # Stratified split
            indices = torch.arange(len(logits_train))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.25, stratify=labels_cpu, random_state=self.seed
            )
            # Apply the split (first val then train as train is updated inplace)
            logits_val = logits_train[val_idx]
            labels_val = labels_train[val_idx]
            sample_weights_val = sample_weights_train[val_idx]
            logits_train = logits_train[train_idx]
            labels_train = labels_train[train_idx]

            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay_adam_optimizer,
            )
            # Initialize with default values
            s_prime = torch.empty(0, device=self.device)  # Initialize with empty tensor
            s_double_prime = torch.empty(0, device=self.device)
            val_s_prime = torch.empty(0, device=self.device)
            val_s_double_prime = torch.empty(0, device=self.device)
            if self.n_classes is None:
                raise RuntimeError(
                    "Number of classes were not properly found during fitting!"
                )
            labels_train_one_hot = torch.empty(0, device=self.device)

            if self.method in ["temperature", "dirichlet"]:
                class_weights_train = torch.tensor(
                    compute_class_weight(
                        class_weight="balanced",
                        classes=labels_train.unique().cpu().numpy(),
                        y=labels_train.cpu().numpy(),
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                criterion = nn.CrossEntropyLoss(weight=class_weights_train)
            else:
                # Create one-hot encoded labels
                labels_train_one_hot = F.one_hot(
                    labels_train, num_classes=self.n_classes
                ).float()
                # Calculate class weights for balanced loss
                class_counts = labels_train_one_hot.sum(dim=0)
                pos_weights = labels_train.size(0) / (2 * class_counts)
                # Use multi-label binary cross entropy (effectively one-vs-rest for each class)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                # Forward pass
                probs_train = torch.softmax(logits_train, dim=1).clamp(
                    self.eps, 1 - self.eps
                )
                s_prime = torch.log(probs_train)
                s_double_prime = -torch.log(1 - probs_train)
                probs_val = torch.softmax(logits_val, dim=1).clamp(
                    self.eps, 1 - self.eps
                )
                val_s_prime = torch.log(probs_val)
                val_s_double_prime = -torch.log(1 - probs_val)

            last_lr = self.lr
            # ReduceLROnPlateau reduces the learning rate when a metric has stopped improving
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.factor_learning_rate_scheduler,
                patience=self.patience_learning_rate_scheduler,
            )
            early_stopping = EarlyStopping(
                patience=self.patience_early_stopping,
                min_delta=self.min_delta_early_stopping,
            )

            track_losses = []
            track_gradients = []
            for epoch in range(self.n_epochs):
                optimizer.zero_grad()
                if self.method in ["temperature", "dirichlet"]:
                    calibrated_logits = self.model(logits_train)
                else:
                    assert len(s_prime) > 0 and len(s_double_prime) > 0, (
                        "Variables not initialized for this method"
                    )
                    calibrated_logits = self.model(s_prime, s_double_prime)
                loss = (
                    criterion(calibrated_logits, labels_train)
                    if self.method in ["temperature", "dirichlet"]
                    else criterion(calibrated_logits, labels_train_one_hot)
                )
                if self.method != "dirichlet":
                    # Add L2 regularization for smoothness
                    loss += self.reg_lambda * sum(
                        [torch.norm(p) ** 2 for p in self.model.parameters()]
                    )
                else:
                    # Add ODIR (Off-Diagonal and Intercept Regularization)
                    # Intuition Behind ODIR:
                    # Diagonal Elements: The diagonal elements of weights are not regularized because they capture
                    # class-specific biases (e.g., some classes may be inherently more confident than others).
                    # Off-Diagonal Elements: The off-diagonal elements are regularized because they represent interactions
                    # between classes, which are more likely to overfit to noise in the data.
                    # Intercept Terms (Biases): The intercept terms are regularized separately because they operate on an
                    # additive scale, unlike the multiplicative scale of the transformation matrix.
                    off_diag_elemnts = self.model.weights.triu(
                        diagonal=1
                    ) + self.model.weights.tril(diagonal=-1)
                    off_diag_penalty = (
                        self.reg_lambda
                        * off_diag_elemnts.pow(2).sum()
                        / (self.n_classes * (self.n_classes - 1))
                    )  # The term 1/(k*(k-1)) normalizes the penalty by the number of off-diagonal elements.
                    # The term 1/k normalizes the penalty by the number of classes (mean over classes).
                    intercept_penalty = self.reg_mu * self.model.biases.pow(2).mean()
                    loss += off_diag_penalty + intercept_penalty

                loss.backward(retain_graph=True)
                optimizer.step()
                track_losses.append(loss.item())
                track_gradients.append(self._calculate_parameter_gradients(self.model))

                # Evaluation on validation set
                with torch.no_grad():
                    calibrated_logits_val = (
                        self.model(logits_val)
                        if self.method in ["temperature", "dirichlet"]
                        else self.model(val_s_prime, val_s_double_prime)
                    )
                    calibrated_probs_val = torch.softmax(calibrated_logits_val, dim=1)
                    # Compute ECE on validation set
                    ece_val = self._compute_weighted_ece(
                        labels_val, calibrated_probs_val, sample_weights_val
                    )
                # Update learning rate
                scheduler.step(ece_val)
                # Get current learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                # Print message if learning rate has changed
                if current_lr != last_lr:
                    print(
                        f"Learning rate decreased from {last_lr:.5e} to {current_lr:.5e}"
                    )
                    last_lr = current_lr
                    early_stopping.reset_counter()
                # Check for early stopping
                early_stopping(ece_val)
                if early_stopping.early_stop:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!\n")
                    break
            if self.verbose:
                # print optimum parameters based on model type
                self._plot_loss_and_gradients(self.model, track_losses, track_gradients)
        else:
            model = IsotonicRegressionCalibration(device=self.device)
            if self.classes is None:
                raise RuntimeError("Classes were not properly found during fitting!")
            # Fit a separate calibrator for each class
            self.iso_calibrators = []
            for i, cls in enumerate(self.classes):
                class_logits = logits_train[:, i]
                class_labels = (labels_train == cls).float()
                calibrator = model._fit_isotonic_calibrator(
                    class_logits, class_labels, sample_weights_train
                )
                self.iso_calibrators.append(calibrator)

        self.calibrated = True

    def _calculate_parameter_gradients(self, model):
        total_norm = 0.0
        # Iterate through all parameters in the model
        for p in model.parameters():
            # Check if parameter has gradients (some layers might not)
            if p.grad is not None:
                # Calculate L2 norm (Euclidean norm) of gradients for this parameter
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def _plot_loss_and_gradients(self, model, track_losses, track_gradients):
        # print optimum parameters based on model type
        if self.save_path is not None:
            optimum_params_path = os.path.join(self.save_path, "optimum_params.txt")
        if self.method == "temperature":
            # Save the optimum parameters to a file
            with open(optimum_params_path, "w") as f:
                f.write(f"Temperature scaling optimum T: {model.T.item():.4f}")

        elif self.method == "dirichlet":
            # Save the optimum parameters to a file
            with open(optimum_params_path, "w") as f:
                f.write(
                    f"Dirichlet calibration optimum parameters:\nweights \n{model.weights.detach().cpu().numpy()}"
                    f"\nbiases:\n {model.biases.detach().cpu().numpy()}"
                )
        elif self.method == "platt":
            # Save the optimum parameters to a file
            with open(optimum_params_path, "w") as f:
                f.write(
                    f"Platt scaling optimum parameters:\na- {model.a.detach().cpu().numpy()}, b- {model.b.detach().cpu().numpy()}"
                )
        else:
            # Save the optimum parameters to a file
            with open(optimum_params_path, "w") as f:
                f.write(
                    f"Beta calibration optimum parameters:"
                    f"a- {model.a.detach().cpu().numpy()}, b- {model.b.detach().cpu().numpy()}, "
                    f"c- {model.c.detach().cpu().numpy()}"
                )
        # Plot loss and gradients
        fig, axs = pyplot.subplots(1, 2, figsize=(20, 6))
        axs[0].plot(range(len(track_losses)), track_losses)
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss")
        axs[0].set_yscale("log")
        axs[1].plot(range(len(track_gradients)), track_gradients)
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Gradient Norm")
        axs[1].set_yscale("log")
        fig.suptitle(f"Calibration Method: {self.method} - Loss and Gradient Norms")
        if self.save_path is not None:
            pyplot.savefig(
                f"{self.save_path}/calibration_{self.method}_loss_and_gradients.png"
            )

    def predict(self, X):
        """Predict calibrated probabilities for test logits.

        Args:
            X (torch.Tensor): New raw data
            (shape: [n_samples] for binary, [n_samples, n_classes] for multi-class).

        Returns:
            torch.Tensor: Calibrated probabilities
            (shape: [n_samples] for binary, [n_samples, n_classes] for multi-class).

        """
        if not self.calibrated:
            raise RuntimeError("Call fit() first")
        if not hasattr(self, "base_model"):
            raise ValueError("base_model required for raw data prediction")

        with torch.no_grad():
            # Step 1: Get uncalibrated logits
            logits = self.base_model(X.to(self.device))
            # Step 2: AApply calibration and convert to probabilities
            return self._apply_calibration(logits)

    def evaluate_calibration(
        self, logits_test, labels_test, sample_weights_test=None, n_bins=10
    ):
        """Calculate the Expected Calibration Error (ECE) for uncalibrated and calibrated predictions.

        Args:
            logits_test (torch.Tensor): Model logits/scores before calibration.
            labels_test (torch.Tensor): True class labels for test samples.
            sample_weights_test (torch.Tensor, optional): Sample weights for test data. Defaults to None.
            n_bins (int, optional): Number of bins for ECE calculation. Defaults to 10.

        Returns:
            dict: Dictionary containing:
                - 'ece_before': Expected Calibration Error before calibration
                - 'ece_after': Expected Calibration Error after calibration

        """
        if not self.use_binary_calibration:
            probs_uncalibrated = torch.softmax(torch.tensor(logits_test).float(), dim=1)
        else:
            probs_uncalibrated = torch.sigmoid(torch.tensor(logits_test).float())

        ece_before = self._compute_weighted_ece(
            labels_test, probs_uncalibrated, sample_weights_test, n_bins
        )

        probs_calibrated = self.predict(logits_test)
        ece_after = self._compute_weighted_ece(
            labels_test, probs_calibrated, sample_weights_test, n_bins
        )

        return {"ece_before": ece_before, "ece_after": ece_after}

    def _compute_weighted_ece(self, y_true, y_prob, sample_weights=None, n_bins=10):
        if not self.use_binary_calibration:
            confidence, predictions = torch.max(y_prob, dim=1)
        else:
            confidence = y_prob[:, 1] if y_prob.dim() == 2 else y_prob
            predictions = (confidence >= 0.5).long()
        accuracy = (predictions == y_true).float()

        # Create bins and calculate weighted ECE
        bin_indices = torch.floor(confidence * n_bins).long()
        bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)

        ece = 0.0
        total_weight = sample_weights.sum()

        for b in range(n_bins):
            mask = bin_indices == b
            if torch.any(mask):
                sample_weights = sample_weights
                bin_weights = sample_weights[mask]
                bin_total_weight = bin_weights.sum()

                bin_confidence = (
                    confidence[mask] * bin_weights
                ).sum() / bin_total_weight
                bin_accuracy = (accuracy[mask] * bin_weights).sum() / bin_total_weight
                bin_weight = bin_total_weight / total_weight

                ece += bin_weight * torch.abs(bin_confidence - bin_accuracy)

        return ece.item()

    def _apply_calibration(self, logits):
        """Core calibration logic for all methods."""
        # n_samples = logits.shape[0]

        # Binary case handling
        if self.use_binary_calibration:
            if self.method == "isotonic":
                pos_probs = self._isotonic_predict_binary(logits)
                return torch.stack([1 - pos_probs, pos_probs], dim=1)
            else:
                calibrated = self._parametric_calibrate_binary(logits)
                return (
                    torch.sigmoid(calibrated)
                    if self.method == "temperature"
                    else torch.sigmoid(calibrated).unsqueeze(1)
                )

        # Multi-class case
        if self.method == "isotonic":
            return self._isotonic_predict_multiclass(logits)
        else:
            calibrated = self._parametric_calibrate_multiclass(logits)
            return torch.softmax(calibrated, dim=1)

    # Helper methods
    def _parametric_calibrate_binary(self, logits):
        """Handle Platt/Beta/Temperature scaling for binary."""
        if self.method == "temperature":
            return self.model(logits)

        probs = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        s_prime = torch.log(probs)
        s_double_prime = -torch.log(1 - probs)
        return self.model(s_prime, s_double_prime)

    def _parametric_calibrate_multiclass(self, logits):
        """Handle Platt/Beta/Temperature/Dirichlet for multi-class."""
        if self.method in ["temperature", "dirichlet"]:
            return self.model(logits)

        probs = torch.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)
        s_prime = torch.log(probs)
        s_double_prime = -torch.log(1 - probs)
        return self.model(s_prime, s_double_prime)

    def _isotonic_predict_binary(self, logits):
        """Isotonic binary prediction."""
        if self.iso_calibrators is None:
            raise RuntimeError(
                "Calibration model was not properly initialized during fitting"
            )
        return IsotonicRegressionCalibration(
            device=self.device
        )._predict_isotonic_calibration(self.iso_calibrators[0], logits)

    def _isotonic_predict_multiclass(self, logits):
        if self.iso_calibrators is None:
            raise RuntimeError(
                "Calibration model was not properly initialized during fitting"
            )
        """Isotonic multi-class prediction"""
        probs = torch.zeros(logits.shape, device=self.device)
        for i, _ in enumerate(self.iso_calibrators):
            probs[:, i] = self._isotonic_predict_binary(logits[:, i])
        return probs / probs.sum(dim=1, keepdim=True).clamp(min=self.eps)

    def save_calibrated_model(self, filepath: str) -> None:
        """Save the complete calibrated model state for pipeline deployment.

        Args:
            filepath: Path to save the calibrated model

        """
        if not self.calibrated:
            raise RuntimeError("Model must be calibrated before saving")

        # Create comprehensive state dictionary
        state = {
            "method": self.method,
            "use_binary_calibration": self.use_binary_calibration,
            "classes": self.classes,
            "n_classes": self.n_classes,
            "eps": self.eps,
            "calibrated": self.calibrated,
            "model_state": None,
            "iso_calibrators": None,
        }

        # Save parametric model state
        if self.model is not None:
            state["model_state"] = {
                "state_dict": self.model.state_dict(),
                "model_class": self.model.__class__.__name__,
            }

        # Save isotonic calibrators
        if self.iso_calibrators is not None:
            state["iso_calibrators"] = self.iso_calibrators

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save with both pickle and torch formats for robustness
        # torch.save(state, filepath)
        torch.save(state, os.path.join(filepath, "calibrator.pt"))

        # # Also save a human-readable summary
        # summary_path = filepath.replace(".pkl", "_summary.txt")
        # self._save_calibration_summary(summary_path, state)

        print(f"Calibrated model saved to: {filepath}")

    @classmethod
    def load_calibrated_model(
        cls, filepath: str, device: torch.device | None = None
    ) -> "CalibratedModel":
        """Load a previously saved calibrated model for inference.

        Args:
            filepath: Path to the saved calibrated model
            device: Target device for the model

        Returns:
            CalibratedModel: Loaded calibrated model ready for inference

        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibrated model file not found: {filepath}")

        # Load state (weights_only=False since we're saving custom objects)
        state = torch.load(filepath, weights_only=False, map_location="cpu")

        # Create new instance with minimal initialization
        instance = cls.__new__(cls)
        instance.device = device or torch.device("cpu")
        instance.method = state["method"]
        instance.use_binary_calibration = state["use_binary_calibration"]
        instance.classes = state["classes"]
        instance.n_classes = state["n_classes"]
        instance.eps = state["eps"]
        instance.calibrated = state["calibrated"]
        instance.base_model = None  # Will be set externally if needed

        # Restore parametric model
        if state["model_state"] is not None:
            model_class_name = state["model_state"]["model_class"]

            # Reconstruct the appropriate model
            if model_class_name == "TemperatureScaling":
                instance.model = TemperatureScaling(device=instance.device)
            elif model_class_name == "PlattScaling":
                instance.model = PlattScaling(
                    n_classes=instance.n_classes, device=instance.device
                )
            elif model_class_name == "BetaCalibration":
                instance.model = BetaCalibration(
                    n_classes=instance.n_classes, device=instance.device
                )
            elif model_class_name == "DirichletCalibration":
                instance.model = DirichletCalibration(
                    n_classes=instance.n_classes, device=instance.device
                )
            else:
                raise ValueError(f"Unknown model class: {model_class_name}")

            # Load the state dictionary
            instance.model.load_state_dict(state["model_state"]["state_dict"])
            instance.model.eval()  # Set to evaluation mode
        else:
            instance.model = None

        # Restore isotonic calibrators
        instance.iso_calibrators = state.get("iso_calibrators")

        print(f"Calibrated model loaded from: {filepath}")
        return instance

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict calibrated probabilities from logits (primary inference method).

        Args:
            logits: Model logits tensor

        Returns:
            Calibrated probabilities

        """
        if not self.calibrated:
            raise RuntimeError("Model must be calibrated before prediction")

        with torch.no_grad():
            return self._apply_calibration(logits.to(self.device))

    def predict_with_base_model(self, X: Data) -> torch.Tensor:
        """End-to-end prediction with base model + calibration.

        Args:
            X: Raw input data

        Returns:
            Calibrated probabilities

        """
        if self.base_model is None:
            raise ValueError("base_model must be set for end-to-end prediction")

        with torch.no_grad():
            logits = self.base_model(X.to(str(self.device)))
            return self._apply_calibration(logits)

    def set_base_model(self, base_model: nn.Module) -> None:
        """Set the base model for end-to-end prediction.

        Args:
            base_model: The trained base model that outputs logits

        """
        self.base_model = base_model
        if hasattr(base_model, "eval"):
            base_model.eval()

    def _save_calibration_summary(self, filepath: str, state: dict[str, Any]) -> None:
        """Save human-readable calibration summary."""
        with open(filepath, "w") as f:
            f.write("CALIBRATION MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Method: {state['method']}\n")
            f.write(f"Number of classes: {state['n_classes']}\n")
            f.write(f"Binary calibration: {state['use_binary_calibration']}\n")
            f.write(
                f"Classes: {state['classes'].tolist() if state['classes'] is not None else None}\n"
            )

            if state["model_state"] is not None:
                f.write(f"\nParametric Model: {state['model_state']['model_class']}\n")
                if hasattr(self.model, "T"):
                    f.write(f"Temperature: {self.model.T.item():.4f}\n")

            if state["iso_calibrators"] is not None:
                f.write(
                    f"\nIsotonic calibrators: {len(state['iso_calibrators'])} classes\n"
                )

    def get_calibration_info(self) -> dict[str, Any]:
        """Get information about the current calibration state.

        Returns:
            Dictionary with calibration information

        """
        info = {
            "calibrated": self.calibrated,
            "method": self.method,
            "n_classes": self.n_classes,
            "use_binary_calibration": self.use_binary_calibration,
            "has_base_model": self.base_model is not None,
        }

        if self.calibrated and self.model is not None:
            if hasattr(self.model, "T"):
                info["temperature"] = self.model.T.item()
            elif hasattr(self.model, "a") and hasattr(self.model, "b"):
                info["parameters"] = {
                    "a": self.model.a.detach().cpu().numpy().tolist(),
                    "b": self.model.b.detach().cpu().numpy().tolist(),
                }

        return info


# Pipeline utility class for easier deployment
class CalibrationPipeline:
    """High-level pipeline class for model calibration and deployment."""

    def __init__(
        self, base_model: nn.Module, device: torch.device = torch.device("cpu")
    ):
        self.base_model = base_model
        self.device = device
        self.calibrated_model = None

    def calibrate(
        self,
        logits_train: torch.Tensor,
        labels_train: torch.Tensor,
        sample_weights_train: torch.Tensor | None,
        save_path: str | None = None,
        method: Literal[
            "temperature", "isotonic", "platt", "beta", "dirichlet"
        ] = "temperature",
        **calibration_kwargs,
    ) -> "CalibrationPipeline":
        """Calibrate the model using training data.

        Args:
            logits_train: Training logits
            labels_train: Training labels
            method: Calibration method
            **calibration_kwargs: Additional arguments for CalibratedModel

        Returns:
            Self for method chaining

        """
        self.calibrated_model = CalibratedModel(
            base_model=self.base_model,
            method=method,
            device=self.device,
            save_path=save_path,
            **calibration_kwargs,
        )

        self.calibrated_model.fit(logits_train, labels_train, sample_weights_train)
        return self

    def save(self, filepath: str) -> "CalibrationPipeline":
        """Save the calibrated pipeline."""
        if self.calibrated_model is None:
            raise RuntimeError("Pipeline must be calibrated before saving")

        self.calibrated_model.save_calibrated_model(filepath)
        return self

    @classmethod
    def load(
        cls,
        filepath: str,
        base_model: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> "CalibrationPipeline":
        """Load a calibrated pipeline."""
        pipeline = cls(base_model, device)
        pipeline.calibrated_model = CalibratedModel.load_calibrated_model(
            filepath, device
        )
        pipeline.calibrated_model.set_base_model(base_model)
        return pipeline

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make calibrated predictions."""
        if self.calibrated_model is None:
            raise RuntimeError(
                "Pipeline must be calibrated or loaded before prediction"
            )

        return self.calibrated_model.predict_with_base_model(X)

    def predict_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Make calibrated predictions from logits."""
        if self.calibrated_model is None:
            raise RuntimeError(
                "Pipeline must be calibrated or loaded before prediction"
            )

        return self.calibrated_model.predict_proba(logits)


def generate_mineral_data(
    n_samples: int = 500,
    spacing: float = 50,
    depth: float = -500.0,
    n_features: int = 10,
    n_classes: int = 5,
    threshold_binary: float = 0.3,
    min_samples_per_class: int = 15,
    seed: int = 42,
):
    """Generate synthetic mineral exploration data with realistic features.

    Parameters
    ----------
    n_samples : int
        Number of drill core samples to generate
    spacing : float
        Average distance between samples in meters
    depth : float
        Max ranges for depth coordinates (usually negative)
    n_features : int
        Number of features to generate
    n_classes : int
        Number of classes for gold concentration.
        Use 2 for binary classification (gold/no gold),
        or higher values for finer-grained concentration levels
    threshold_binary: float
        Threshold for binary classification
    min_samples_per_class : int
        Minimum number of samples required for each class
    seed : int
        Random seed for reproducibility

    Returns
    -------
    coordinates : ndarray
        Array of shape (n_samples, 3) containing x, y, z coordinates
    features : ndarray
        Scaled array of shape (n_samples, n_features) containing mineral features
    labels : ndarray
        Array of shape (n_samples,) containing gold concentration labels (0-4)

    """
    rng = np.random.default_rng(seed)  # Set random seed for reproducibility

    # Calculate the grid size based on n_samples and spacing
    grid_size = int(np.sqrt(n_samples)) + 1
    area_size = grid_size * spacing

    # Define x_range and y_range based on spacing and n_samples
    x_range = (0, area_size)
    y_range = (0, area_size)

    # 1. Generate spatial coordinates with appropriate spacing
    # First create a grid
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)

    # Create all possible grid points
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    # Add some random jitter to make it more realistic (not exactly on grid)
    jitter = spacing * rng.uniform(0.3, 0.6, 1)  # % of spacing for natural randomness
    grid_points[:, 0] += rng.uniform(-jitter, jitter, len(grid_points))
    grid_points[:, 1] += rng.uniform(-jitter, jitter, len(grid_points))

    # Select n_samples points from the grid
    indices = rng.choice(
        len(grid_points), min(n_samples, len(grid_points)), replace=False
    )
    xy_coordinates = grid_points[indices]

    # Generate z coordinates (depth)
    z_coordinates = rng.uniform(depth, 0, n_samples)

    # Combine to form complete coordinates
    coordinates = np.column_stack((xy_coordinates, z_coordinates))

    # 2. Create mineralization hotspots (mineralization centers)
    # In a mineral exploration context, hotspot strengths represent the maximum concentration or intensity of gold
    # at each "source" location in the simulated area. Not all gold deposits are created equal - some have higher
    # mineral content than others. Values greater than 1.0 represent "high-grade" hotspots that can potentially yield
    # gold values above the baseline (before applying distance decay). Values below 1.0 represent "lower-grade" hotspots
    # that will produce somewhat weaker signals. The range isn't centered at 1.0 (it's 0.7-1.2) to create a slight
    # positive skew, which is common in real mineral deposits
    n_hotspots = rng.integers(1, 5)
    hotspot_strengths = rng.uniform(0.7, 1.2, n_hotspots)
    hotspots = np.zeros((n_hotspots, 3))
    hotspots[:, 0] = rng.uniform(x_range[0], x_range[1], n_hotspots)
    hotspots[:, 1] = rng.uniform(y_range[0], y_range[1], n_hotspots)
    hotspots[:, 2] = rng.uniform(depth, 0, n_hotspots)

    # 3. Calculate gold values based on distance to nearest hotspot
    # Reshape for broadcasting: coordinates (n_samples, 3), hotspots (n_hotspots, 3)
    # Result: distances will be (n_samples, n_hotspots)
    distances = np.sqrt(
        np.sum(
            (coordinates[:, np.newaxis, :] - hotspots[np.newaxis, :, :]) ** 2,
            axis=2,
        )
    )

    # Find closest hotspot for each sample
    min_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    # Get the strength of each sample's closest hotspot
    closest_strengths = hotspot_strengths[min_indices]

    # Calculate gold values using exponential decay with distance
    noise_level = rng.uniform(0.01, 0.1)
    exp_decay_factor = rng.uniform(0.001, 0.01)
    gold_values = closest_strengths * np.exp(
        -min_distances * exp_decay_factor
    ) + rng.normal(0, noise_level, size=n_samples)

    # Clip values to 0-1 range
    gold_values = np.clip(gold_values, 0, 1)

    # 4. Convert to categorical labels
    if n_classes == 2:  # Binary classification: gold/no gold
        labels = (gold_values >= threshold_binary).astype(int)
    else:  # Multi-class classification
        # Create bins for digitizing
        bins = np.linspace(0, 1, n_classes, endpoint=False)[1:]  # n_classes-1 bin edges
        labels = np.digitize(gold_values, bins)  # 0 to n_classes-1

    # 5. Check if we have the minimum number of samples per class
    # Add samples for underrepresented classes if needed
    class_counts = [np.sum(labels == i) for i in range(n_classes)]

    for class_idx in range(n_classes):
        samples_needed = max(0, min_samples_per_class - class_counts[class_idx])

        if samples_needed > 0:
            print(f"\nAdding {samples_needed} more samples for class {class_idx}")

            # Parameters for each class based on distance from hotspots
            if n_classes == 2:  # Binary case: gold/no gold
                if class_idx == 0:  # No gold - far from hotspots
                    max_dist = 600
                    min_dist = 300
                else:  # Gold - close to hotspots
                    max_dist = 200
                    min_dist = 0
            else:  # Multi-class case
                # Calculate distance ranges based on number of classes
                # Class 0 is furthest from hotspots, highest class is closest
                class_range = 600 / n_classes
                max_dist = 600 - class_idx * class_range
                min_dist = max(0, max_dist - class_range)

            new_coordinates = []
            new_gold_values = []

            while len(new_coordinates) < samples_needed:
                # Pick a random hotspot
                hotspot_idx = rng.integers(0, n_hotspots)

                # Sample at appropriate distance
                angle = rng.uniform(0, 2 * np.pi)
                phi = rng.uniform(0, np.pi)
                distance = rng.uniform(min_dist, max_dist)

                # Convert to Cartesian coordinates
                dx = distance * np.sin(phi) * np.cos(angle)
                dy = distance * np.sin(phi) * np.sin(angle)
                dz = distance * np.cos(phi)

                new_x = hotspots[hotspot_idx, 0] + dx
                new_y = hotspots[hotspot_idx, 1] + dy
                new_z = hotspots[hotspot_idx, 2] + dz

                # Keep within bounds
                new_x = max(x_range[0], min(x_range[1], new_x))
                new_y = max(y_range[0], min(y_range[1], new_y))
                new_z = max(depth, min(0, new_z))

                # Calculate gold value
                pt = np.array([new_x, new_y, new_z])
                distances = np.sqrt(np.sum((pt - hotspots) ** 2, axis=1))
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                strength = hotspot_strengths[min_idx]

                gold_value = strength * np.exp(
                    -min_dist * exp_decay_factor
                ) + rng.normal(0, noise_level)
                gold_value = max(0, min(1, gold_value))

                # Check if it falls in the desired class
                if n_classes == 2:  # Binary case
                    new_label = int(gold_value >= threshold_binary)
                else:  # Multi-class case
                    bins = np.linspace(0, 1, n_classes, endpoint=False)[1:]
                    new_label = np.digitize([gold_value], bins)[0]

                if new_label == class_idx:
                    new_coordinates.append([new_x, new_y, new_z])
                    new_gold_values.append(gold_value)

            # Add new samples
            if new_coordinates:
                coordinates = np.vstack([coordinates, np.array(new_coordinates)])
                gold_values = np.append(gold_values, new_gold_values)

    # Recalculate labels for all samples
    if n_classes == 2:  # Binary classification: gold/no gold
        labels = (gold_values >= threshold_binary).astype(int)
    else:  # Multi-class classification
        bins = np.linspace(0, 1, n_classes, endpoint=False)[1:]
        labels = np.digitize(gold_values, bins)

    # 6. Generate features for all samples
    n_total_samples = len(coordinates)
    features = np.zeros((n_total_samples, n_features))

    # Define possible features based on priority/importance
    feature_generators = [
        # Pathfinder elements - strongly correlated with gold
        lambda gold_value: gold_value * 800 + rng.normal(0, 30),  # Arsenic
        lambda gold_value: gold_value * 400 + rng.normal(0, 15),  # Silver
        lambda gold_value: gold_value * 200 + rng.normal(0, 20),  # Copper
        # Somewhat correlated elements
        lambda gold_value: gold_value * 100 + rng.normal(0, 30),  # Lead
        lambda gold_value: gold_value * 50 + rng.normal(0, 20),  # Zinc
        # Geological features
        lambda gold_value: 60 + rng.normal(0, 10) - gold_value * 10,  # Silica
        lambda gold_value: gold_value * 8 + rng.normal(0, 1),  # Sulfides
        lambda gold_value: gold_value * 5 + rng.normal(0, 0.5),  # Alteration
        lambda gold_value: gold_value * 6 + rng.normal(0, 0.7),  # Vein density
        # Less correlated feature
        lambda gold_value: rng.normal(5, 1),  # Rock competency
        # Additional features if needed
        lambda gold_value: gold_value * 3 + rng.normal(0, 0.8),  # Bismuth
        lambda gold_value: gold_value * 50 + rng.normal(0, 15),  # Antimony
        lambda gold_value: rng.normal(3, 1.5),  # Iron
        lambda gold_value: gold_value * 10 + rng.normal(0, 2),  # Tellurium
        lambda gold_value: 20 - gold_value * 5 + rng.normal(0, 3),  # Carbonate
    ]

    # Feature names for reference
    feature_names = [
        "Arsenic",
        "Silver",
        "Copper",
        "Lead",
        "Zinc",
        "Silica",
        "Sulfides",
        "Alteration",
        "Vein_density",
        "Rock_competency",
        "Bismuth",
        "Antimony",
        "Iron",
        "Tellurium",
        "Carbonate",
    ]

    # Ensure we don't try to generate more features than we have generators for
    actual_n_features = min(n_features, len(feature_names))
    if actual_n_features < n_features:
        print(
            f"Warning: Requested {n_features} features but only {actual_n_features} are defined."
            f"Generating {actual_n_features} features."
        )

    # Create correlations between gold and features
    for i in range(n_total_samples):
        gold_value = gold_values[i]

        # Generate each feature based on the number requested
        for j in range(actual_n_features):
            features[i, j] = feature_generators[j](gold_value)

    # Ensure all features are positive (where it makes sense)
    features = np.maximum(features, 0)

    # Print summary
    print(f"Generated {n_total_samples} samples with {actual_n_features} features.")
    print(f"{n_hotspots} hotspots with respective strengths {hotspot_strengths}")
    print("\nLabel distribution:")
    for i in range(n_classes):
        count = np.sum(labels == i)
        print(f"Class {i}: {count} points ({count / n_total_samples * 100:.2f}%)")

    return coordinates, features, labels


def visualize_graph(
    coordinates,
    labels,
    src,
    dst,
    edge_opacity=0.3,
    label_map={0: "low", 1: "medium-low", 2: "medium", 3: "medium-high", 4: "high"},
    title="3D Geospatial Graph",
):
    """Create an interactive 3D visualization of the geospatial graph.

    Parameters
    ----------
        coordinates (array-like): Coordinate points for constructing the graph
        labels (array-like): Node labels for stratification
        src (array-like): Source nodes
        dst (array-like): Destination nodes
        edge_opacity (float): Opacity of edge between nodes
        title (str): Title for the visualization

    Returns
    -------
        None: Displays the interactive plot

    """
    # Create color map for different classes
    unique_classes = len(np.unique(labels))
    colors = pyplot.cm.rainbow(np.linspace(0, 1, unique_classes))
    color_map = {
        i: f"rgb({int(255 * c[0])},{int(255 * c[1])},{int(255 * c[2])})"
        for i, c in enumerate(colors)
    }

    # Create node trace
    node_trace = go.Scatter3d(
        x=coordinates[:, 0],  # Easting
        y=coordinates[:, 1],  # Northing
        z=coordinates[:, 2],  # Depth
        mode="markers",
        marker={
            "size": 5,
            "color": [color_map[label] for label in labels],
            "opacity": 0.8,
        },
        text=[f"Prospectivity: {label_map[label]}" for label in labels],
        hoverinfo="text",
        name="Nodes",
    )

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_z = []

    for s, d in zip(src, dst, strict=False):
        edge_x.extend([coordinates[s, 0], coordinates[d, 0], None])
        edge_y.extend([coordinates[s, 1], coordinates[d, 1], None])
        edge_z.extend([coordinates[s, 2], coordinates[d, 2], None])
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line={"color": "gray", "width": 1},
        opacity=edge_opacity,
        hoverinfo="none",
        name="Edges",
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update layout
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Easting",
            "yaxis_title": "Northing",
            "zaxis_title": "Depth",
            "aspectmode": "data",
        },
        showlegend=True,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )
    fig.update_layout(width=1000, height=600)
    # Add legend entries for classes
    for label, color in color_map.items():
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"size": 10, "color": color},
                name=f"{label_map[label]}",
                showlegend=True,
            )
        )
    return fig


ScalerType = StandardScaler | MinMaxScaler | RobustScaler


def scale_data(data: np.ndarray, scaler: ScalerType) -> np.ndarray:
    """Scale input data using the specified scaler.

    Parameters
    ----------
    data : np.ndarray
        Input data to be scaled
    scaler : ScalerType
        Scaler object (StandardScaler, MinMaxScaler, or RobustScaler)

    Returns
    -------
    np.ndarray
        Scaled data transformed by the fitted scaler

    """
    return scaler.fit_transform(data)


def construct_graph(
    coordinates,
    features,
    labels,
    connection_radius: float,
    n_splits: int,
    test_size: float,
    calib_size: int,
    seed: int,
):
    """Create graphs from geospatial data using distance matrix with a held-out test set.

    Args:
    coordinates (array-like): Coordinate points for constructing the graph
    features (array-like): Node features
    labels (array-like): Node labels for stratification
    n_splits (int): Number of folds for cross-validation
    test_size (float): Proportion of data to use as test set (e.g., 0.2 for 20%)
    calib_size (float): Proportion of data to use as calibration set
        (e.g., 0.5 for 50%)
    connection_radius (float): Distance threshold to consider interconnected nodes
    seed (int): Random seed for reproducibility

    Returns:
    tuple containing:
        - list of Data: PyG Data objects for each fold (train/val splits)
        - Data: Single PyG Data object for test (and an optional calibration) set

    """
    # Convert features and labels to torch tensors
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    edge_index, edge_attr = prepare_edge_data(coordinates, connection_radius)

    n_nodes = len(labels)
    # First split into train+val and test
    train_val_idx, temp_idx = train_test_split(
        np.arange(n_nodes),
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    # Initialize stratified k-fold on the train+val data
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Create a list to store Data objects for each fold
    fold_data = []

    # Generate folds from the train+val data
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(features[train_val_idx], labels[train_val_idx])
    ):
        # Map the fold indices back to original indices
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

        # Create boolean masks for this fold
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Create PyG Data object for this fold (train/val only)
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            train_mask=train_mask,
            val_mask=val_mask,
            fold=fold_idx,
        )

        fold_data.append(data)

    if calib_size is None:
        test_idx = temp_idx
        # Create separate test Data object
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask[test_idx] = True
        test_data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            test_mask=test_mask,
        )
    else:
        test_idx, calib_idx = train_test_split(
            temp_idx,
            train_size=calib_size,
            stratify=labels[temp_idx],
            random_state=seed,
        )
        # Create separate test Data object
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask[test_idx] = True
        # Create separate calibration Data object
        calib_mask = torch.zeros(n_nodes, dtype=torch.bool)
        calib_mask[calib_idx] = True
        test_data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            test_mask=test_mask,
            calib_mask=calib_mask,
        )
    return fold_data, test_data


def prepare_edge_data(coordinates, connection_radius: float = 150):
    """Prepare edge connectivity and attributes for a graph neural network from coordinate data. This function computes pairwise distances between points and creates edge connections based on a distance threshold, making the resulting graph undirected. It also generates edge attributes including both raw distances and inverse distances.

    Args:
        coordinates (numpy.ndarray): Array of point coordinates with shape [num_nodes, num_dimensions]
        connection_radius (float): Distance to consider interconnected nodes and create edges. Defaults to 150.0.

    Returns:
        tuple: Contains:
            - edge_index (torch.Tensor): Tensor of shape [2, num_edges] containing source and
              destination node indices for each edge
            - edge_attr (torch.Tensor): Tensor of shape [num_edges, 2] containing edge attributes
              [inverse_distance, raw_distance] for each edge
    Notes:
        - Self-loops are explicitly excluded (nodes cannot connect to themselves)
        - The graph is made undirected by adding reciprocal edges
        - Edge attributes could include both inverse squared distance (1/d) and raw distance (d)

    """
    # Compute pairwise Euclidean distances
    dist_matrix = distance_matrix(coordinates, coordinates)
    # Find edges based on distance threshold avoiding self-node thru distance > 0
    # to force the model to learn purely from neighboring nodes
    src, dst = np.where((dist_matrix < connection_radius) & (dist_matrix > 0))
    # # or alternatively, include self-nodes assuming current node features are also
    # # important for prediction (node own features along with its neighbors)
    # src, dst = np.where(dist_matrix < connection_radius)
    print(f"\nNumber of edges found (directed): {len(src)}")
    # Make the graph undirected by adding reciprocal edges; for each edge A→B,
    # add the reverse edge B→A to assure same information passage both ways
    # between connected points (we can get node properties of A from B or B
    # from A using edge attributes)
    src_undirected = np.concatenate([src, dst])
    dst_undirected = np.concatenate([dst, src])
    print(f"Number of edges after making undirected: {len(src_undirected)}")
    # Create edge_index tensor
    edge_index = torch.tensor(
        np.array([src_undirected, dst_undirected]), dtype=torch.long
    )
    # Edge Attributes
    edge_distances = dist_matrix[src_undirected, dst_undirected]
    # Avoid division by zero that matters for self-nodes
    inverse_distances_squared = torch.tensor(
        1.0 / (edge_distances + 1e-6) ** 2, dtype=torch.float32
    ).unsqueeze(1)  # Shape: [num_edges, 1]

    edge_attr = inverse_distances_squared
    return edge_index, edge_attr


def export_graph_to_html(
    graph,
    coordinates,
    node_indices,
    connection_radius,
    save_path: str,
    dataset_idx: int | None = None,
    dataset_tag: str = "train",
    filename="graph.html",
):
    """Export the graph to an interactive HTML file using Plotly."""
    edge_index, _ = prepare_edge_data(coordinates[node_indices], connection_radius)
    src, dst = edge_index
    fig = visualize_graph(
        coordinates[node_indices], graph.y[node_indices].numpy(), src, dst
    )
    if dataset_idx is not None:
        filename = f"{save_path}/{dataset_tag}_graph_{dataset_idx}.html"
    else:
        filename = f"{save_path}/{dataset_tag}_graph.html"
    fig.write_html(filename)
    print(f"Graph exported to {filename}")


def export_all_graphs_to_html(
    fold_data,
    test_data,
    coordinates,
    connection_radius,
    save_path: str,
):
    for i, graph in enumerate(fold_data):
        node_indices = graph.train_mask
        export_graph_to_html(
            graph,
            coordinates,
            node_indices,
            connection_radius=connection_radius,
            save_path=save_path,
            dataset_idx=i + 1,
            dataset_tag="train",
        )
        node_indices = graph.val_mask
        export_graph_to_html(
            graph,
            coordinates,
            node_indices,
            connection_radius=connection_radius,
            save_path=save_path,
            dataset_idx=i + 1,
            dataset_tag="val",
        )
    node_indices = test_data.test_mask
    export_graph_to_html(
        test_data,
        coordinates,
        node_indices,
        connection_radius=connection_radius,
        save_path=save_path,
        dataset_tag="test",
    )
    node_indices = test_data.calib_mask
    export_graph_to_html(
        test_data,
        coordinates,
        node_indices,
        connection_radius=connection_radius,
        save_path=save_path,
        dataset_tag="calib",
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
    """Plot training and validation losses and accuracies.

    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_accuracies (list): Training accuracies
        val_accuracies (list): Validation accuracies
        title (str): Plot title

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


def plot_confusion_matrix(
    y_true,
    y_pred,
    sample_weights,
    classes: list,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
):
    """Plot confusion matrix using seaborn.

    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        classes (list): Class names

    """
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)
    cm = np.round(cm).astype(int)
    pyplot.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    pyplot.title(f"{title}")
    pyplot.ylabel("True Label")
    pyplot.xlabel("Predicted Label")
    pyplot.title(title)
    if save_path:
        pyplot.savefig(save_path, bbox_inches="tight", dpi=300)



def plot_roc_curve(ax, y_true, y_prob, sample_weights, title="ROC Curve"):
    auc_score = roc_auc_score(y_true, y_prob, sample_weight=sample_weights)
    fpr, tpr, _ = roc_curve(y_true, y_prob, sample_weight=sample_weights)
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="no-skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")


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
