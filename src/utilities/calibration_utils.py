"""Utilities for calibrating machine learning model predictions.

This module provides classes and functions for:
- Computing calibration metrics (ECE, MCE)
- Applying various calibration methods:
  - Temperature Scaling
  - Isotonic Regression
  - Platt Scaling
  - Beta Calibration
  - Dirichlet Calibration
- Managing calibrated model pipelines for deployment
"""

import os
from typing import ClassVar, Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data

from src.utilities.early_stopping import EarlyStopping


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
            # All predictions fall into a single bin (or no bins have samples)
            # Use overall accuracy and confidence as single-bin stats
            overall_weight = sample_weights.sum()
            overall_accuracy = (correct @ sample_weights) / overall_weight
            overall_confidence = (max_probs @ sample_weights) / overall_weight
            bin_weights_list = [overall_weight]
            bin_accuracies_list = [overall_accuracy]
            bin_confidences_list = [overall_confidence]
            bin_counts = torch.tensor([len(max_probs)], device=y_prob.device)
            bin_edges = torch.tensor([0.0, 1.0], device=y_prob.device)

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
        X_thresholds = calibrator["X_thresholds"].to(self.device)
        y_thresholds = calibrator["y_thresholds"].to(self.device)

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
        device: torch.device = torch.device("cpu"),
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
            base_model (nn.Module, optional): The original model that outputs logits.
            method (str): Calibration method. Options: "beta" (Beta Calibration)
                or "temperature" (Temperature Scaling).
            device (torch.device): Device for computation.
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
            seed (int): Random seed for reproducibility.

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
        self._validate_inputs(logits_train, labels_train)
        self._setup_class_info(logits_train, labels_train)

        if sample_weights_train is None:
            sample_weights_train = torch.ones_like(
                labels_train, dtype=torch.float32, device=self.device
            )
        if self.method == "isotonic":
            self._fit_non_parametric(logits_train, labels_train, sample_weights_train)
        else:
            self._fit_parametric(logits_train, labels_train, sample_weights_train)

        self.calibrated = True

    def _fit_non_parametric(self, logits, labels, sample_weights):
        model = IsotonicRegressionCalibration(device=self.device)
        if self.classes is None:
            raise RuntimeError("Classes were not properly found during fitting!")
        if self.use_binary_calibration:
            self.iso_calibrators = [
                model._fit_isotonic_calibrator(
                    logits,
                    (labels == self.classes[1]).float(),
                    sample_weights,
                )
            ]
        else:
            # Fit a separate calibrator for each class
            self.iso_calibrators = [
                model._fit_isotonic_calibrator(
                    logits[:, i], (labels == cls).float(), sample_weights
                )
                for i, cls in enumerate(self.classes)
            ]

    def _fit_parametric(self, logits, labels, sample_weights):
        # 1. Initialize model and split data
        self.model = self._create_parametric_model()
        assert self.model is not None, "model is not properly initialized!"
        (
            logits_train,
            logits_val,
            labels_train,
            labels_val,
            sample_weights_train,
            sample_weights_val,
        ) = self._train_val_split(logits, labels, sample_weights)
        # 2. Prepare method-specific transformations
        train_s_prime, train_s_double_prime, val_s_prime, val_s_double_prime = (
            None,
            None,
            None,
            None,
        )
        criterion = None

        if self.method in ["temperature", "dirichlet"]:
            # Temperature/Dirichlet use raw logits
            train_s_prime, train_s_double_prime = logits_train, None
            val_s_prime, val_s_double_prime = logits_val, None
            if self.use_binary_calibration:
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=self._get_pos_weights(labels_train)
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    weight=self._get_class_weights(labels_train)
                )
        elif self.method in ["platt", "beta"]:
            # Calculate s_prime and s_double_prime locally
            train_s_prime, train_s_double_prime = (
                self._calculate_s_prime_s_double_prime(logits_train)
            )
            val_s_prime, val_s_double_prime = self._calculate_s_prime_s_double_prime(
                logits_val
            )
            # For binary or multi-class (multi-label binary cross entropy; i.e, effectively one-vs-rest for each class)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=self._get_pos_weights(labels_train)
            )
        # 3. Train with unified loop
        self._train_model(
            criterion,
            train_data=(
                train_s_prime,
                train_s_double_prime,
                labels_train,
                sample_weights_train,
            ),
            val_data=(val_s_prime, val_s_double_prime, labels_val, sample_weights_val),
        )

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
                f"{self.save_path}/calibration_{self.method}_loss_and_gradient.png"
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
        if self.base_model is None:
            raise ValueError("base_model must be set before prediction")

        with torch.no_grad():
            # Step 1: Get uncalibrated logits
            logits = self.base_model(X.to(self.device))
            # Step 2: AApply calibration and convert to probabilities
            return self._apply_calibration(logits)

    def _compute_weighted_ece(self, y_true, y_prob, sample_weights, n_bins=10):
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

        return ece

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
        if self.model is None:
            raise RuntimeError("Calibration model not initialized")
        if self.method == "temperature":
            return self.model(logits)
        s_prime, s_double_prime = self._calculate_s_prime_s_double_prime(logits)
        return self.model(s_prime, s_double_prime)

    def _parametric_calibrate_multiclass(self, logits):
        """Handle Platt/Beta/Temperature/Dirichlet for multi-class."""
        if self.model is None:
            raise RuntimeError("Calibration model not initialized")

        if self.method in ["temperature", "dirichlet"]:
            return self.model(logits)
        s_prime, s_double_prime = self._calculate_s_prime_s_double_prime(logits)
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
        torch.save(state, os.path.join(filepath, "calibrator.pt"))
        print(f"Calibrated model saved to {filepath}.")

    def _train_val_split(self, logits, labels, sample_weights):
        """Stratified split of calibration data into train/val sets.

        Args:
            logits: Tensor of shape [n_samples, ...]
            labels: Tensor of shape [n_samples]
            sample_weights: Tensor of shape [n_samples]

        Returns:
            Tuple of (train_logits, val_logits, train_labels, val_labels, train_weights, val_weights)

        """
        # Convert to CPU numpy for sklearn (if not already)
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
        indices = torch.arange(len(logits))

        # Stratified split
        test_size = 0.25
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, stratify=labels_np, random_state=self.seed
        )

        # Apply split
        train_logits = logits[train_idx]
        val_logits = logits[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_weights = sample_weights[train_idx]
        val_weights = sample_weights[val_idx]

        return (
            train_logits,
            val_logits,
            train_labels,
            val_labels,
            train_weights,
            val_weights,
        )

    def _create_parametric_model(self):
        """Instantiate the correct parametric model."""
        if self.method == "temperature":
            return TemperatureScaling(
                device=self.device, initial_temperature=self.initial_temperature
            )
        elif self.method == "platt":
            return PlattScaling(n_classes=self.n_classes, device=self.device)
        elif self.method == "beta":
            return BetaCalibration(n_classes=self.n_classes, device=self.device)
        elif self.method == "dirichlet":
            return DirichletCalibration(n_classes=self.n_classes, device=self.device)

    def _validate_inputs(self, logits, labels):
        """Validate shapes/dtypes of logits and labels."""
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"logits must be torch.Tensor, got {type(logits)}")

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits ({logits.shape[0]}), "
                f"labels ({labels.shape[0]})"
            )
        if self.method == "dirichlet" and logits.dim() != 2:
            raise ValueError(
                "Dirichlet calibration requires 2D logits [batch, classes]"
            )

    def _setup_class_info(self, logits, labels):
        """Initialize class-related attributes."""
        self.classes = torch.unique(labels)
        self.n_classes = len(self.classes)

        # Determine if binary (special case)
        self.use_binary_calibration = self.n_classes <= 2 and (
            logits.dim() == 1 or logits.shape[1] == 1
        )
        print(
            f"Using {'binary' if self.use_binary_calibration else 'multi-class'} calibration"
        )
        if self.method == "dirichlet" and self.use_binary_calibration:
            raise ValueError(
                "Dirichlet calibration requires multi-class logits "
                f"(got {logits.shape} for {self.n_classes} classes)"
            )

    def _train_model(self, criterion, train_data, val_data):
        """Unified training loop for all parametric calibration methods.

        Args:
            criterion: Loss function to use for training
            train_data: Tuple of (s_prime, s_double_prime, labels, sample_weights)
            val_data: Tuple of (val_s_prime, val_s_double_prime, val_labels, _)

        """
        # Unpack data
        s_prime, s_double_prime, labels, _ = train_data
        val_s_prime, val_s_double_prime, val_labels, val_sample_weights = val_data
        if self.model is None:
            raise RuntimeError("Calibration model not initialized")
        # Initialize training components
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
        early_stopping = EarlyStopping(
            patience=self.patience_early_stopping,
            min_delta=self.min_delta_early_stopping,
        )

        track_losses = []
        track_gradients = []
        # Training loop
        for epoch in range(self.n_epochs):
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            if self.method in ["temperature", "dirichlet"]:
                calibrated = self.model(s_prime)
                # make a validation data type check to make it robust
                loss = criterion(calibrated, labels)
            else:  # Platt/Beta
                calibrated = self.model(s_prime, s_double_prime)
                if self.n_classes is None:
                    raise ValueError("n_classes must be initialized before training")
                loss = criterion(
                    calibrated, F.one_hot(labels, num_classes=self.n_classes).float()
                )

            # Regularization
            loss += self._compute_regularization()

            # Backward pass
            loss.backward(retain_graph=True)
            optimizer.step()
            track_losses.append(loss.item())
            track_gradients.append(self._calculate_parameter_gradients(self.model))

            # Validation
            with torch.no_grad():
                self.model.eval()
                if self.method in ["temperature", "dirichlet"]:
                    val_calibrated = self.model(val_s_prime)
                else:
                    val_calibrated = self.model(val_s_prime, val_s_double_prime)
                val_probs = (
                    torch.sigmoid(val_calibrated)
                    if self.use_binary_calibration
                    else torch.softmax(val_calibrated, dim=1)
                )
                ece = self._compute_weighted_ece(
                    val_labels, val_probs, sample_weights=val_sample_weights, n_bins=15
                )

            # LR scheduling and early stopping
            scheduler.step(ece)
            early_stopping(ece)

            if early_stopping.early_stop:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if self.verbose:
            # print optimum parameters based on model type
            self._plot_loss_and_gradients(self.model, track_losses, track_gradients)

    # Helper methods used by _train_model
    def _get_class_weights(self, labels):
        """Compute balanced class weights for CrossEntropyLoss."""
        if self.classes is None:
            raise ValueError("classes must be initialized before training")
        if not hasattr(self, "classes") or len(self.classes) <= 2:
            return None
        return torch.tensor(
            compute_class_weight(
                "balanced", classes=self.classes.cpu().numpy(), y=labels.cpu().numpy()
            ),
            device=self.device,
            dtype=torch.float32,
        )

    def _get_pos_weights(self, labels):
        """Compute pos_weight for BCEWithLogitsLoss."""
        # counts = torch.bincount(labels)
        # return (labels.shape[0] - counts) / counts.clamp(min=1)
        if self.use_binary_calibration:
            return labels.size(0) / (2 * labels.sum())
        else:
            if self.n_classes is None:
                raise ValueError("n_classes must be initialized before training")
            labels_train_one_hot = F.one_hot(labels, num_classes=self.n_classes).float()
            # Calculate class weights for balanced loss
            class_counts = labels_train_one_hot.sum(dim=0)
            return labels.size(0) / (2 * class_counts)

    def _compute_regularization(self):
        """Method-specific regularization terms."""
        if self.model is None:
            raise RuntimeError("Calibration model not initialized")
        if self.method == "dirichlet":
            # Add ODIR (Off-Diagonal and Intercept Regularization)
            # Intuition Behind ODIR:
            # Diagonal Elements: The diagonal elements of weights are not regularized because they capture
            # class-specific biases (e.g., some classes may be inherently more confident than others).
            # Off-Diagonal Elements: The off-diagonal elements are regularized because they represent interactions
            # between classes, which are more likely to overfit to noise in the data.
            # Intercept Terms (Biases): The intercept terms are regularized separately because they operate on an
            # additive scale, unlike the multiplicative scale of the transformation matrix.
            weights = self.model.weights
            biases = self.model.biases
            if not isinstance(weights, torch.Tensor) or not isinstance(
                biases, torch.Tensor
            ):
                raise TypeError("Expected weights and biases to be Tensors")
            off_diag = torch.triu(weights, diagonal=1) + torch.tril(
                weights, diagonal=-1
            )
            return torch.mean(self.reg_lambda * torch.pow(off_diag, 2)) + torch.mean(
                self.reg_mu * torch.pow(biases, 2)
            )  # weight penalty + intercept penalty
        else:
            return self.reg_lambda * sum(
                p.pow(2).sum() for p in self.model.parameters()
            )

    def _calculate_s_prime_s_double_prime(self, logits):
        probs_train = (
            torch.sigmoid(logits)
            if self.use_binary_calibration
            else torch.softmax(logits, dim=1)
        )
        probs_train = probs_train.clamp(self.eps, 1 - self.eps)
        return torch.log(probs_train), -torch.log(1 - probs_train)

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

        print(f"Calibrated model loaded from {filepath}.")
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
        self.base_model = base_model.to(self.device)
        if hasattr(base_model, "eval"):
            base_model.eval()


# Pipeline utility class for easier deployment
class CalibrationPipeline:
    """High-level pipeline class for model calibration and deployment."""

    def __init__(
        self, base_model: nn.Module, device: torch.device = torch.device("cpu")
    ):
        """Initialize the CalibrationPipeline with a base model and device.

        Args:
            base_model (nn.Module): The base model to be calibrated.
            device (torch.device): The device to use for computation (default: CPU).

        """
        self.base_model = base_model
        self.device = device
        self.calibrated_model = None

    def calibrate(
        self,
        logits_train: torch.Tensor,
        labels_train: torch.Tensor,
        sample_weights_train: torch.Tensor | None,
        save_path: str,
        method: Literal[
            "temperature", "isotonic", "platt", "beta", "dirichlet"
        ] = "temperature",
        **calibration_kwargs,
    ) -> "CalibrationPipeline":
        """Calibrate the model using training data.

        Args:
            logits_train: Training logits
            labels_train: Training labels
            sample_weights_train (torch.Tensor): Weights of each sample (shape: [n_samples])
            save_path: Path to save the calibrated model
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

    def predict(self, X: Data) -> torch.Tensor:
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
