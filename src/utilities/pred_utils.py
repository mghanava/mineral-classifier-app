import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from torch_geometric.data import Data

from src.utilities.calibration_utils import CalibrationPipeline


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel matrix between samples X and Y.

    Args:
        X: tensor of shape (n1, d)
        Y: tensor of shape (n2, d)
        bandwidth: kernel bandwidth parameter

    Returns:
        Kernel matrix of shape (n1, n2)

    """
    # Compute pairwise squared distances
    X_norm = (X**2).sum(dim=1, keepdim=True)  # (n1, 1)
    Y_norm = (Y**2).sum(dim=1, keepdim=True)  # (n2, 1)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    dist_sq = X_norm + Y_norm.T - 2 * torch.mm(X, Y.T)

    # Apply RBF kernel
    return torch.exp(-dist_sq / (2 * bandwidth**2))


def median_bandwidth(X: torch.Tensor, Y: torch.Tensor, subsample: int = 1000) -> float:
    """Compute median heuristic for bandwidth selection.

    Args:
        X: tensor of shape (n1, d)
        Y: tensor of shape (n2, d)
        subsample: number of pairs to use for median computation. Computing all pairwise distances is O(n²), so we randomly sample subsample points if we have more than that.

    Returns:
        Median distance as bandwidth

    """
    # Combine samples
    Z = torch.cat([X, Y], dim=0)
    n = Z.shape[0]

    # Subsample for efficiency
    if n > subsample:
        idx = torch.randperm(n, device=Z.device)[:subsample]
        Z = Z[idx]

    # Compute pairwise distances
    Z_norm = (Z**2).sum(dim=1, keepdim=True)
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    dist_sq = Z_norm + Z_norm.T - 2 * torch.mm(Z, Z.T)

    # Get upper triangular part (excluding diagonal)
    triu_idx = torch.triu_indices(Z.shape[0], Z.shape[0], offset=1, device=Z.device)
    distances = torch.sqrt(dist_sq[triu_idx[0], triu_idx[1]])

    return torch.median(distances).item()


def mmd_unbiased(X: torch.Tensor, Y: torch.Tensor, bandwidth: float) -> float:
    """Compute unbiased MMD² estimate.

    Args:
        X: tensor of shape (n1, d)
        Y: tensor of shape (n2, d)
        bandwidth: kernel bandwidth

    Returns:
        Unbiased MMD² estimate

    """
    n1, n2 = X.shape[0], Y.shape[0]

    # K_XX
    K_XX = rbf_kernel(X, X, bandwidth)
    # Remove diagonal for unbiased estimate
    K_XX.fill_diagonal_(0)
    term1 = K_XX.sum() / (n1 * (n1 - 1))

    # K_YY
    K_YY = rbf_kernel(Y, Y, bandwidth)
    K_YY.fill_diagonal_(0)
    term2 = K_YY.sum() / (n2 * (n2 - 1))

    # K_XY
    K_XY = rbf_kernel(X, Y, bandwidth)
    term3 = K_XY.sum() / (n1 * n2)

    mmd_sq = term1 + term2 - 2 * term3
    return mmd_sq.item()


def mmd_permutation_test(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_permutations: int = 1000,
    bandwidth: float | None = None,
) -> tuple[float, float, float]:
    """Perform MMD two-sample test with permutation testing.

    Args:
        X: First sample, tensor of shape (n1, d)
        Y: Second sample, tensor of shape (n2, d)
        n_permutations: Number of permutations for p-value estimation
        bandwidth: RBF kernel bandwidth (if None, uses median heuristic)

    Returns:
        Tuple of (mmd_statistic, p_value, bandwidth_used)

    """
    # Ensure tensors are on same device
    device = X.device
    Y = Y.to(device)

    n1, n2 = X.shape[0], Y.shape[0]
    n_total = n1 + n2

    # Compute bandwidth if not provided
    if bandwidth is None:
        bandwidth = median_bandwidth(X, Y)

    # Compute observed MMD statistic
    mmd_observed = mmd_unbiased(X, Y, bandwidth)

    # Combine samples for permutation
    Z = torch.cat([X, Y], dim=0)

    # Permutation test
    mmd_null = []
    for _ in range(n_permutations):
        # Random permutation
        perm_idx = torch.randperm(n_total, device=device)

        # Split permuted data
        X_perm = Z[perm_idx[:n1]]
        Y_perm = Z[perm_idx[n1:]]

        # Compute MMD for permuted data
        mmd_perm = mmd_unbiased(X_perm, Y_perm, bandwidth)
        mmd_null.append(mmd_perm)

    # Compute p-value; What fraction of random permutations gave an MMD statistic as large or larger than what we actually observed?
    # The logic:
    # Null hypothesis: samples come from same distribution
    # If null is true: permuted data should look similar to original data
    # If we see extreme values rarely in permutations: our observed difference is unlikely under the null
    # P-value: probability of seeing our result (or more extreme) by chance alone
    mmd_null = np.array(mmd_null)
    p_value = (mmd_null >= mmd_observed).mean()

    return mmd_observed, p_value, bandwidth


def prediction(
    base_data: Data,
    pred_data: Data,
    model: torch.nn.Module,
    calibrator_path: str,
    n_permutations: int = 5000,
    save_path: str | None = None,
    device: torch.device = torch.device("cpu"),
):
    # hide the labels from model
    hidden_labels = pred_data.y
    pred_data.y = None
    calibrated_probs = CalibrationPipeline.load(
        filepath=calibrator_path,
        base_model=model,
        device=device,
    ).predict(pred_data)
    model.eval()
    with torch.no_grad():
        model.to(device)
        data = pred_data.to(str(device))
        logits = model(data)
        uncalibrated_probs = F.softmax(logits, dim=1)
    # Create a DataFrame with one column per class probability
    calib_prob_array = calibrated_probs.cpu().numpy()
    num_classes = calib_prob_array.shape[1]
    calib_prob_columns = {
        f"calib_prob_class_{i}": calib_prob_array[:, i] for i in range(num_classes)
    }
    uncalib_prob_array = uncalibrated_probs.cpu().numpy()
    uncalib_prob_columns = {
        f"uncalib_prob_class_{i}": uncalib_prob_array[:, i] for i in range(num_classes)
    }
    prob_columns = {**calib_prob_columns, **uncalib_prob_columns}
    calib_entropies = -torch.sum(
        calibrated_probs * torch.log(calibrated_probs + 1e-8), dim=1
    )
    uncalib_entropies = -torch.sum(
        uncalibrated_probs * torch.log(uncalibrated_probs + 1e-8), dim=1
    )
    # Create a DataFrame with the results
    result_df = pd.DataFrame(
        {
            **prob_columns,
            "predicted_label": calib_prob_array.argmax(axis=1),
            "true_label": hidden_labels.cpu().numpy()
            if isinstance(hidden_labels, torch.Tensor)
            else int(hidden_labels)
            if hidden_labels is not None
            else None,
            "calibrated_entropy": calib_entropies.cpu().numpy(),
            "uncalibrated_entropy": uncalib_entropies.cpu().numpy(),
        }
    )
    fig = result_df.hist(figsize=(20, 10))
    if save_path is not None:
        result_df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)
        pyplot.savefig(os.path.join(save_path, "histograms.png"))
    pyplot.close()

    # investigate domain shift using MMD
    print("\nInvestigating domain shift using MMD ...\n")
    mmd_statistic, p_value, _ = mmd_permutation_test(
        base_data.unscaled_features,
        data.unscaled_features,
        n_permutations=n_permutations,
    )
    print(f"MMD Statistic: {mmd_statistic}, p-value: {p_value}")
    return fig
