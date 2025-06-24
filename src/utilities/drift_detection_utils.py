"""Utility functions for drift detection using MMD and energy statistics."""

from typing import Literal

import numpy as np
import torch


def calculate_pairwise_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances between samples X and Y.

    Args:
        X: tensor of shape (n1, d)
        Y: tensor of shape (n2, d)

    Returns:
        Distance matrix of shape (n1, n2)

    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    X_norm = (X**2).sum(dim=1, keepdim=True)  # (n1, 1)
    Y_norm = (Y**2).sum(dim=1, keepdim=True)  # (n2, 1)
    # return pairwise squared distances
    return X_norm + Y_norm.T - 2 * torch.mm(X, Y.T)


def calculate_pairwise_euclidean_distance(
    X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances between samples X and Y.

    Args:
        X: tensor of shape (n1, d)
        Y: tensor of shape (n2, d)

    Returns:
        Squared root distance matrix of shape (n1, n2)

    """
    dist_sq = calculate_pairwise_distances(X, Y)
    # Clamp to avoid numerical issues with sqrt
    dist_sq = torch.clamp(dist_sq, min=0.0)
    # Return the square root of the distances
    return torch.sqrt(dist_sq)


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
    dist_sq = calculate_pairwise_distances(X, Y)
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


def energy_statistic(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute the energy statistic between two samples.

    Energy statistic: E = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    where X, X' are independent copies from first sample and Y, Y' from second.

    Args:
        X: First sample, tensor of shape (n1, d)
        Y: Second sample, tensor of shape (n2, d)

    Returns:
        Energy statistic value

    """
    n1, n2 = X.shape[0], Y.shape[0]

    # Term 1: 2 * E[||X - Y||]
    dist_XY = calculate_pairwise_euclidean_distance(X, Y)
    term1 = 2 * dist_XY.mean()
    # Term 2: E[||X - X'||] (exclude diagonal)
    dist_XX = calculate_pairwise_euclidean_distance(X, X)
    # Remove diagonal for unbiased estimate
    dist_XX.fill_diagonal_(0)
    term2 = dist_XX.sum() / (n1 * (n1 - 1))

    # Term 3: E[||Y - Y'||] (exclude diagonal)
    dist_YY = calculate_pairwise_euclidean_distance(Y, Y)
    dist_YY.fill_diagonal_(0)
    term3 = dist_YY.sum() / (n2 * (n2 - 1))

    energy_stat = term1 - term2 - term3
    return energy_stat.item()


def perform_permutation_test(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_permutations: int = 1000,
    method: Literal["mmd", "energy"] = "mmd",
    bandwidth: float | None = None,
) -> tuple[float, float]:
    """Perform energy two-sample test with permutation testing.

    Args:
        X: First sample, tensor of shape (n1, d)
        Y: Second sample, tensor of shape (n2, d)
        n_permutations: Number of permutations for p-value estimation
        method: Which test statistic to use, either "mmd" or "energy"
        bandwidth: Kernel bandwidth for MMD test; if None, uses median heuristic

    Returns:
        Tuple of (observed_statistic, p_value)

    """
    # Ensure tensors are on same device
    device = X.device
    Y = Y.to(device)
    # Combine samples for permutation
    Z = torch.cat([X, Y], dim=0)

    n1, n2 = X.shape[0], Y.shape[0]
    n_total = n1 + n2

    if method == "mmd":
        # Compute bandwidth if not provided
        if bandwidth is None:
            bandwidth = median_bandwidth(X, Y)
        observed_statistic = mmd_unbiased(X, Y, bandwidth)
    elif method == "energy":
        observed_statistic = energy_statistic(X, Y)

    # Permutation test
    null = []
    for _ in range(n_permutations):
        # Random permutation
        perm_idx = torch.randperm(n_total, device=device)

        # Split permuted data
        X_perm = Z[perm_idx[:n1]]
        Y_perm = Z[perm_idx[n1:]]

        # Compute statistic for permuted data
        perm = (
            mmd_unbiased(X_perm, Y_perm, bandwidth)
            if method == "mmd" and bandwidth is not None
            else energy_statistic(X_perm, Y_perm)
        )
        null.append(perm)

    # Compute p-value; What fraction of random permutations gave an MMD or energy statistic as large or larger than what we actually observed?
    # The logic:
    # Null hypothesis: samples come from same distribution
    # If null is true: permuted data should look similar to original data
    # If we see extreme values rarely in permutations: our observed difference is unlikely under the null
    # P-value: probability of seeing our result (or more extreme) by chance alone
    p_value = (np.array(null) >= observed_statistic).mean()

    return observed_statistic, p_value
