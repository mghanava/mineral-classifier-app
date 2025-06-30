"""Utility functions for drift detection using MMD and energy statistics."""

import os
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel matrix between samples X and Y.

    Args:
        X: tensor of shape (n1, d)
        Y: tensor of shape (n2, d)
        bandwidth: kernel bandwidth parameter

    Returns:
        Kernel matrix of shape (n1, n2)

    """
    # Compute squared (**2) Euclidean distance (L2 norm)
    dist_sq = torch.cdist(X, Y, p=2) ** 2
    # Apply RBF kernel
    return torch.exp(-dist_sq / (2 * bandwidth**2))


def median_bandwidth(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute median heuristic for bandwidth selection.

    Args:
        X: tensor of shape (n1, d)
        Y: tensor of shape (n2, d)

    Returns:
        Median distance as bandwidth

    """
    # Combine samples
    Z = torch.cat([X, Y], dim=0)
    # Compute pairwise distances
    dist_sq = torch.cdist(Z, Z, p=2) ** 2
    # Get upper triangular part (excluding diagonal)
    triu_idx = torch.triu_indices(Z.shape[0], Z.shape[0], offset=1, device=Z.device)
    distances = torch.sqrt(dist_sq[triu_idx[0], triu_idx[1]])

    return torch.median(distances).item()


def auto_regularization(
    X: torch.Tensor | None = None,
    Y: torch.Tensor | None = None,
    C: torch.Tensor | None = None,
) -> float:
    """Automatically select a regularization parameter based on the cost matrix.

    Args:
        X: Optional first sample, tensor of shape (n1, d) (if None, C should be given)
        Y: Optional second sample, tensor of shape (n2, d) (if None, C should be given)
        C: Optional precomputed cost matrix (if None, will compute it)

    Returns:
        Regularization parameter (float), typically a small fraction of the median nonzero distance.

    """
    # Compute pairwise cost matrix (Euclidean distance)
    if C is None:
        if X is not None and Y is not None:
            C = torch.cdist(X, Y, p=2)
        else:
            raise ValueError("Either C or both X and Y must be provided.")
    non_zero_elements = C[C > 0]  # Exclude zero distances
    reg = 0.05 * torch.median(non_zero_elements).item()
    reg = max(reg, 1e-6)
    return reg


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
    dist_XY = torch.cdist(X, Y, p=2)
    term1 = 2 * dist_XY.mean()
    # Term 2: E[||X - X'||] (exclude diagonal)
    dist_XX = torch.cdist(X, X, p=2)
    # Remove diagonal for unbiased estimate
    dist_XX.fill_diagonal_(0)
    term2 = dist_XX.sum() / (n1 * (n1 - 1))

    # Term 3: E[||Y - Y'||] (exclude diagonal)
    dist_YY = torch.cdist(Y, Y, p=2)
    dist_YY.fill_diagonal_(0)
    term3 = dist_YY.sum() / (n2 * (n2 - 1))

    energy_stat = term1 - term2 - term3
    return energy_stat.item()


def sinkhorn_iterations_stable(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Numerically stable Sinkhorn iterations using log-domain arithmetic.

    Args:
        C: Cost matrix of shape (n1, n2)
        a: Source distribution weights of shape (n1,)
        b: Target distribution weights of shape (n2,)
        reg: Regularization parameter (entropy regularization)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Optimal transport matrix P of shape (n1, n2)

    """
    device = C.device
    n1, n2 = C.shape

    # Scale the problem to avoid underflow
    C_max = C.max().item()
    C_scaled = C / (C_max + 1e-12)  # Avoid division by zero
    reg_scaled = reg / C_max  # Adjust reg accordingly

    # Work in log domain for stability
    log_a = torch.log(a)
    log_b = torch.log(b)
    # Initialize dual variables in log domain
    f = torch.zeros(n1, device=device)
    g = torch.zeros(n2, device=device)

    for _ in range(max_iter):
        f_prev = f.clone()

        # Stable log-sum-exp updates
        f = reg_scaled * (
            log_a - torch.logsumexp((-C_scaled + g.unsqueeze(0)) / reg_scaled, dim=1)
        )
        g = reg_scaled * (
            log_b - torch.logsumexp((-C_scaled.T + f.unsqueeze(0)) / reg_scaled, dim=1)
        )

        # Early termination if NaN appears (suggests reg is too small)
        if torch.isnan(f).any() or torch.isnan(g).any():
            raise RuntimeError("NaN detected in Sinkhorn iterations. Increase reg.")

        # Check convergence
        if torch.norm(f - f_prev) < tol:
            break

    # Compute transport matrix in log domain then exponentiate
    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C_scaled) / reg_scaled
    P = torch.exp(log_P)

    return P


def sinkhorn_wasserstein_distance(
    X: torch.Tensor, Y: torch.Tensor, reg: float | None = None, max_iter: int = 100
) -> float:
    """Compute Sinkhorn-regularized Wasserstein distance between two samples.

    Args:
        X: First sample, tensor of shape (n1, d)
        Y: Second sample, tensor of shape (n2, d)
        reg: Regularization parameter (if None, uses heuristic)
        max_iter: Maximum Sinkhorn iterations

    Returns:
        Sinkhorn-Wasserstein distance

    """
    device = X.device
    n1, n2 = X.shape[0], Y.shape[0]

    # Uniform distributions
    a = torch.ones(n1, device=device) / n1
    b = torch.ones(n2, device=device) / n2

    # Compute pairwise cost matrix (Euclidean distance)
    C = torch.cdist(X, Y, p=2)
    # Default regularization: 1% of median distance
    if reg is None:
        reg = auto_regularization(C=C)
    try:
        P = sinkhorn_iterations_stable(C, a, b, reg, max_iter)
        wasserstein_dist = torch.sum(P * C)
        return wasserstein_dist.item()

    except RuntimeError as e:
        print(f"Sinkhorn failed: {e}. Using sliced Wasserstein fallback.")
        # Fallback: 1D Wasserstein per feature (more stable)
        return sum(
            stats.wasserstein_distance(X[:, i].cpu().numpy(), Y[:, i].cpu().numpy())
            for i in range(X.shape[1])
        )


def perform_permutation_test(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_permutations: int = 1000,
    method: Literal["mmd", "energy", "sinkhorn-wasserstein"] = "mmd",
    bandwidth: float | None = None,
    reg: float | None = None,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Perform energy two-sample test with permutation testing.

    Args:
        X: First sample, tensor of shape (n1, d)
        Y: Second sample, tensor of shape (n2, d)
        n_permutations: Number of permutations for p-value estimation
        method: Which test statistic to use, either "mmd" or "energy"
        bandwidth: Kernel bandwidth for MMD test; if None, uses median heuristic
        reg: Regularization parameter (if None, uses automatic selection)
        max_iter: Maximum Sinkhorn iterations

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
        print(f"Auto-selected bandwidth: {bandwidth:.6f}")
        observed_statistic = mmd_unbiased(X, Y, bandwidth)
    elif method == "energy":
        observed_statistic = energy_statistic(X, Y)
    else:
        # Compute regularization parameter if not provided
        if reg is None:
            reg = auto_regularization(X, Y)
            print(f"Auto-selected reg: {reg:.6f}")
        observed_statistic = sinkhorn_wasserstein_distance(X, Y, reg, max_iter)

    # Permutation test
    null = []
    for i in range(n_permutations):
        # Random permutation
        perm_idx = torch.randperm(n_total, device=device)

        # Split permuted data
        X_perm = Z[perm_idx[:n1]]
        Y_perm = Z[perm_idx[n1:]]

        # Compute statistic for permuted data
        if method == "mmd" and bandwidth is not None:
            perm = mmd_unbiased(X_perm, Y_perm, bandwidth)
        elif method == "energy":
            perm = energy_statistic(X_perm, Y_perm)
        else:
            perm = sinkhorn_wasserstein_distance(X_perm, Y_perm, reg, max_iter)
        null.append(perm)
        if (i + 1) % 100 == 0:
            print(f"Permutation {i + 1}/{n_permutations}")

    # Compute p-value; under the null hypothesis, both samples come from the same distribution, so the distance should be small. The p-value represents the probability of observing a distance as large or larger than what is actually observed, assuming the null hypothesis is true.
    p_value = (np.array(null) >= observed_statistic).mean()

    return observed_statistic, p_value


def compare_multidimensional_distributions(
    X, Y, feature_names=None, methods="all", save_path: str | None = None
):
    """Comprehensive visualization suite for comparing high-dimensional distributions.

    Args:
        X: array of shape (n_samples, n_features)
        Y: array of shape (n_samples, n_features)
        feature_names: list of feature names (optional)
        methods: 'all' or list of methods to use
        save_path: directory to save plots (optional)

    """
    # Convert to numpy if needed
    X = X.cpu().numpy() if hasattr(X, "cpu") else np.array(X)
    Y = Y.cpu().numpy() if hasattr(Y, "cpu") else np.array(Y)
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    print(f"Comparing distributions: X{X.shape} vs Y{Y.shape}")
    print("=" * 50)

    available_methods = {
        "marginals": plot_marginal_distributions,
        "pca": plot_pca_comparison,
        "tsne": plot_tsne_comparison,
        "parallel": plot_parallel_coordinates,
        "correlation": plot_correlation_comparison,
        "pairwise": plot_pairwise_relationships,
        "statistical": statistical_comparison,
        "distance": plot_distance_distributions,
    }

    if methods == "all":
        methods = list(available_methods.keys())
    elif isinstance(methods, str):
        methods = [methods]
    for method in methods:
        if method in available_methods:
            print(f"\n{method.upper()} ANALYSIS:")
            print("-" * 30)
            if method == "statistical":
                available_methods[method](X, Y, feature_names)
            else:
                available_methods[method](X, Y, feature_names, save_path=save_path)
        else:
            print(f"Unknown method: {method}")


def plot_marginal_distributions(X, Y, feature_names, save_path):
    """Compare marginal (per-feature) distributions."""
    n_features = X.shape[1]

    # Create subplots - arrange in grid
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols

    _, axes = pyplot.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_features == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_features):
        row, col = i // cols, i % cols
        ax = axes[row][col] if rows > 1 else axes[col]

        # Histograms
        ax.hist(X[:, i], bins=30, alpha=0.6, label="X", density=True, color="blue")
        ax.hist(Y[:, i], bins=30, alpha=0.6, label="Y", density=True, color="red")

        # Add statistics
        x_mean = X[:, i].mean()
        y_mean = Y[:, i].mean()

        ax.axvline(
            x_mean,
            color="blue",
            linestyle="--",
            alpha=0.8,
            label=f"X mean: {x_mean:.2f}",
        )
        ax.axvline(
            y_mean,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Y mean: {y_mean:.2f}",
        )

        ax.set_title(f"{feature_names[i]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Statistical test
        _, p_val_1 = stats.ks_2samp(X[:, i], Y[:, i])
        _, p_val_2 = stats.mannwhitneyu(X[:, i], Y[:, i])
        ax.text(
            0.02,
            0.98,
            f"KS p-val: {p_val_1:.3f} \nMann-Whitney p-val: {p_val_2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    # Hide empty subplots
    for i in range(n_features, rows * cols):
        row, col = i // cols, i % cols
        if rows > 1:
            axes[row][col].set_visible(False)
        else:
            axes[col].set_visible(False)

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "marginal_distributions.png"),
        bbox_inches="tight",
        dpi=300,
    )


def plot_pca_comparison(X, Y, feature_names, save_path):
    """PCA projection comparison."""
    # Combine data for consistent PCA transformation
    combined = np.vstack([X, Y])

    # Fit PCA
    pca = PCA()
    combined_pca = pca.fit_transform(combined)

    # Split back
    X_pca = combined_pca[: len(X)]
    Y_pca = combined_pca[len(X) :]

    # Plot first few components
    n_components = min(4, X.shape[1])

    fig, axes = pyplot.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 2D projections
    for i in range(min(3, n_components - 1)):
        ax = axes[i]
        ax.scatter(
            X_pca[:, 0], X_pca[:, i + 1], alpha=0.6, s=20, label="X", color="blue"
        )
        ax.scatter(
            Y_pca[:, 0], Y_pca[:, i + 1], alpha=0.6, s=20, label="Y", color="red"
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(
            f"PC{i + 2} ({pca.explained_variance_ratio_[i + 1]:.2%} variance)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Explained variance
    ax = axes[3]
    ax.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.grid(True, alpha=0.3)

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "pca.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # Print component loadings
    print("PCA Component Loadings (first 3 components):")
    components_df = pd.DataFrame(
        pca.components_[:3].T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(3)],
    )
    print(components_df.round(3))


def plot_tsne_comparison(X, Y, feature_names, save_path):
    """t-SNE embedding comparison."""
    # Combine data
    combined = np.vstack([X, Y])
    labels = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])

    # Fit t-SNE (with different perplexities)
    perplexities = [5, 30, 50] if len(combined) > 100 else [5, 15]

    fig, axes = pyplot.subplots(
        1, len(perplexities), figsize=(6 * len(perplexities), 5)
    )
    if len(perplexities) == 1:
        axes = [axes]

    for i, perp in enumerate(perplexities):
        if len(combined) < 4 * perp:
            continue

        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            metric="cosine",
            init="pca",
            n_jobs=-1,
        )
        embedding = tsne.fit_transform(combined)

        ax = axes[i]
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="coolwarm", alpha=0.7, s=30
        )
        ax.set_title(f"t-SNE (perplexity={perp})")
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = pyplot.colorbar(scatter, ax=ax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["X", "Y"])

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "tsne.png"),
        bbox_inches="tight",
        dpi=300,
    )


def plot_parallel_coordinates(X, Y, feature_names, save_path):
    """Parallel coordinates plot."""
    # Sample data if too large
    max_samples = 500
    rng = np.random.default_rng()
    if len(X) > max_samples:
        idx_X = rng.choice(len(X), max_samples, replace=False)
        X_sample = X[idx_X]
    else:
        X_sample = X

    if len(Y) > max_samples:
        idx_Y = rng.choice(len(Y), max_samples, replace=False)
        Y_sample = Y[idx_Y]
    else:
        Y_sample = Y

    # Normalize features for parallel coordinates
    all_data = np.vstack([X_sample, Y_sample])
    normalized = (all_data - all_data.min(axis=0)) / (
        all_data.max(axis=0) - all_data.min(axis=0) + 1e-8
    )

    X_norm = normalized[: len(X_sample)]
    Y_norm = normalized[len(X_sample) :]

    fig, ax = pyplot.subplots(figsize=(12, 6))

    # Plot lines
    for i in range(len(X_norm)):
        ax.plot(
            range(len(feature_names)), X_norm[i], color="blue", alpha=0.1, linewidth=0.5
        )

    for i in range(len(Y_norm)):
        ax.plot(
            range(len(feature_names)), Y_norm[i], color="red", alpha=0.1, linewidth=0.5
        )

    # Add mean lines
    ax.plot(
        range(len(feature_names)),
        X_norm.mean(axis=0),
        color="blue",
        linewidth=3,
        label="X mean",
        alpha=0.8,
    )
    ax.plot(
        range(len(feature_names)),
        Y_norm.mean(axis=0),
        color="red",
        linewidth=3,
        label="Y mean",
        alpha=0.8,
    )

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_ylabel("Normalized Value")
    ax.set_title("Parallel Coordinates Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "parallel_coordinates.png"),
        bbox_inches="tight",
        dpi=300,
    )


def plot_correlation_comparison(X, Y, feature_names, save_path):
    """Compare correlation structures."""
    # Compute correlation matrices
    X_corr = np.corrcoef(X.T)
    Y_corr = np.corrcoef(Y.T)
    corr_diff = X_corr - Y_corr

    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))

    # X correlation
    im1 = axes[0].imshow(X_corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("X Correlation Matrix")
    axes[0].set_xticks(range(len(feature_names)))
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_xticklabels(feature_names, rotation=45)
    axes[0].set_yticklabels(feature_names)
    pyplot.colorbar(im1, ax=axes[0])

    # Y correlation
    im2 = axes[1].imshow(Y_corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Y Correlation Matrix")
    axes[1].set_xticks(range(len(feature_names)))
    axes[1].set_yticks(range(len(feature_names)))
    axes[1].set_xticklabels(feature_names, rotation=45)
    axes[1].set_yticklabels(feature_names)
    pyplot.colorbar(im2, ax=axes[1])

    # Difference
    max_diff = np.max(np.abs(corr_diff))
    im3 = axes[2].imshow(corr_diff, cmap="RdBu", vmin=-max_diff, vmax=max_diff)
    axes[2].set_title("Correlation Difference (X - Y)")
    axes[2].set_xticks(range(len(feature_names)))
    axes[2].set_yticks(range(len(feature_names)))
    axes[2].set_xticklabels(feature_names, rotation=45)
    axes[2].set_yticklabels(feature_names)
    pyplot.colorbar(im3, ax=axes[2])

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "correlations.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # Print largest correlation differences
    print("Largest correlation differences:")
    n_features = len(feature_names)
    diffs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            diffs.append(
                (
                    abs(corr_diff[i, j]),
                    feature_names[i],
                    feature_names[j],
                    corr_diff[i, j],
                )
            )

    diffs.sort(reverse=True)
    for diff_val, feat1, feat2, raw_diff in diffs[:5]:
        print(f"  {feat1} vs {feat2}: {raw_diff:+.3f}")


def plot_pairwise_relationships(X, Y, feature_names, save_path):
    """Plot pairwise feature relationships (subset if too many)."""
    n_features = X.shape[1]

    # Select most interesting pairs (highest correlation differences)
    X_corr = np.corrcoef(X.T)
    Y_corr = np.corrcoef(Y.T)
    corr_diff = np.abs(X_corr - Y_corr)

    # Get top pairs
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append((corr_diff[i, j], i, j))

    pairs.sort(reverse=True)
    top_pairs = pairs[: min(6, len(pairs))]

    n_plots = len(top_pairs)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = pyplot.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (diff_val, i, j) in enumerate(top_pairs):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]

        ax.scatter(X[:, i], X[:, j], alpha=0.6, s=20, label="X", color="blue")
        ax.scatter(Y[:, i], Y[:, j], alpha=0.6, s=20, label="Y", color="red")

        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_title(f"Corr diff: {diff_val:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_plots, rows * cols):
        row, col = idx // cols, idx % cols
        if rows > 1:
            axes[row][col].set_visible(False)
        else:
            axes[col].set_visible(False)

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "pairwise_relationships.png"),
        bbox_inches="tight",
        dpi=300,
    )


def statistical_comparison(X, Y, feature_names):
    """Statistical comparison summary."""
    results = []

    for i, name in enumerate(feature_names):
        x_data, y_data = X[:, i], Y[:, i]

        # Basic statistics
        x_mean, x_std = x_data.mean(), x_data.std()
        y_mean, y_std = y_data.mean(), y_data.std()

        # Statistical tests
        ks_stat, ks_p = stats.ks_2samp(x_data, y_data)
        t_stat, t_p = stats.ttest_ind(x_data, y_data)

        results.append(
            {
                "Feature": name,
                "X_mean": x_mean,
                "Y_mean": y_mean,
                "Mean_diff": abs(x_mean - y_mean),
                "X_std": x_std,
                "Y_std": y_std,
                "KS_pval": ks_p,
                "TTest_pval": t_p,
            }
        )

    df = pd.DataFrame(results)
    print("Statistical Comparison Summary:")
    print(df.round(4))

    # Highlight significant differences
    print("\nFeatures with significant differences (p < 0.05):")
    sig_features = df[df["KS_pval"] < 0.05]["Feature"].tolist()
    if sig_features:
        print(f"  KS test: {sig_features}")
    else:
        print("  KS test: None")


def plot_distance_distributions(X, Y, feature_names, save_path):
    """Compare distance distributions."""
    from scipy.spatial.distance import pdist

    # Sample for efficiency
    max_sample = 1000
    X_sample = X[: min(max_sample, len(X))]
    Y_sample = Y[: min(max_sample, len(Y))]

    # Compute distances
    X_dists = pdist(X_sample)
    Y_dists = pdist(Y_sample)

    # Cross distances (sample)
    cross_sample = min(500, len(X_sample), len(Y_sample))
    cross_dists = []
    for i in range(cross_sample):
        for j in range(cross_sample):
            cross_dists.append(np.linalg.norm(X_sample[i] - Y_sample[j]))

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5))

    # Distance histograms
    axes[0].hist(
        X_dists,
        bins=50,
        alpha=0.7,
        label=f"Within X (mean={X_dists.mean():.2f})",
        density=True,
    )
    axes[0].hist(
        Y_dists,
        bins=50,
        alpha=0.7,
        label=f"Within Y (mean={Y_dists.mean():.2f})",
        density=True,
    )
    axes[0].hist(
        cross_dists,
        bins=50,
        alpha=0.7,
        label=f"Cross X-Y (mean={np.mean(cross_dists):.2f})",
        density=True,
    )
    axes[0].set_xlabel("Euclidean Distance")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distance Distributions")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(
        [X_dists, Y_dists, cross_dists], labels=["Within X", "Within Y", "Cross X-Y"]
    )
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].set_title("Distance Box Plots")
    axes[1].grid(True, alpha=0.3)

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "distance_distributions.png"),
        bbox_inches="tight",
        dpi=300,
    )


def jensen_shannon_pc1_analysis(X, Y, n_bins=50):
    """Compute Jensen-Shannon divergence on PC1 of combined data.

    Args:
        X: Base dataset (n_samples, n_features)
        Y: Prediction dataset (n_samples, n_features)
        n_bins: Number of bins for histogram estimation

    Returns:
        js_distance: Jensen-Shannon distance [0,1]
        pc1_X: PC1 values for X
        pc1_Y: PC1 values for Y
        pca: Fitted PCA object

    """
    # Fit PCA on combined data for consistent transformation
    combined_data = np.vstack([X, Y])
    pca = PCA(n_components=1)
    pca.fit(combined_data)

    # Transform both datasets
    pc1_X = pca.transform(X).flatten()
    pc1_Y = pca.transform(Y).flatten()

    # Create common bin edges based on combined range
    min_val = min(pc1_X.min(), pc1_Y.min())
    max_val = max(pc1_X.max(), pc1_Y.max())
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    # Compute histograms (probability distributions)
    hist_X, _ = np.histogram(pc1_X, bins=bin_edges, density=True)
    hist_Y, _ = np.histogram(pc1_Y, bins=bin_edges, density=True)

    # Normalize to probability distributions
    hist_X = hist_X / hist_X.sum()
    hist_Y = hist_Y / hist_Y.sum()

    # Add small epsilon to avoid log(0) issues
    epsilon = 1e-8
    hist_X = hist_X + epsilon
    hist_Y = hist_Y + epsilon

    # Renormalize after adding epsilon
    hist_X = hist_X / hist_X.sum()
    hist_Y = hist_Y / hist_Y.sum()

    # Compute Jensen-Shannon distance (base=2 for [0,1] range)
    js_distance = jensenshannon(hist_X, hist_Y, base=2)

    return js_distance, pc1_X, pc1_Y, pca, (hist_X, hist_Y, bin_edges)


def plot_pc1_distributions(pc1_X, pc1_Y, hist_data, js_distance, save_path):
    """Plot PC1 distributions and Jensen-Shannon result."""
    hist_X, hist_Y, bin_edges = hist_data
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Overlaid histograms
    ax1.hist(pc1_X, bins=50, alpha=0.6, label="X (Base)", color="blue", density=True)
    ax1.hist(
        pc1_Y, bins=50, alpha=0.6, label="Y (Prediction)", color="red", density=True
    )
    ax1.set_xlabel("PC1 Values")
    ax1.set_ylabel("Density")
    ax1.set_title(f"PC1 Distributions\nJS Distance = {js_distance:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Probability distributions used for JS
    ax2.plot(bin_centers, hist_X, "b-", linewidth=2, label="X distribution")
    ax2.plot(bin_centers, hist_Y, "r-", linewidth=2, label="Y distribution")
    ax2.fill_between(bin_centers, hist_X, alpha=0.3, color="blue")
    ax2.fill_between(bin_centers, hist_Y, alpha=0.3, color="red")
    ax2.set_xlabel("PC1 Values")
    ax2.set_ylabel("Probability")
    ax2.set_title("Probability Distributions for JS Calculation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig(
        os.path.join(save_path, "pc1_distributions.png"),
        bbox_inches="tight",
        dpi=300,
    )
    return fig
