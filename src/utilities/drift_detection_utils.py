"""Utilities for detecting and analyzing distribution drift between datasets.

This module provides the AnalyzeDrift class which implements various statistical methods
for detecting and visualizing distribution shifts between two datasets, including:
- Maximum Mean Discrepancy (MMD)
- Energy Distance
- Wasserstein Distance
- PCA and Kernel PCA visualizations
- Mutual Information analysis
- Marginal distribution comparisons
"""

import os
import warnings
from typing import Literal

import numpy as np
import ot
import torch
from matplotlib import pyplot
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import mutual_info_regression
from torch_geometric.data import Data

warnings.filterwarnings("ignore")


class AnalyzeDrift:
    """A class for analyzing distribution drift between datasets.

    This class implements various statistical methods to detect and visualize
    distribution shifts between base and prediction datasets, including MMD,
    Energy Distance, Wasserstein Distance, and visualization techniques.
    """

    def __init__(
        self,
        base_data: Data,
        pred_data: Data,
        feature_names: list | None = None,
        gamma: float | None = None,
        n_permutations: int = 1000,
        save_path: str | None = None,
    ):
        """Initialize the AnalyzeDrift class with two datasets to compare.

        Args:
            base_data: Base dataset containing features to analyze.
            pred_data: Prediction dataset to compare against base data.
            feature_names: List of feature names. If None, default names are generated.
            gamma: Parameter for RBF kernel. If None, median heuristic is used.
            n_permutations: Number of permutations for statistical tests.
            save_path: Directory path to save analysis results. If None, results aren't saved.

        """
        self.n_permutations = n_permutations
        X1 = base_data.unscaled_features
        X2 = pred_data.unscaled_features
        # Ensure tensors are on same device
        self.device, self.n_features = X1.device, X1.shape[1]
        self.n1, self.n2 = X1.shape[0], X2.shape[0]
        self.X1 = X1
        self.X2 = X2.to(self.device)
        self.X1_scaled = base_data.x
        self.X2_scaled = pred_data.x
        if gamma is None:
            if self.X1_scaled is not None and self.X2_scaled is not None:
                self.gamma = self._median_heuristic_gamma(
                    self.X1_scaled, self.X2_scaled
                )
                print(f"median heuristic gamma {self.gamma} used in rbf kernel!\n")
            else:
                raise ValueError(
                    "self.X1_scaled and self.X2_scaled must not be None when gamma is None."
                )
        else:
            self.gamma = gamma
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]
        self.save_path = save_path

    def _mmd_unbiased(self, X1: torch.Tensor, X2: torch.Tensor) -> float:
        """Compute unbiased MMD² estimate.

        Args:
            X1: First sample, tensor of shape (n1, d)
            X2: Second sample, tensor of shape (n2, d)

        Returns:
            Unbiased MMD² estimate

        """
        # K_XX
        K_XX = self._rbf_kernel(X1, X1, self.gamma)
        # Remove diagonal for unbiased estimate
        K_XX.fill_diagonal_(0)
        term1 = K_XX.sum() / (self.n1 * (self.n1 - 1))
        # K_YY
        K_YY = self._rbf_kernel(X2, X2, self.gamma)
        K_YY.fill_diagonal_(0)
        term2 = K_YY.sum() / (self.n2 * (self.n2 - 1))
        # K_XY
        K_XY = self._rbf_kernel(X1, X2, self.gamma)
        term3 = K_XY.sum() / (self.n1 * self.n2)

        mmd_sq = term1 + term2 - 2 * term3
        return mmd_sq.item()

    def _energy_statistic(self, X1: torch.Tensor, X2: torch.Tensor) -> float:
        """Compute the energy statistic between two samples.

        Energy statistic: E = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        where X, X' are independent copies from first sample and Y, Y' from second.

        Args:
            X1: First sample, tensor of shape (n1, d)
            X2: Second sample, tensor of shape (n2, d)

        Returns:
            Energy statistic value

        """
        # Term 1: 2 * E[||X - Y||]
        dist_XY = torch.cdist(X1, X2, p=2)
        term1 = 2 * dist_XY.mean()
        # Term 2: E[||X - X'||] (exclude diagonal)
        dist_XX = torch.cdist(X1, X1, p=2)
        # Remove diagonal for unbiased estimate
        dist_XX.fill_diagonal_(0)
        term2 = dist_XX.sum() / (self.n1 * (self.n1 - 1))

        # Term 3: E[||Y - Y'||] (exclude diagonal)
        dist_YY = torch.cdist(X2, X2, p=2)
        dist_YY.fill_diagonal_(0)
        term3 = dist_YY.sum() / (self.n2 * (self.n2 - 1))

        energy_stat = term1 - term2 - term3
        return energy_stat.item()

    def _wasserstein_distance(self, X1: torch.Tensor, X2: torch.Tensor) -> float:
        """Compute the Wasserstein distance (Earth Mover's Distance) between two samples.

        Args:
            X1: First sample, tensor of shape (n1, d)
            X2: Second sample, tensor of shape (n2, d)

        Returns:
            Wasserstein distance as a float.

        """
        a = torch.ones(self.n1) / self.n1
        b = torch.ones(self.n2) / self.n2
        M = ot.dist(X1, X2, metric="sqeuclidean")
        return ot.emd2(a, b, M).item()  # type: ignore

    def _perform_permutation_test(
        self, method: Literal["mmd", "energy", "wasserstein"]
    ) -> tuple[float, float]:
        """Perform energy two-sample test with permutation testing.

        Args:
            method: Which test statistic to use; "mmd", "energy" or "wasserstein".

        Returns:
            Tuple of (observed_statistic, p_value)

        """
        # Combine samples for permutation
        if self.X1_scaled is None or self.X2_scaled is None:
            raise ValueError("self.X1_scaled and self.X2_scaled must not be None.")
        Z = torch.cat([self.X1_scaled, self.X2_scaled], dim=0)
        if method == "mmd":
            observed_statistic = self._mmd_unbiased(self.X1_scaled, self.X2_scaled)
        elif method == "energy":
            observed_statistic = self._energy_statistic(self.X1_scaled, self.X2_scaled)
        else:
            observed_statistic = self._wasserstein_distance(
                self.X1_scaled, self.X2_scaled
            )

        # Permutation test
        null = []
        for i in range(self.n_permutations):
            # Random permutation
            perm_idx = torch.randperm(self.n1 + self.n2, device=self.device)

            # Split permuted data
            X_perm = Z[perm_idx[: self.n1]]
            Y_perm = Z[perm_idx[self.n1 :]]

            # Compute statistic for permuted data
            if method == "mmd":
                perm = self._mmd_unbiased(X_perm, Y_perm)
            elif method == "energy":
                perm = self._energy_statistic(X_perm, Y_perm)
            else:
                perm = self._wasserstein_distance(X_perm, Y_perm)
            null.append(perm)
            if (i + 1) % 100 == 0:
                print(f"Permutation {i + 1}/{self.n_permutations}")

        # Compute p-value; under the null hypothesis, both samples come from the same distribution, so the distance should be small. The p-value represents the probability of observing a distance as large or larger than what is actually observed, assuming the null hypothesis is true.
        p_value = (np.array(null, dtype=np.float32) >= observed_statistic).mean()
        return observed_statistic, p_value

    def _rbf_kernel(
        self, X1: torch.Tensor, X2: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Compute RBF (Gaussian) kernel matrix between samples X and Y.

        Args:
            X1: tensor of shape (n1, d), first set of samples.
            X2: tensor of shape (n2, d), second set of samples.
            gamma: kernel gamma parameter.

        Returns:
            Kernel matrix of shape (n1, n2)

        """
        # Compute squared (**2) Euclidean distance (L2 norm)
        dist_sq = torch.cdist(X1, X2, p=2) ** 2
        # Apply RBF kernel
        return torch.exp(-gamma * dist_sq)

    def _median_heuristic_gamma(self, X1: torch.Tensor, X2: torch.Tensor) -> float:
        """Compute median heuristic for gamma selection.

        Args:
            X1: First sample, tensor of shape (n1, d)
            X2: Second sample, tensor of shape (n2, d)

        Returns:
            Gamma parameter for RBF kernel (1 / (2 * median_dist^2))

        """
        # Combine samples
        Z = torch.cat([X1, X2], dim=0)
        # Compute pairwise distances
        dist = torch.cdist(Z, Z, p=2)
        # Exclude diagonal (self-distances)
        distances = dist[
            ~torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
        ]
        median_dist = torch.median(distances).item() + 1e-8
        gamma = 1.0 / (2 * median_dist**2)
        return gamma

    def _normalized_mutual_info_matrix(
        self,
        X: np.ndarray,
        normalization: str = "min",
        discrete_features: str = "auto",
        random_state: int | None = None,
    ):
        """Compute normalized pairwise mutual information matrix for all features in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
        normalization : {'min', 'joint', 'avg'}, default='min'
            Normalization method:
            - 'min': divide by min(H(X), H(Y)) (most common)
            - 'joint': divide by H(X,Y) (symmetric uncertainty)
            - 'avg': divide by avg(H(X), H(Y))
        discrete_features : {'auto', bool, array-like}, default='auto'
            Whether to consider features as discrete
        random_state : int or RandomState, optional
            Seed for reproducibility

        Returns
        -------
        nmi_matrix : ndarray, shape (n_features, n_features)
            Symmetric matrix of normalized mutual information scores [0,1]

        """
        n_features = X.shape[1]
        nmi_matrix = np.zeros((n_features, n_features))

        # Compute marginal entropies
        entropies = np.zeros(n_features)
        for i in range(n_features):
            if (isinstance(discrete_features, str) and discrete_features == "auto") or (
                isinstance(discrete_features, list | np.ndarray)
                and not discrete_features[i]
            ):
                hist = np.histogram(X[:, i], bins="fd")[0]
                hist = hist / hist.sum()
                entropies[i] = entropy(hist, base=2)
            else:
                counts = np.bincount(X[:, i].astype(int))
                counts = counts / counts.sum()
                entropies[i] = entropy(counts, base=2)

        # Compute pairwise mutual information (symmetric)
        for i in range(n_features):
            mi_i = mutual_info_regression(
                X,
                X[:, i],
                discrete_features=discrete_features,
                random_state=random_state,
            )
            for j in range(i, n_features):
                if i == j:
                    nmi_matrix[i, j] = 1.0
                else:
                    mi_j = mutual_info_regression(
                        X,
                        X[:, j],
                        discrete_features=discrete_features,
                        random_state=random_state,
                    )
                    # Symmetrize MI
                    mi_sym = 0.5 * (mi_i[j] + mi_j[i])
                    # Normalization
                    if normalization == "min":
                        denominator = min(entropies[i], entropies[j])
                    elif normalization == "joint":
                        denominator = entropies[i] + entropies[j] - mi_sym
                    elif normalization == "avg":
                        denominator = (entropies[i] + entropies[j]) / 2
                    else:
                        raise ValueError(
                            "normalization must be 'min', 'joint', or 'avg'"
                        )
                    denominator = max(denominator, 1e-10)
                    nmi_matrix[i, j] = nmi_matrix[j, i] = mi_sym / denominator

        return nmi_matrix

    def _compare_multidimensional_distributions(self):
        # Convert to numpy if needed
        X1 = self.X1.cpu().numpy() if hasattr(self.X1, "cpu") else np.array(self.X1)
        X2 = self.X2.cpu().numpy() if hasattr(self.X2, "cpu") else np.array(self.X2)
        X1_scaled = (
            self.X1_scaled.cpu().numpy()
            if self.X1_scaled is not None and hasattr(self.X1_scaled, "cpu")
            else np.array(self.X1_scaled)
            if self.X1_scaled is not None
            else None
        )
        X2_scaled = (
            self.X2_scaled.cpu().numpy()
            if self.X2_scaled is not None and hasattr(self.X2_scaled, "cpu")
            else np.array(self.X2_scaled)
            if self.X2_scaled is not None
            else None
        )
        available_methods = {
            "distance": self._plot_distance_distributions,
            "marginals": self._plot_marginal_distributions,
            "pairwise": self._plot_pairwise_relationships,
            "pca": self._plot_pca_comparison,
        }
        methods = list(available_methods.keys())
        for method in methods:
            print(f"\n{method.upper()} ANALYSIS:")
            print("-" * 30)
            arg1, arg2 = (
                (X1_scaled, X2_scaled) if method in ["pairwise", "pca"] else (X1, X2)
            )
            if arg1 is not None and arg2 is not None:
                available_methods[method](arg1, arg2)
            else:
                print(f"Cannot perform {method} analysis: input data is None")

    def _plot_distance_distributions(self, X1: np.ndarray, X2: np.ndarray):
        """Compare distance distributions."""
        # Sample for efficiency
        max_sample = 1000
        X_sample = X1[: min(max_sample, len(X1))]
        Y_sample = X2[: min(max_sample, len(X2))]

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
            [X_dists, Y_dists, cross_dists],
            labels=["Within X", "Within Y", "Cross X-Y"],
        )
        axes[1].set_ylabel("Euclidean Distance")
        axes[1].set_title("Distance Box Plots")
        axes[1].grid(True, alpha=0.3)

        pyplot.tight_layout()
        pyplot.show()
        if self.save_path is not None:
            pyplot.savefig(
                os.path.join(self.save_path, "distance_distributions.png"),
                bbox_inches="tight",
                dpi=300,
            )
        return fig

    def _plot_marginal_distributions(self, X1: np.ndarray, X2: np.ndarray):
        """Compare marginal (per-feature) distributions."""
        # Create subplots - arrange in grid
        cols = min(3, self.n_features)
        rows = (self.n_features + cols - 1) // cols

        fig, axes = pyplot.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if self.n_features == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(self.n_features):
            row, col = i // cols, i % cols
            ax = axes[row][col] if rows > 1 else axes[col]

            # Histograms
            ax.hist(X1[:, i], bins=30, alpha=0.6, label="X", density=True, color="blue")
            ax.hist(X2[:, i], bins=30, alpha=0.6, label="Y", density=True, color="red")

            # Add statistics
            x_mean = X1[:, i].mean()
            y_mean = X2[:, i].mean()

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

            ax.set_title(f"{self.feature_names[i]}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Statistical test
            _, p_val_1 = stats.ks_2samp(self.X1[:, i], self.X2[:, i])
            _, p_val_2 = stats.mannwhitneyu(self.X1[:, i], self.X2[:, i])
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
        for i in range(self.n_features, rows * cols):
            row, col = i // cols, i % cols
            if rows > 1:
                axes[row][col].set_visible(False)
            else:
                axes[col].set_visible(False)

        pyplot.tight_layout()
        pyplot.show()
        if self.save_path is not None:
            pyplot.savefig(
                os.path.join(self.save_path, "marginal_distributions.png"),
                bbox_inches="tight",
                dpi=300,
            )
        return fig

    def _plot_pairwise_relationships(self, X1: np.ndarray, X2: np.ndarray):
        X_mi = self._normalized_mutual_info_matrix(X1)
        Y_mi = self._normalized_mutual_info_matrix(X2)
        mi_diff = X_mi - Y_mi
        fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))

        # X correlation
        im1 = axes[0].imshow(X_mi, cmap="coolwarm", vmin=0, vmax=1)
        axes[0].set_title("Base Features Mutual Information Matrix")
        axes[0].set_xticks(range(len(self.feature_names)))
        axes[0].set_yticks(range(len(self.feature_names)))
        axes[0].set_xticklabels(
            self.feature_names, rotation=45, ha="right", rotation_mode="anchor"
        )
        axes[0].set_yticklabels(self.feature_names)
        pyplot.colorbar(im1, ax=axes[0])

        # Y correlation
        im2 = axes[1].imshow(Y_mi, cmap="coolwarm", vmin=0, vmax=1)
        axes[1].set_title("Prediction Features Mutual Information Matrix")
        axes[1].set_xticks(range(len(self.feature_names)))
        axes[1].set_yticks(range(len(self.feature_names)))
        axes[1].set_xticklabels(
            self.feature_names, rotation=45, ha="right", rotation_mode="anchor"
        )
        axes[1].set_yticklabels(self.feature_names)
        pyplot.colorbar(im2, ax=axes[1])

        # Difference
        max_diff = np.max(np.abs(mi_diff))
        im3 = axes[2].imshow(mi_diff, cmap="RdBu", vmin=-max_diff, vmax=max_diff)
        axes[2].set_title("Mutual Information Difference")
        axes[2].set_xticks(range(len(self.feature_names)))
        axes[2].set_yticks(range(len(self.feature_names)))
        axes[2].set_xticklabels(
            self.feature_names, rotation=45, ha="right", rotation_mode="anchor"
        )
        axes[2].set_yticklabels(self.feature_names)
        pyplot.colorbar(im3, ax=axes[2])

        pyplot.tight_layout()
        pyplot.show()
        if self.save_path is not None:
            pyplot.savefig(
                os.path.join(self.save_path, "mutual_information.png"),
                bbox_inches="tight",
                dpi=300,
            )

        return fig

    def _plot_pca_comparison(self, X1: np.ndarray, X2: np.ndarray):
        """PCA projection comparison."""
        # Combine data for consistent PCA transformation
        X_combined_scaled = np.vstack([X1, X2])
        # Fit PCA
        pca = PCA()
        pca.fit(X_combined_scaled)

        # Transform both datasets
        X_pca = pca.transform(X1)
        Y_pca = pca.transform(X2)

        kpca = KernelPCA(n_components=self.n_features, kernel="rbf", gamma=self.gamma)
        kpca.fit(X_combined_scaled)

        # Transform both datasets
        X_kpca = kpca.transform(X1)
        Y_kpca = kpca.transform(X2)

        fig, axes = pyplot.subplots(3, 3, figsize=(18, 15))

        # --- Row 1: PCA scatter plots ---
        pca_pairs = [(0, 1), (0, 2), (1, 2)]
        pca_labels = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
        for idx, (i, j) in enumerate(pca_pairs):
            axes[0, idx].scatter(
                X_pca[:, i], X_pca[:, j], alpha=0.5, label="X", color="blue", marker="o"
            )
            axes[0, idx].scatter(
                Y_pca[:, i], Y_pca[:, j], alpha=0.5, label="Y", color="red", marker="o"
            )
            axes[0, idx].set_xlabel(pca_labels[idx][0])
            axes[0, idx].set_ylabel(pca_labels[idx][1])
            axes[0, idx].set_title(f"PCA: {pca_labels[idx][0]} vs {pca_labels[idx][1]}")
            axes[0, idx].legend()
            axes[0, idx].grid(True, alpha=0.3)

        # --- Row 2: KernelPCA scatter plots ---
        kpca_pairs = [(0, 1), (0, 2), (1, 2)]
        kpca_labels = [("KPCA1", "KPCA2"), ("KPCA1", "KPCA3"), ("KPCA2", "KPCA3")]
        for idx, (i, j) in enumerate(kpca_pairs):
            axes[1, idx].scatter(
                X_kpca[:, i],
                X_kpca[:, j],
                alpha=0.5,
                label="X",
                color="blue",
                marker="^",
            )
            axes[1, idx].scatter(
                Y_kpca[:, i],
                Y_kpca[:, j],
                alpha=0.5,
                label="Y",
                color="red",
                marker="^",
            )
            axes[1, idx].set_xlabel(kpca_labels[idx][0])
            axes[1, idx].set_ylabel(kpca_labels[idx][1])
            axes[1, idx].set_title(
                f"KernelPCA: {kpca_labels[idx][0]} vs {kpca_labels[idx][1]}"
            )
            axes[1, idx].legend()
            axes[1, idx].grid(True, alpha=0.3)
        # --- Row 3: PCA/KPCA variance histograms and info ---
        axes[2, 0].bar(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
            alpha=0.7,
            color="blue",
        )
        axes[2, 0].set_title("PCA Explained Variance")
        axes[2, 0].set_xlabel("PC")
        axes[2, 0].set_ylabel("Variance Ratio")
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].bar(
            range(1, len(kpca.eigenvalues_) + 1),
            kpca.eigenvalues_,
            alpha=0.7,
            color="blue",
        )
        axes[2, 1].set_title("KernelPCA Variance")
        axes[2, 1].set_xlabel("KPC")
        axes[2, 1].set_ylabel("Variance")
        axes[2, 1].grid(True, alpha=0.3)

        axes[2, 2].axis("off")
        axes[2, 2].text(
            0.5,
            0.5,
            "Top: PCA\nMiddle: KernelPCA\nBottom: Variance",
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

        pyplot.tight_layout()
        pyplot.show()
        if self.save_path is not None:
            fig.savefig(
                os.path.join(self.save_path, "pca_kpca_grid.png"),
                bbox_inches="tight",
                dpi=300,
            )
        return fig

    def export_drift_analysis_to_file(self):
        """Export drift analysis results to a text file.

        Performs permutation tests using MMD, energy, and Wasserstein distances
        and writes the results to a text file in the save_path directory.
        Includes the test statistics, p-values, and interpretation of whether
        significant domain shift was detected.

        The results are only exported if save_path was specified during initialization.
        """
        methods: list[Literal["mmd", "energy", "wasserstein"]] = [
            "mmd",
            "energy",
            "wasserstein",
        ]
        if self.save_path is not None:
            results = []
            for method in methods:
                print(f"{method} permutation test ...")
                observed_statistic, p_value = self._perform_permutation_test(
                    method=method
                )
                results.append((method, observed_statistic, p_value))
            with open(os.path.join(self.save_path, "drift_results.txt"), "w") as f:
                for method, observed_statistic, p_value in results:
                    f.write(f"{method} Statistic: {observed_statistic}\n")
                    f.write(f"p-value: {p_value}\n")
                    if p_value < 0.05:
                        f.write(
                            "Domain shift detected between base and prediction datasets.\n"
                        )
                    else:
                        f.write(
                            "No significant domain shift detected between base and prediction datasets.\n"
                        )
                f.write("\n")

    def export_drift_analysis_plots(self):
        """Generate and display visualizations comparing the base and prediction datasets.

        Creates multiple plots to analyze distribution differences including:
        - Distance distributions between samples
        - Marginal distributions of individual features
        - Pairwise feature relationships
        - PCA and Kernel PCA comparisons

        If save_path was specified during initialization, all plots are saved as PNG files.
        """
        self._compare_multidimensional_distributions()
