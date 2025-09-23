"""Utility functions for generating and processing synthetic mineral exploration data.

This module provides functions for:
- Generating synthetic mineral exploration datasets
- Constructing and visualizing geospatial graphs
- Scaling and preprocessing data
- Managing data splits for machine learning tasks
"""

import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from scipy.spatial import KDTree, distance_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def _generate_coordinates(
    radius: float,
    depth: float,
    n_samples: int,
    spacing: float,
    existing_sample_coords: np.ndarray | None = None,
    rng: np.random.Generator = np.random.default_rng(42),
) -> np.ndarray:
    """Generate 3D coordinates for samples within a circular area, ensuring minimum spacing.

    This function generates random 3D coordinates (x, y, z) for a specified number of samples within a cylinder defined by a radius and depth. It uses a KDTree to efficiently check for and enforce a minimum spacing between samples, and can optionally avoid existing sample locations.

    Parameters
    ----------
    radius : float
        The radius of the circular area for generating x, y coordinates.
    depth : float
        The maximum depth for generating z coordinates (typically a negative value).
    n_samples : int
        The target number of new samples to generate.
    spacing : float
        The minimum allowable distance between any two samples in the x-y plane.
    existing_sample_coords : np.ndarray | None, optional
        An array of existing sample coordinates (shape: [n_existing, 3]) to consider when enforcing spacing. Defaults to None.
    rng : np.random.Generator, optional
        A random number generator instance for reproducibility. Defaults to np.random.default_rng(42).

    Returns
    -------
    np.ndarray
        An array of shape [n_generated, 3] containing the x, y, z coordinates of the newly generated samples. The number of generated samples may be less than n_samples if the space becomes too crowded to meet the spacing constraint.

    """
    # Initialize
    all_xy = (
        np.empty((0, 2))
        if existing_sample_coords is None
        else existing_sample_coords[:, :2]
    )

    new_points = []

    # Create KDTree for fast distance queries
    tree = None if len(all_xy) == 0 else KDTree(all_xy)

    attempts = 0
    max_attempts = n_samples * 100  # Reasonable limit

    while len(new_points) < n_samples and attempts < max_attempts:
        # Generate candidate point
        candidate_xy = rng.uniform(-radius, radius, 2)

        # Check if within circle
        if np.linalg.norm(candidate_xy) > radius:
            attempts += 1
            continue

        # Check spacing using KDTree
        if tree is not None:
            dist, _ = tree.query(candidate_xy.reshape(1, -1))
            if dist[0] < spacing:
                attempts += 1
                continue

        # Valid point found
        candidate_z = rng.uniform(depth, 0)
        new_point = np.array([candidate_xy[0], candidate_xy[1], candidate_z])
        new_points.append(new_point)

        # Update the tree with the new point
        new_xy = candidate_xy.reshape(1, -1)
        if tree is None:
            all_xy = new_xy
            tree = KDTree(all_xy)
        else:
            all_xy = np.vstack([all_xy, new_xy])
            tree = KDTree(all_xy)

        attempts = 0  # Reset attempts counter

    if len(new_points) < n_samples:
        print(f"Warning: Only generated {len(new_points)} out of {n_samples} boreholes")

    return np.array(new_points)


def _create_hotspots(
    depth: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_hotspots_max: int = 10,
    existing_hotspots: np.ndarray | None = None,
    hotspot_drift: float | None = 0.1,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.ndarray, np.ndarray]:
    """Create mineralization hotspots with optional continuity from previous data.

    Parameters
    ----------
    depth : float
        Maximum depth for hotspot placement
    x_range : tuple[float, float]
        Range for x coordinates
    y_range : tuple[float, float]
        Range for y coordinates
    n_hotspots_max : int
        Number of hotspots to generate
    existing_hotspots : np.ndarray, optional
        Previous hotspot locations to maintain continuity
    hotspot_drift : float
        Small perturbation factor for previous hotspots
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Hotspot locations and their strengths

    """
    # In a mineral exploration context, hotspot strengths represent the maximum concentration or intensity of gold at each "source" location in the simulated area. Not all gold deposits are created equal - some have higher  mineral content than others. Values greater than 1.0 represent "high-grade" hotspots that can potentially yield gold values above the baseline (before applying distance decay). Values below 1.0 represent "lower-grade" hotspots that will produce somewhat weaker signals. The range isn't centered at 1.0 (it's 0.7-1.2) to create a slight positive skew, which is common in real mineral deposits
    n_hotspots = int(rng.integers(1, n_hotspots_max + 1))
    # Initialize hotspots array
    hotspots = np.zeros((n_hotspots, 3))
    # Reuse some previous hotspots for spatial continuity if provided
    if existing_hotspots is not None and hotspot_drift is not None:
        n_keep = n_hotspots // 2  # Keep half of previous hotspots
        n_keep = min(
            n_keep, len(existing_hotspots)
        )  # Make sure we don't exceed available hotspots
        # Add small random walk to hotspot positions
        hotspots[:n_keep] = existing_hotspots[:n_keep] + rng.normal(
            0, hotspot_drift, (n_keep, 3)
        )

        # Generate new hotspots for remaining spots
        new_hotspots = n_hotspots - n_keep
        hotspots[n_keep:, 0] = rng.uniform(x_range[0], x_range[1], new_hotspots)
        hotspots[n_keep:, 1] = rng.uniform(y_range[0], y_range[1], new_hotspots)
        hotspots[n_keep:, 2] = rng.uniform(depth, 0, new_hotspots)
    else:
        # Generate all new hotspots
        hotspots[:, 0] = rng.uniform(x_range[0], x_range[1], n_hotspots)
        hotspots[:, 1] = rng.uniform(y_range[0], y_range[1], n_hotspots)
        hotspots[:, 2] = rng.uniform(depth, 0, n_hotspots)

    # Generate hotspot strengths (mineralization intensities)
    hotspot_strengths = rng.uniform(0.7, 1.2, n_hotspots)

    return hotspots, hotspot_strengths


def _calculate_gold_values(
    n_samples: int,
    coordinates: np.ndarray,
    hotspots: np.ndarray,
    hotspot_strengths: np.ndarray,
    noise_level=0.05,
    exp_decay_factor=0.005,
    rng: np.random.Generator = np.random.default_rng(42),
) -> np.ndarray:
    """Calculate gold values based on proximity to mineralization hotspots.

    This function determines a "gold value" for each sample coordinate based on its distance to the nearest mineralization hotspot. The value is influenced by the hotspot's strength and decreases exponentially with distance. Random noise is added, and the final values are scaled to a [0, 1] range.

    Parameters
    ----------
    n_samples : int
        The number of samples for which to calculate gold values.
    coordinates : np.ndarray
        An array of shape [n_samples, 3] with the x, y, z coordinates of the samples.
    hotspots : np.ndarray
        An array of shape [n_hotspots, 3] with the coordinates of the mineralization hotspots.
    hotspot_strengths : np.ndarray
        An array of shape [n_hotspots,] with the strength of each hotspot.
    noise_level : float, optional
        The standard deviation of the Gaussian noise to add to the gold values. Defaults to 0.05.
    exp_decay_factor : float, optional
        The exponential decay factor for the distance from hotspots. Defaults to 0.005.
    rng : np.random.Generator, optional
        A random number generator instance for reproducibility. Defaults to np.random.default_rng(42).

    Returns
    -------
    np.ndarray
        An array of shape [n_samples,] containing the calculated gold values, scaled to the range [0, 1].

    """
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
    gold_values = closest_strengths * np.exp(-min_distances * exp_decay_factor)
    gold_values += rng.normal(0, noise_level, size=n_samples)

    # scale values to 0-1 probability range to keep the distribution shape
    gold_values = (gold_values - gold_values.min()) / (
        gold_values.max() - gold_values.min()
    )
    # ensure strict [0, 1] bounds (in case of numerical instability)
    gold_values = np.clip(gold_values, 0, 1)
    return gold_values


def _assign_labels(
    gold_values: np.ndarray, n_classes: int, threshold_binary: float
) -> np.ndarray:
    """Convert continuous gold values to discrete class labels.

    This function takes an array of continuous gold values (typically in the range [0, 1]) and assigns a discrete class label to each value. The method of assignment depends on the number of classes.

    Parameters
    ----------
    gold_values : np.ndarray
        An array of continuous gold values to be converted to labels.
    n_classes : int
        The total number of desired classes.
    threshold_binary : float
        The threshold used for binary classification when n_classes is 2. Gold values greater than or equal to this threshold will be assigned class 1, and others class 0.

    Returns
    -------
    np.ndarray
        An array of integer labels corresponding to the input gold values. The labels will range from 0 to n_classes-1.

    """
    if n_classes == 2:  # Binary classification: gold/no gold
        return (gold_values >= threshold_binary).astype(int)
    else:  # Multi-class classification
        # Create bins for digitizing
        bins = np.linspace(0, 1, n_classes, endpoint=False)[1:]  # n_classes-1 bin edges
        return np.digitize(gold_values, bins)  # 0 to n_classes-1


def _change_label_distribution(
    labels: np.ndarray,
    min_samples_per_class: int,
    n_classes: int,
    depth: float,
    x_range: tuple,
    y_range: tuple,
    coordinates: np.ndarray,
    gold_values: np.ndarray,
    rng: np.random.Generator = np.random.default_rng(42),
):
    """Adjust class distribution by adding samples to underrepresented classes.

    This function checks the number of samples in each class and, if any class has fewer samples than `min_samples_per_class`, it generates new synthetic samples for that class to meet the minimum requirement. The new samples are generated with appropriate spatial coordinates and gold values corresponding to their assigned class.

    Parameters
    ----------
    labels : np.ndarray
        The initial array of class labels for the dataset.
    min_samples_per_class : int
        The minimum desired number of samples for each class.
    n_classes : int
        The total number of classes in the dataset.
    depth : float
        The maximum depth for generating z coordinates for new samples.
    x_range : tuple
        A tuple (min, max) specifying the range for generating x coordinates.
    y_range : tuple
        A tuple (min, max) specifying the range for generating y coordinates.
    coordinates : np.ndarray
        The existing array of sample coordinates.
    gold_values : np.ndarray
        The existing array of gold values.
    rng : np.random.Generator, optional
        A random number generator instance for reproducibility. Defaults to np.random.default_rng(42).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the updated labels, coordinates, and gold values. If new samples were added, these arrays will be larger than the input arrays.

    """
    # Calculate class counts and needed samples
    class_counts = np.bincount(labels, minlength=n_classes)
    samples_needed = np.maximum(0, min_samples_per_class - class_counts)
    bins = np.linspace(0, 1, n_classes + 1)
    # Generate new samples for each class that needs them
    new_coordinates = []
    new_gold_values = []
    new_labels = []

    for class_idx, n_needed in enumerate(samples_needed):
        if n_needed <= 0:
            continue
        print(f"Adding {n_needed} more samples for class {class_idx}")
        # Sample gold values within the bin for this class
        gold_bin_min, gold_bin_max = bins[class_idx], bins[class_idx + 1]
        sampled_gold = rng.uniform(gold_bin_min, gold_bin_max, n_needed)
        new_gold_values.append(sampled_gold)
        # Sample coordinates within the spatial bounds
        sampled_x = rng.uniform(x_range[0], x_range[1], n_needed)
        sampled_y = rng.uniform(y_range[0], y_range[1], n_needed)
        sampled_z = rng.uniform(depth, 0, n_needed)
        sampled_coords = np.column_stack([sampled_x, sampled_y, sampled_z])
        new_coordinates.append(sampled_coords)
        new_labels.append(np.full(n_needed, class_idx))

    # Combine new samples with existing data
    if new_coordinates:
        coordinates = np.vstack([coordinates, *new_coordinates])
        gold_values = np.concatenate([gold_values, *new_gold_values])
        labels = np.concatenate([labels, *new_labels])

    return labels, coordinates, gold_values


def _generate_features(
    n_samples: int,
    labels: np.ndarray,
    coordinates: np.ndarray,
    hotspots: np.ndarray,
    gold_values: np.ndarray,
    n_classes: int,
    n_features: int,
    noise_level: float = 0.2,
    existing_prototypes: np.ndarray | None = None,
    prototype_evolution_rate: float | None = 0.3,
    class_separation: float = 2.0,
    class_influence: float = 0.3,
    spatial_weight: float = 0.5,
    interaction_strength: float = 0.15,
    correlation_strength: float = 0.2,
    feature_weight_range: tuple[float, float] = (0.5, 2.0),
    feature_offset_std: float = 0.3,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic synthetic features with spatial, chemical, and class relationships.

    This function creates discriminative features that simulate realistic relationships found in mineral exploration data, including spatial correlations, non-linear chemical relationships, and controlled class separability while maintaining dataset continuity.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate features for. Must match the length of other input arrays (coordinates, labels, gold_values).
    labels : np.ndarray of shape (n_total_samples,)
        Class labels for each sample, typically ranging from 0 to n_classes-1.
        Used to create class-specific feature patterns and prototypes.
    coordinates : np.ndarray of shape (n_total_samples, 3)
        Sample coordinates in 3D space [x, y, z]. Typically represents drill hole locations where z is depth (usually negative values).
    hotspots : np.ndarray of shape (n_hotspots, 3)
        Coordinates of mineralization centers or geological features of interest.
    gold_values : np.ndarray of shape (n_total_samples,)
        Continuous target values (e.g., gold concentrations) in range [0, 1].
    n_classes : int
        Number of distinct classes in the dataset.
    n_features : int
        Number of features to generate.
    noise_level : float, default=0.2
        Base noise level controlling feature variability.
    existing_prototypes : np.ndarray of shape (n_classes, n_features), optional
        Class prototypes from a previous dataset generation cycle.
    prototype_evolution_rate : float, optional, default=0.3
        Rate at which prototypes evolve from previous generation.
    class_separation : float, default=2.0
        Controls the base separation between class centroids in feature space.
    class_influence : float, default=0.3
        Strength of class-specific patterns in feature generation.
    spatial_weight : float, default=0.5
        Weight given to spatial components in feature generation.
    interaction_strength : float, default=0.15
        Strength of interaction terms between features.
    correlation_strength : float, default=0.2
        Strength of correlations between features within classes.
    feature_weight_range : tuple[float, float], default=(0.5, 2.0)
        Range for random feature weights in transformations.
    feature_offset_std : float, default=0.3
        Standard deviation for random feature offsets.
    rng : np.random.Generator, default=np.random.default_rng(42)
        Random number generator for reproducibility.

    Returns
    -------
    features : np.ndarray of shape (n_total_samples, n_features)
        Generated feature matrix with standardized properties.
    prototypes : np.ndarray of shape (n_classes, n_features)
        Class prototype vectors for future generations.

    Notes
    -----
    The function creates features through several mechanisms:
    1. Base Transformations of gold values
    2. Spatial Components and relationships
    3. Class-specific patterns and prototypes
    4. Feature Interactions and correlations
    5. Controlled noise and variability

    Raises
    ------
    ValueError
        If array dimensions are inconsistent.
    np.linalg.LinAlgError
        If covariance matrix becomes singular.

    """
    features = np.zeros((n_samples, n_features))

    # 1. Create base transformations of gold values (non-linear relationships)
    transformations = [
        gold_values,  # Linear
        np.sqrt(gold_values),  # Square root
        gold_values**2,  # Quadratic
        np.log1p(gold_values),  # Log transform
        np.sin(gold_values * np.pi),  # Periodic
    ]

    # 2. Add spatial components
    # Distance to nearest hotspot
    distances = np.sqrt(
        np.sum(
            (coordinates[:, np.newaxis, :] - hotspots[np.newaxis, :, :]) ** 2,
            axis=2,
        )
    )
    min_distance = np.min(distances, axis=1)
    max_distance = np.max(min_distance)
    spatial_proximity = 1 - (min_distance / max_distance)  # Closer = higher value

    # Depth component (normalized)
    depth_normalized = (coordinates[:, 2] - np.min(coordinates[:, 2])) / (
        np.max(coordinates[:, 2]) - np.min(coordinates[:, 2]) + 1e-8
    )

    # Add spatial transformations
    transformations.extend(
        [
            spatial_proximity,
            depth_normalized,
            spatial_proximity * gold_values,  # Interaction term
        ]
    )

    # 3. Generate features using different transformations with tunable weights/offsets
    for i in range(n_features):
        # Select transformation (cycle through available ones)
        base_transform = transformations[i % len(transformations)]

        # Add tunable random weight and offset
        weight = rng.uniform(feature_weight_range[0], feature_weight_range[1])
        offset = rng.normal(0, feature_offset_std)

        features[:, i] = base_transform * weight + offset

    # 4. Create controlled class-specific prototypes
    if existing_prototypes is not None and prototype_evolution_rate is not None:
        # Evolve previous prototypes with tunable evolution rate
        prototypes = existing_prototypes + rng.normal(
            0, noise_level * prototype_evolution_rate, existing_prototypes.shape
        )
    else:
        # Create new prototypes with controlled separation
        prototypes = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            # Create well-separated prototypes using class_separation parameter
            angle = 2 * np.pi * i / n_classes  # Distribute classes in circle
            base_x = class_separation * np.cos(angle)
            base_y = class_separation * np.sin(angle)

            # Fill remaining dimensions with controlled random values
            prototype = rng.normal(0, 0.5, n_features)
            if n_features >= 2:
                prototype[0] = base_x
                prototype[1] = base_y
            else:
                prototype[0] = class_separation * (i / max(n_classes - 1, 1) * 2 - 1)

            prototypes[i] = prototype

    # 5. Add class-specific adjustments with tunable influence
    for i in range(n_samples):
        class_adjustment = prototypes[labels[i]] * class_influence
        features[i] += class_adjustment

    # 6. Add controlled feature interactions
    if n_features >= 2:
        for i in range(0, n_features - 1, 2):
            # Add interaction between pairs of features
            interaction = features[:, i] * features[:, i + 1]
            # Scale interaction to prevent explosion
            interaction = (
                interaction / (np.std(interaction) + 1e-8) * interaction_strength
            )
            features[:, i] += interaction

    # 7. Add class-specific correlation structure
    for class_idx in range(n_classes):
        mask = labels == class_idx
        if not np.any(mask):
            continue

        n_class_samples = np.sum(mask)

        # Create controlled covariance matrix with tunable correlation
        class_correlation = np.eye(n_features) * (1 - correlation_strength)
        class_correlation += correlation_strength / n_features  # Add off-diagonal terms

        # Generate correlated noise
        try:
            correlated_noise = rng.multivariate_normal(
                mean=np.zeros(n_features),
                cov=class_correlation * (noise_level**2),
                size=n_class_samples,
            )
            features[mask] += correlated_noise
        except np.linalg.LinAlgError:
            # Fallback to independent noise if covariance issues
            features[mask] += rng.normal(0, noise_level, (n_class_samples, n_features))

    # 8. Weight spatial vs non-spatial features
    # Apply spatial weighting to relevant features
    for i in range(n_features):
        if i < len(transformations) and any(
            "spatial" in str(type(t)) or "proximity" in str(type(t))
            for t in [transformations[i % len(transformations)]]
        ):
            features[:, i] *= spatial_weight
        elif i >= 2 and (
            "spatial_proximity" in locals() or "depth_normalized" in locals()
        ):  # Spatial features are typically in positions 2+
            # Weight spatial components
            features[:, i] = (1 - spatial_weight) * features[:, i] + spatial_weight * (
                spatial_proximity if "spatial_proximity" in locals() else 0
            )

    # 9. Add final independent noise layer
    final_noise = rng.normal(0, noise_level * 0.5, (n_samples, n_features))
    features += final_noise

    # 9. Normalize features to prevent extreme values
    for i in range(n_features):
        feature_std = np.std(features[:, i])
        if feature_std > 0:
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / feature_std
        # Clip extreme outliers
        features[:, i] = np.clip(features[:, i], -4, 4)

    return features, prototypes


def generate_mineral_data(
    radius: float,
    depth: float = -500,
    n_samples: int = 1000,
    spacing: float = 10,
    n_features: int = 5,
    n_classes: int = 2,
    threshold_binary: float = 0.3,
    min_samples_per_class: int | None = 50,
    n_hotspots_max: int = 10,
    mineral_noise_level: float = 0.05,
    exp_decay_factor: float = 0.005,
    feature_noise_level: float = 0.2,
    class_separation: float = 2.0,
    class_influence: float = 0.3,
    spatial_weight: float = 0.5,
    interaction_strength: float = 0.15,
    correlation_strength: float = 0.2,
    feature_weight_range: tuple[float, float] = (0.5, 2.0),
    feature_offset_std: float = 0.3,
    existing_sample_coords: np.ndarray | None = None,
    existing_prototypes: np.ndarray | None = None,
    prototype_evolution_rate: float | None = 0.3,
    existing_hotspots: np.ndarray | None = None,
    hotspot_drift: float | None = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic mineral exploration data with realistic features.

    Parameters
    ----------
    radius : float
        Radius of the circular area in which points will be generated
    depth : float
        Max ranges for depth coordinates (usually negative)
    n_samples : int
        Number of drill core samples to generate
    spacing : float
        Average distance between samples in meters
    n_features : int
        Number of features to generate
    n_classes : int
        Number of classes for gold concentration.
        Use 2 for binary classification (gold/no gold),
        or higher values for finer-grained concentration levels
    threshold_binary: float
        Threshold for binary classification
    min_samples_per_class : int, optional
        Minimum number of samples required for each class. If None, no minimum is enforced.
    n_hotspots_max : int
        Number of mineralization hotspots to generate in the area.
    mineral_noise_level : float
        Noise level for gold value generation (default: 0.05)
    exp_decay_factor : float
        Decay factor for gold value distance falloff (default: 0.005)
    feature_noise_level : float
        Noise level for feature generation (default: 0.2)
    class_separation : float
        Controls the base separation between class centroids in feature space.
    class_influence : float
        Strength of class-specific patterns in feature generation.
    spatial_weight : float
        Weight given to spatial components in feature generation.
    interaction_strength : float
        Strength of interaction terms between features.
    correlation_strength : float
        Strength of correlations between features within classes.
    feature_weight_range : tuple[float, float]
        Range for random feature weights in transformations.
    feature_offset_std : float
        Standard deviation for random feature offsets.
    existing_sample_coords : np.ndarray, optional
        Array of existing point coordinates to avoid when generating new points.
    existing_prototypes : np.ndarray, optional
        Previous feature prototypes to maintain feature space continuity.
    prototype_evolution_rate : float, optional
        Rate at which prototypes evolve from previous generation.
    existing_hotspots : np.ndarray, optional
        Previous hotspot locations to maintain spatial continuity.
    hotspot_drift : float, optional
        Small perturbation factor for previous hotspots.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    coordinates : ndarray
        Array of shape (n_samples, 3) containing x, y, z coordinates.
    features : ndarray
        Scaled array of shape (n_samples, n_features) containing mineral features.
    labels : ndarray
        Array of shape (n_samples,) containing gold concentration labels (0-n_classes).
    hotspots : ndarray
        Array of shape (n_hotspots, 3) containing hotspot coordinates.
    prototypes : ndarray
        Array of shape (n_classes, n_features) containing class prototype vectors.

    """
    rng = np.random.default_rng(seed)  # Set random seed for reproducibility
    # 1. Generate spatial coordinates with appropriate spacing
    coordinates = _generate_coordinates(
        radius, depth, n_samples, spacing, existing_sample_coords, rng
    )
    x_range = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_range = coordinates[:, 1].min(), coordinates[:, 1].max()
    # 2. Create mineralization hotspots and their strengths (mineralization centers)
    hotspots, hotspot_strengths = _create_hotspots(
        depth, x_range, y_range, n_hotspots_max, existing_hotspots, hotspot_drift, rng
    )
    # 3. Calculate gold values based on distance to nearest hotspot
    gold_values = _calculate_gold_values(
        n_samples,
        coordinates,
        hotspots,
        hotspot_strengths,
        mineral_noise_level,
        exp_decay_factor,
        rng=rng,
    )
    # 4. Convert to categorical labels
    labels = _assign_labels(gold_values, n_classes, threshold_binary)
    # 5. Generate features for all samples
    n_total_samples = n_samples
    features, prototypes = _generate_features(
        n_total_samples,
        labels,
        coordinates,
        hotspots,
        gold_values,
        n_classes,
        n_features,
        feature_noise_level,
        existing_prototypes,
        prototype_evolution_rate,
        class_separation,
        class_influence,
        spatial_weight,
        interaction_strength,
        correlation_strength,
        feature_weight_range,
        feature_offset_std,
        rng,
    )
    # (Optional) 6. Check if we have the minimum number of samples per class and redistribute if needed
    if min_samples_per_class is not None:
        labels, coordinates, gold_values = _change_label_distribution(
            labels,
            min_samples_per_class,
            n_classes,
            depth,
            x_range,
            y_range,
            coordinates,
            gold_values,
            rng,
        )
        # Regenerate features after changing label distribution
        n_total_samples = len(coordinates)
        features, prototypes = _generate_features(
            n_total_samples,
            labels,
            coordinates,
            hotspots,
            gold_values,
            n_classes,
            n_features,
            feature_noise_level,
            existing_prototypes,
            prototype_evolution_rate,
            class_separation,
            class_influence,
            spatial_weight,
            interaction_strength,
            correlation_strength,
            feature_weight_range,
            feature_offset_std,
            rng,
        )
    # Print summary of generated data
    print("Label distribution:")
    for i in range(n_classes):
        count = np.sum(labels == i)
        print(f"Class {i}: {count} points ({count / n_total_samples * 100:.2f}%)")
    return coordinates, features, labels, hotspots, prototypes


def visualize_graph(
    coordinates: np.ndarray,
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    edge_opacity=0.3,
    labels_map=None,
    title="3D Geospatial Graph",
):
    """Create an interactive 3D visualization of the geospatial graph.

    Args:
        coordinates (array-like): Coordinate points for constructing the graph.
        labels (array-like): Node labels for stratification.
        src (array-like): Source node indices for edges.
        dst (array-like): Destination node indices for edges.
        edge_opacity (float): Opacity of edges between nodes.
        labels_map (dict, optional): Mapping from label values to label names.
        title (str): Title for the visualization.

    Returns:
        None: Displays the interactive plot.

    """
    # Create color map for different classes
    classes = np.unique(labels)
    unique_classes = len(classes)
    cmap = get_cmap("Set1")
    colors = cmap(np.linspace(0, 1, unique_classes))
    color_map = {
        i: f"rgb({int(255 * c[0])},{int(255 * c[1])},{int(255 * c[2])})"
        for i, c in enumerate(colors)
    }
    if labels_map is None:
        labels_map = {i: f"Class {i}" for i in range(unique_classes)}
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
        text=[f"Prospectivity: {labels_map[label]}" for label in labels],
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
    # Add legend entries for classes
    class_count = [np.sum(labels == i) for i in classes]
    for label, color in color_map.items():
        count = class_count[label]
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"size": 10, "color": color},
                name=f"{labels_map[label]} ({count})",
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


def _split_graph_no_leakage(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_weight: torch.Tensor,
    n_splits: int | None,
    test_size: float | None,
    calib_size: float | None,
    seed: int | None,
) -> tuple[list[tuple[Data, Data]], Data, Data]:
    """Split graph data without leakage through nodes or edges.

    Args:
        x: Node feature matrix of shape [num_nodes, num_features]
        y: Node labels of shape [num_nodes]
        edge_index: Graph connectivity in COO format of shape [2, num_edges]
        edge_attr: Edge feature matrix of shape [num_edges, num_edge_features]
        edge_weight: Edge weight matrix of shape [num_edges]
        n_splits (int, optional): Number of folds for cross-validation splitting
        test_size (float, optional): Fraction of data to use for testing (between 0 and 1)
        calib_size (float, optional): Fraction of data to use for calibration (between 0 and 1)
        seed (int, optional): Random seed for reproducibility

    Returns:
        tuple: (fold_data, test_data, calib_data) where:
            - fold_data is a list of (train_data, val_data) tuples for each fold
            - test_data is a Data object containing the test set
            - calib_data is a Data object containing the calibration set

    """
    features = x.numpy()
    labels = y.numpy()
    n_nodes = len(labels)

    # First split into train+val and temp (test+calib)
    train_val_idx, temp_idx = train_test_split(
        np.arange(n_nodes),
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    # Split temp into test and calibration
    test_idx, calib_idx = train_test_split(
        temp_idx,
        test_size=calib_size,
        stratify=labels[temp_idx],
        random_state=seed,
    )

    if n_splits is None:
        raise ValueError("n_splits must be specified when should_split is True.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_data = []

    # Generate folds from the train+val data
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(
        skf.split(features[train_val_idx], labels[train_val_idx])
    ):
        # Map fold indices back to original indices
        train_idx = train_val_idx[train_fold_idx]
        val_idx = train_val_idx[val_fold_idx]
        # Create separate subgraphs for train and val
        train_data = _create_subgraph(
            x, y, edge_index, edge_attr, edge_weight, train_idx, fold_idx, "train"
        )
        val_data = _create_subgraph(
            x, y, edge_index, edge_attr, edge_weight, val_idx, fold_idx, "val"
        )
        fold_data.append((train_data, val_data))
    # Create test and calibration data
    test_data = _create_subgraph(
        x, y, edge_index, edge_attr, edge_weight, test_idx, None, "test"
    )
    calib_data = _create_subgraph(
        x, y, edge_index, edge_attr, edge_weight, calib_idx, None, "calib"
    )

    return fold_data, test_data, calib_data


def _create_subgraph(
    x, y, edge_index, edge_attr, edge_weight, node_indices, fold_idx, split_type
):
    """Create a subgraph containing only specified nodes and their connections."""
    node_indices = torch.tensor(node_indices, dtype=torch.long)

    # Extract subgraph with only edges between nodes in node_indices
    sub_edge_index, sub_edge_attr, sub_edge_mask = subgraph(
        node_indices,
        edge_index,
        edge_attr,
        relabel_nodes=True,
        num_nodes=x.size(0),
        return_edge_mask=True,
    )

    # Extract node features and labels for the subgraph
    sub_x = x[node_indices]
    sub_y = y[node_indices]

    data = Data(
        x=sub_x,
        y=sub_y,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        edge_weight=edge_weight[sub_edge_mask] if edge_weight is not None else None,
        original_node_indices=node_indices,  # Keep track of original indices
        fold=fold_idx,
        split_type=split_type,
    )

    return data


def construct_graph(
    coordinates: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    k_nearest: int | None = None,
    connection_radius: float | None = None,
    distance_percentile: float | None = None,
    add_self_loops: bool = True,
    n_splits: int | None = None,
    test_size: float | None = None,
    calib_size: float | None = None,
    seed: int | None = None,
    scaler: ScalerType | None = RobustScaler(),
    should_split: bool = True,
    make_edge_weight: bool = True,
    make_edge_weight_method: str | None = "minmax",
) -> tuple[Data, list[tuple[Data, Data]], Data, Data] | Data:
    """Create graphs from geospatial data using distance matrix with a held-out test set.

    Args:
        coordinates (array-like): Coordinate points for constructing the graph
        features (array-like): Node features
        labels (array-like): Node labels for stratification
        k_nearest (int, optional): Number of nearest neighbors to connect for each node
        connection_radius (float, optional): Distance threshold to consider interconnected nodes
        distance_percentile (float, optional): Percentile of distances to use as connection threshold
        add_self_loops (bool): Whether to add self-loops to the graph (default: True)
        n_splits (int): Number of folds for cross-validation
        test_size (float): Proportion of data to use as test set (e.g., 0.2 for 20%)
        calib_size (float): Proportion of data to use as calibration set (e.g., 0.5 for 50%)
        seed (int): Random seed for reproducibility
        scaler (ScalerType): Scaler to use for feature scaling (default: RobustScaler)
        should_split (bool): Whether to perform train/val/test split (default: True, if False, only returns the base graph without splits)
        make_edge_weight (bool): Whether to compute edge weights (default: True)
        make_edge_weight_method (str, optional): Method to normalize edge weights ('minmax', 'standard', 'softmax', or 'log')

    Returns:
        If should_split is True:
            tuple: (base_data, fold_data, test_data)
                - base_data (Data): PyG Data object with all nodes/features/edges
                - fold_data (list[Data]): List of PyG Data objects for each fold (train/val splits)
                - test_data (Data): PyG Data object for test (and optionally calibration) set
        If should_split is False:
            Data: PyG Data object with all nodes/features/edges

    """
    # Convert features and labels to torch tensors
    if scaler is not None:
        scaled_features = scale_data(data=features, scaler=scaler)
    else:
        scaled_features = features
    x = torch.tensor(scaled_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    edge_index, edge_attr, edge_weight = prepare_edge_data(
        coordinates,
        k_nearest,
        connection_radius,
        distance_percentile,
        add_self_loops,
        make_edge_weight,
        make_edge_weight_method,
    )

    base_data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        coordinates=torch.tensor(coordinates, dtype=torch.float32),
        unscaled_features=torch.tensor(
            features, dtype=torch.float32
        ),  # store for data drift detection
    )

    if should_split and n_splits is not None and edge_weight is not None:
        fold_data, test_data, calib_data = _split_graph_no_leakage(
            x,
            y,
            edge_index,
            edge_attr,
            edge_weight,
            n_splits,
            test_size,
            calib_size,
            seed,
        )
        return base_data, fold_data, test_data, calib_data
    return base_data


def _create_graph_adaptive_distance(
    dist_matrix: np.ndarray, add_self_loops: bool, percentile: float
) -> tuple[np.ndarray, np.ndarray]:
    dist_matrix_for_percentile = dist_matrix.copy()
    # Mask out self-distances (set diagonal to infinity)
    np.fill_diagonal(dist_matrix_for_percentile, np.inf)
    # Vectorized percentile computation for all nodes at once
    thresholds = np.percentile(dist_matrix_for_percentile, percentile, axis=1)
    if not add_self_loops:
        # Create boolean mask for connections
        connections = (dist_matrix <= thresholds[:, np.newaxis]) & (dist_matrix > 0.0)
    else:
        # Include self-nodes assuming current node features are also important for prediction (node own features)
        connections = dist_matrix <= thresholds[:, np.newaxis]
    # Get source and destination indices
    src, dst = np.where(connections)
    return src, dst


def _create_graph_fixed_distance(
    dist_matrix: np.ndarray, add_self_loops: bool, connection_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    if not add_self_loops:
        src, dst = np.where((dist_matrix < connection_radius) & (dist_matrix > 0))
    else:
        src, dst = np.where(dist_matrix < connection_radius)
    return src, dst


def _create_graph_k_nearest(
    coordinates: np.ndarray, add_self_loops: bool, k: int
) -> tuple[np.ndarray, np.ndarray]:
    tree = KDTree(coordinates)
    _, indices = tree.query(coordinates, k=k + 1)

    if add_self_loops:
        # Keep all k+1 neighbors (including self)
        src = np.repeat(np.arange(len(coordinates)), k + 1)
        dst = indices.flatten()
    else:
        # Keep only k neighbors (excluding self)
        src = np.repeat(np.arange(len(coordinates)), k)
        # Ensure indices is 2D for consistent slicing
        assert indices.ndim == 2, "Expected indices to be a 2D array."
        dst = indices[:, 1:].flatten()  # Skip first column (self)

    return src, dst


def prepare_edge_data(
    coordinates: np.ndarray,
    k_nearest: int | None = None,
    connection_radius: float | None = None,
    distance_percentile: float | None = None,
    add_self_loops: bool = False,
    make_edge_weight: bool = False,
    make_edge_weight_method: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepare edge connectivity and attributes for a graph neural network from coordinate data. This function computes pairwise distances between points and creates edge connections based on a distance threshold, making the resulting graph undirected. It also generates edge attributes including both raw distances and inverse distances.

    Args:
        coordinates (numpy.ndarray): Array of point coordinates with shape [num_nodes, num_dimensions]
        k_nearest (int, optional): Number of nearest neighbors to connect for each node
        connection_radius (float, optional): Distance threshold to consider interconnected nodes
        distance_percentile (float, optional): Percentile of distances to use as connection threshold
        add_self_loops (bool): Whether to add self-loops to the graph (default: True)
        make_edge_weight (bool): Whether to compute edge weights (default: True)
        make_edge_weight_method (str, optional): Method to normalize edge weights ('minmax', 'standard', 'softmax', or 'log')

    Returns:
        tuple: Contains:
            - edge_index (torch.Tensor): Tensor of shape [2, num_edges] containing source and destination node indices for each edge
            - edge_attr (torch.Tensor): Tensor of shape [num_edges, 2] containing edge attributes [inverse_distance, raw_distance] for each edge
            - edge_weight (torch.Tensor or None): Tensor of shape [num_edges] containing normalized edge weights, or None if not computed
    Notes:
        - Self-loops are explicitly excluded (nodes cannot connect to themselves)
        - The graph is made undirected by adding reciprocal edges
        - Edge attributes could include both inverse squared distance (1/d) and raw distance (d)

    """
    # Compute pairwise Euclidean distances
    dist_matrix = distance_matrix(coordinates, coordinates)
    if k_nearest is not None:
        src, dst = _create_graph_k_nearest(coordinates, add_self_loops, k_nearest)
    elif distance_percentile is not None:
        src, dst = _create_graph_adaptive_distance(
            dist_matrix, add_self_loops, distance_percentile
        )
    elif connection_radius is not None:
        src, dst = _create_graph_fixed_distance(
            dist_matrix, add_self_loops, connection_radius
        )
    else:
        raise ValueError(
            "Either k_nearest or connection_radius or distance_percentile must be set."
        )
    # Make the graph undirected by adding reciprocal edges; for each edge A→B,
    # add the reverse edge B→A to assure same information passage both ways
    # between connected points (we can get node properties of A from B or B
    # from A using edge attributes)
    src_undirected = np.concatenate([src, dst])
    dst_undirected = np.concatenate([dst, src])

    # Create edge_index tensor
    edge_index = torch.tensor(
        np.array([src_undirected, dst_undirected]), dtype=torch.long
    )
    # Edge Attributes
    edge_distances = dist_matrix[src_undirected, dst_undirected]
    # Avoid division by zero that matters for self-nodes
    inverse_distances_squared = torch.tensor(
        1.0 / (edge_distances + 1e-6) ** 2, dtype=torch.float32
    )
    edge_attr = inverse_distances_squared.unsqueeze(1)  # Shape: [num_edges, 1]
    # Edge weights
    if not make_edge_weight:
        edge_weight = None
    else:
        edge_weight = normalize_edge_weight(edge_attr, make_edge_weight_method)
    return edge_index, edge_attr, edge_weight


def export_graph_to_html(
    graph,
    coordinates: np.ndarray,
    node_indices: np.ndarray | None,
    k_nearest: int,
    connection_radius: float,
    distance_percentile: float,
    add_self_loops: bool,
    save_path: str,
    labels_map: dict[int, str],
    dataset_idx: int | None = None,
    dataset_tag: str = "train",
    filename="graph.html",
):
    """Export the graph to an interactive HTML file using Plotly."""
    coordinates = coordinates[node_indices] if node_indices is not None else coordinates
    labels = (
        graph.y[node_indices].numpy() if node_indices is not None else graph.y.numpy()
    )
    edge_index, _, _ = prepare_edge_data(
        coordinates, k_nearest, connection_radius, distance_percentile, add_self_loops
    )
    src, dst = edge_index
    n_nodes, n_edges = graph.num_nodes, graph.num_edges
    network_density = _calculate_network_density(n_nodes, n_edges)
    avg_degree = _calculate_average_degree(n_nodes, n_edges)
    title = (
        f"Graph statistics:\n"
        f"Number of nodes: {n_nodes}\n"
        f"Number of edges: {n_edges}\n"
        f"Network degree: {network_density:.2f}\n"
        f"Average node degree: {avg_degree:.2f}\n"
        f"Has isolated nodes: {graph.has_isolated_nodes()}\n"
        f"Has self-loops: {graph.has_self_loops()}"
    )

    fig = visualize_graph(
        coordinates,
        labels,
        np.array(src),
        np.array(dst),
        labels_map=labels_map,
        title=title,
    )
    if dataset_idx is not None:
        filename = f"{save_path}/{dataset_tag}_graph_{dataset_idx}.html"
    else:
        filename = f"{save_path}/{dataset_tag}_graph.html"
    fig.write_html(
        filename, full_html=False, include_plotlyjs="cdn", config={"responsive": True}
    )
    print(f"Graph exported to {filename}.")


def export_all_graphs_to_html(
    fold_data: list[tuple[Data, Data]],
    test_data: Data,
    calib_data: Data,
    coordinates: np.ndarray,
    k_nearest: int,
    connection_radius: float,
    distance_percentile: float,
    add_self_loops: bool,
    labels_map: dict[int, str],
    save_path: str,
):
    """Export all train, validation, test, and calibration graphs to interactive HTML files.

    Args:
        fold_data (list[Data]): List of PyG Data objects for each fold (train/val splits).
        test_data (Data): PyG Data object for the test set.
        calib_data (Data): PyG Data object for the calibration set.
        coordinates (np.ndarray): Array of node coordinates.
        k_nearest (int, optional): Number of nearest neighbors to connect for each node
        connection_radius (float, optional): Distance threshold to consider interconnected nodes
        distance_percentile (float, optional): Percentile of distances to use as connection threshold
        add_self_loops (bool): Whether to add self-loops to the graph (default: True).
        labels_map (dict[int, str]): Mapping from label indices to label names.
        save_path (str): Directory path to save the exported HTML files.

    Returns:
        None

    """
    for i, data in enumerate(fold_data):
        train_data, val_data = data
        export_graph_to_html(
            train_data,
            coordinates[train_data.original_node_indices],
            None,
            k_nearest=k_nearest,
            connection_radius=connection_radius,
            distance_percentile=distance_percentile,
            add_self_loops=add_self_loops,
            save_path=save_path,
            labels_map=labels_map,
            dataset_idx=i + 1,
            dataset_tag="train",
        )
        export_graph_to_html(
            val_data,
            coordinates[val_data.original_node_indices],
            None,
            k_nearest=k_nearest,
            connection_radius=connection_radius,
            distance_percentile=distance_percentile,
            add_self_loops=add_self_loops,
            save_path=save_path,
            labels_map=labels_map,
            dataset_idx=i + 1,
            dataset_tag="val",
        )
    export_graph_to_html(
        test_data,
        coordinates[test_data.original_node_indices],
        None,
        k_nearest=k_nearest,
        connection_radius=connection_radius,
        distance_percentile=distance_percentile,
        add_self_loops=add_self_loops,
        save_path=save_path,
        labels_map=labels_map,
        dataset_tag="test",
    )
    export_graph_to_html(
        calib_data,
        coordinates[calib_data.original_node_indices],
        None,
        k_nearest=k_nearest,
        connection_radius=connection_radius,
        distance_percentile=distance_percentile,
        add_self_loops=add_self_loops,
        save_path=save_path,
        labels_map=labels_map,
        dataset_tag="calib",
    )


def no_coordinate_overlap(coords1: torch.Tensor, coords2: torch.Tensor) -> bool:
    """Check if there is no coordinate overlap between two sets of coordinates.

    Parameters
    ----------
    coords1 : torch.Tensor
        First set of coordinates to compare
    coords2 : torch.Tensor
        Second set of coordinates to compare

    Returns
    -------
    bool
        True if there is no overlap between coordinates, False otherwise

    """
    # Compare all pairs using broadcasting
    matches = torch.all(
        coords1[:, None] == coords2, dim=2
    )  # dim=2 ensures comparing each element of coords1 with elements of coords2

    return not torch.any(matches)


def analyze_feature_discrimination(
    features,
    labels,
    save_path: str,
    class_names: list,
    scaler: ScalerType = RobustScaler(),
):
    """Analyze and visualize feature discrimination using t-SNE and silhouette scores.

    Parameters
    ----------
    features : array-like
        Feature matrix to analyze
    labels : array-like
        Target labels for each sample
    save_path : str
        Path to save the visualization plot
    class_names : list
        List of class names for the legend
    scaler : ScalerType, optional
        Scaler to use for feature normalization, by default RobustScaler()

    Returns
    -------
    None
        Saves the visualization plot to the specified path

    """
    features_scaled = scale_data(data=features, scaler=scaler)
    silhouette = silhouette_score(features_scaled, labels)

    fig, axes = pyplot.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"t-SNE with Different Perplexity Values \n The Silhouette Coefficient {silhouette:.2f}",
        fontsize=16,
    )

    perplexities = [5, 30, 50, 100]
    cmap = get_cmap("Set1")
    colors = cmap(np.linspace(0, 1, len(class_names)))

    for i, perplexity in enumerate(perplexities):
        row, col = i // 2, i % 2
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=perplexity, max_iter=1000
        )
        features_tsne = tsne.fit_transform(features_scaled)
        # Create scatter plot with discrete colors
        axes[row, col].scatter(
            features_tsne[:, 0],
            features_tsne[:, 1],
            c=[colors[label] for label in labels],  # Map each label to its color,
            alpha=0.7,
            s=30,
        )
        axes[row, col].set_title(f"t-SNE (perplexity={perplexity})")
        axes[row, col].set_xlabel("t-SNE 1")
        axes[row, col].set_ylabel("t-SNE 2")

        # Create legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
                markersize=8,
                label=class_name,
            )
            for i, class_name in enumerate(class_names)
        ]

        axes[row, col].legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.15, 1),
            fontsize=8,
        )

    pyplot.tight_layout()
    if save_path:
        pyplot.savefig(save_path, bbox_inches="tight", dpi=300)


def scaler_setup(params: dict) -> ScalerType:
    """Create and configure a scaler instance based on the provided parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing scaler configuration with keys:
        - data.scaler_type: Type of scaler to use ('RobustScaler', 'StandardScaler', 'MinMaxScaler')
        - data.scaler_params: Optional parameters for scaler initialization

    Returns
    -------
    ScalerType
        Configured scaler instance ready for data scaling

    """
    SCALER_MAP = {
        "RobustScaler": RobustScaler,
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
    }
    # Create scaler instance safely
    scaler_type = params["data"]["scaler_type"]
    scaler_params = params["data"].get("scaler_params", {})
    # Convert specific parameters from list to tuple if needed
    if "quantile_range" in scaler_params and isinstance(
        scaler_params["quantile_range"], list
    ):
        scaler_params["quantile_range"] = tuple(scaler_params["quantile_range"])
    scaler = SCALER_MAP[scaler_type](**scaler_params)
    return scaler


def normalize_edge_weight(edge_attr, make_edge_weight_method: str | None = "minmax"):
    """Normalize versions of inverse distance squared weights."""
    weights = edge_attr.squeeze()  # Shape: [num_edges]
    if make_edge_weight_method is None:
        return weights

    if make_edge_weight_method == "minmax":
        # Scale to [0, 1] range
        return (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    elif make_edge_weight_method == "standard":
        # Zero mean, unit variance
        return (weights - weights.mean()) / (weights.std() + 1e-8)

    elif make_edge_weight_method == "softmax":
        # Convert to probability distribution over edges
        return torch.softmax(weights, dim=0)

    elif make_edge_weight_method == "log":
        # Apply log transform for better numerical stability
        return torch.log1p(weights)

    return weights


def _calculate_network_density(n_nodes, n_edges):
    """Calculate the density of the undirecetd graph.

    Density measures how many edges are in the graph compared to the maximum possible number of edges between nodes. It ranges from 0 (no edges) to 1 (a complete graph where every node is connected to every other node).

    Density = 2 * E / (N * (N - 1)) for undirected graphs where E is the number of edges and N is the number of nodes.
    """
    density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    return density


def _calculate_average_degree(n_nodes, n_edges):
    """Calculate the average node degree of the undirecetd graph.

    It describes the average connectivity of individual nodes rather than the overall connectedness of the entire network.
    Average Degree = 2 * E / N for undirected graphs
    where E is the number of edges and N is the number of nodes
    """
    avg_degree = (2 * n_edges) / n_nodes if n_nodes > 0 else 0
    return avg_degree
