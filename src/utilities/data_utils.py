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


def _generate_coordinates(
    radius: float,
    depth: float,
    n_samples: int,
    spacing: float,
    existing_points: np.ndarray | None = None,
    rng: np.random.Generator = np.random.default_rng(42),
) -> np.ndarray:
    # Initialize
    all_xy = np.empty((0, 2)) if existing_points is None else existing_points[:, :2]

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
    n_hotspots: int = 10,
    n_hotspots_random: bool = True,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.ndarray, np.ndarray]:
    # In a mineral exploration context, hotspot strengths represent the maximum concentration or intensity of gold at each "source" location in the simulated area. Not all gold deposits are created equal - some have higher  mineral content than others. Values greater than 1.0 represent "high-grade" hotspots that can potentially yield gold values above the baseline (before applying distance decay). Values below 1.0 represent "lower-grade" hotspots that will produce somewhat weaker signals. The range isn't centered at 1.0 (it's 0.7-1.2) to create a slight positive skew, which is common in real mineral deposits
    if n_hotspots is not None and n_hotspots_random:
        n_hotspots = rng.integers(1, n_hotspots)
    hotspot_strengths = rng.uniform(0.7, 1.2, n_hotspots)
    hotspots = np.zeros((n_hotspots, 3))
    hotspots[:, 0] = rng.uniform(x_range[0], x_range[1], n_hotspots)
    hotspots[:, 1] = rng.uniform(y_range[0], y_range[1], n_hotspots)
    hotspots[:, 2] = rng.uniform(depth, 0, n_hotspots)
    return hotspots, hotspot_strengths


def _calculate_gold_values(
    n_samples: int,
    coordinates: np.ndarray,
    hotspots: np.ndarray,
    hotspot_strengths: np.ndarray,
    rng: np.random.Generator = np.random.default_rng(42),
) -> np.ndarray:
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
    """Convert gold values to categorical labels based on the number of classes."""
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
    n_total_samples: int,
    gold_values,
    n_features,
    rng: np.random.Generator = np.random.default_rng(42),
):
    # Random weights for each feature
    weights = rng.uniform(0.5, 5.0, size=(n_features,))
    # Broadcast gold_values to shape (n_total_samples, n_features)
    gold_matrix = np.tile(gold_values.reshape(-1, 1), (1, n_features))
    # Add random noise
    noise_level = rng.uniform(0.5, 1.5)
    noise = rng.normal(0, noise_level, size=(n_total_samples, n_features))
    # generate features as random linear combinations of gold values plus noise,
    features = gold_matrix * weights + noise
    # Optionally scale features to positive values
    # features = np.maximum(features, 0)
    return features


# def _generate_features_new(
#     n_total_samples: int,
#     gold_values: np.ndarray,
#     n_features: int,
#     labels: np.ndarray,
#     n_classes: int,
#     rng: np.random.Generator = np.random.default_rng(42),
# ) -> np.ndarray:
#     """Generate discriminative features for mineral classification.

#     Improvements:
#     1. Non-linear relationships with gold values
#     2. Class-specific feature correlations
#     3. Feature interactions
#     4. Geological domain knowledge incorporation
#     """
#     # 1. Generate base features with non-linear relationships
#     features = np.zeros((n_total_samples, n_features))

#     # Create different non-linear transformations of gold values
#     gold_sqrt = np.sqrt(gold_values)
#     gold_squared = gold_values ** 2
#     gold_log = np.log1p(gold_values)  # log1p handles zero values

#     # Distribute these across feature columns with varying weights
#     for i in range(n_features):
#         base = rng.choice([gold_values, gold_sqrt, gold_squared, gold_log])
#         weight = rng.uniform(0.5, 2.0)
#         features[:, i] = base * weight

#     # 2. Add class-specific feature correlations
#     class_profiles = np.zeros((n_classes, n_features))
#     for i in range(n_classes):
#         # Generate correlated features for each class
#         cov_matrix = rng.uniform(0.1, 0.9, size=(n_features, n_features))
#         cov_matrix = cov_matrix @ cov_matrix.T  # ensure positive semi-definite
#         class_profiles[i] = rng.multivariate_normal(
#             mean=rng.uniform(-1, 1, n_features),
#             cov=cov_matrix,
#             size=1
#         )

#     # 3. Add feature interactions
#     for i in range(n_classes):
#         mask = labels == i
#         # Add class-specific profile
#         features[mask] += class_profiles[i]

#         # Add interaction terms between pairs of features
#         for j in range(0, n_features-1, 2):
#             interaction = features[mask, j] * features[mask, j+1]
#             features[mask, j] += interaction * rng.uniform(0.1, 0.3)

#     # 4. Add domain-specific geological indicators
#     geological_features = np.zeros((n_total_samples, n_features))

#     # Simulate alteration intensity (increases with gold content)
#     alteration = gold_values + rng.normal(0, 0.1, n_total_samples)

#     # Simulate mineralization style indicators
#     for i in range(n_classes):
#         mask = labels == i
#         # Each class gets distinct mineralization signatures
#         style_1 = rng.normal(i/n_classes, 0.1, size=np.sum(mask))
#         style_2 = rng.normal((n_classes-i)/n_classes, 0.1, size=np.sum(mask))

#         geological_features[mask, 0] = style_1
#         geological_features[mask, 1] = style_2
#         geological_features[mask, 2] = alteration[mask]

#     # Blend geological features into main feature matrix
#     blend_weights = rng.uniform(0.3, 0.7, n_features)
#     features = (1 - blend_weights) * features + blend_weights * geological_features

#     # Add subtle noise to prevent perfect separation
#     noise_scale = rng.uniform(0.05, 0.15, n_features)
#     noise = rng.normal(0, noise_scale, size=(n_total_samples, n_features))
#     features += noise

#     return features

# def _generate_features_new(
#     n_total_samples: int,
#     gold_values: np.ndarray,
#     n_features: int,
#     labels: np.ndarray,
#     n_classes: int,
#     rng: np.random.Generator = np.random.default_rng(42)
# ) -> np.ndarray:
#     """Generate discriminative features for mineral classification with a simpler approach.

#     Key improvements:
#     1. Non-linear transformations of gold values
#     2. Class-specific feature patterns
#     3. Controlled noise addition
#     """
#     features = np.zeros((n_total_samples, n_features))

#     # 1. Create base features using different non-linear transformations
#     transformations = {
#         'linear': gold_values,
#         'squared': gold_values ** 2,
#         'sqrt': np.sqrt(gold_values),
#         'log': np.log1p(gold_values)
#     }

#     # 2. Assign different transformations to feature columns
#     for i in range(n_features):
#         # Select a random transformation for this feature
#         trans_name = rng.choice(list(transformations.keys()))
#         base_feature = transformations[trans_name]

#         # Add class-specific modifications
#         for class_idx in range(n_classes):
#             mask = (labels == class_idx)
#             # Each class gets a distinct multiplier
#             class_multiplier = 0.5 + (class_idx + 1) / n_classes
#             features[mask, i] = base_feature[mask] * class_multiplier

#     # 3. Add small controlled noise to prevent perfect separation
#     noise = rng.normal(0, 0.1, size=features.shape)
#     features += noise

#     # 4. Normalize features
#     features = (features - features.mean(axis=0)) / features.std(axis=0)

#     return features

# def _generate_features_new(
#     n_total_samples: int,
#     gold_values: np.ndarray,
#     n_features: int,
#     rng: np.random.Generator = np.random.default_rng(42),
# ):
#     features = np.zeros((n_total_samples, n_features))

#     for i in range(n_features):
#         # Create different transformations of gold_values for each feature
#         if i % 4 == 0:
#             # Linear with different scaling
#             weight = rng.uniform(1.0, 5.0)
#             features[:, i] = gold_values * weight
#         elif i % 4 == 1:
#             # Squared (emphasizes high values)
#             weight = rng.uniform(0.5, 2.0)
#             features[:, i] = (gold_values ** 2) * weight
#         elif i % 4 == 2:
#             # Square root (emphasizes low values)
#             weight = rng.uniform(0.5, 2.0)
#             features[:, i] = np.sqrt(gold_values) * weight
#         else:
#             # Threshold-based feature
#             threshold = rng.uniform(0.2, 0.8)
#             steepness = rng.uniform(10, 30)
#             features[:, i] = 1 / (1 + np.exp(-steepness * (gold_values - threshold)))

#         # Add much less noise
#         noise = rng.normal(0, 0.05, n_total_samples)  # Much smaller noise
#         features[:, i] += noise

#     return features


def _generate_features_new(
    n_total_samples: int,
    labels,
    n_classes,
    n_features,
    rng: np.random.Generator = np.random.default_rng(42),
):
    # Random prototype vectors for each class
    prototypes = rng.uniform(-2, 2, size=(n_classes, n_features))
    features = np.zeros((n_total_samples, n_features))
    noise_level = 0.2
    # For each sample, copy its class prototype and add Gaussian noise
    features = np.array(
        [
            prototypes[lbl] + rng.normal(0, noise_level, size=n_features)
            for lbl in labels
        ]
    )

    return features


def generate_mineral_data(
    radius: float,
    depth: float = -500,
    n_samples: int = 1000,
    spacing: float = 10,
    existing_points: np.ndarray | None = None,
    n_features: int = 10,
    n_classes: int = 5,
    threshold_binary: float = 0.3,
    min_samples_per_class: int | None = 15,
    n_hotspots: int = 10,
    n_hotspots_random: bool = True,
    seed: int | None = None,
    use_new_feature_generation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    min_samples_per_class : int, optional
        Minimum number of samples required for each class. If None, no minimum is enforced.
    x_range: tuple[float, float], optional
        Range for x coordinates (Easting). If None, defaults to (0, area_size)
    y_range: tuple[float, float], optional
        Range for y coordinates (Northing). If None, defaults to (0, area_size)
    n_hotspots : int
        Number of mineralization hotspots to generate in the area.
    n_hotspots_random : bool
        If True, randomly select the number of hotspots up to n_hotspots.
    seed : int
        Random seed for reproducibility

    Returns
    -------
    coordinates : ndarray
        Array of shape (n_samples, 3) containing x, y, z coordinates
    features : ndarray
        Scaled array of shape (n_samples, n_features) containing mineral features
    labels : ndarray
        Array of shape (n_samples,) containing gold concentration labels (0-n_classes)

    """
    rng = np.random.default_rng(seed)  # Set random seed for reproducibility
    # 1. Generate spatial coordinates with appropriate spacing
    coordinates = _generate_coordinates(
        radius, depth, n_samples, spacing, existing_points, rng
    )
    x_range = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_range = coordinates[:, 1].min(), coordinates[:, 1].max()
    # 2. Create mineralization hotspots and their strengths (mineralization centers)
    hotspots, hotspot_strengths = _create_hotspots(
        depth, x_range, y_range, n_hotspots, n_hotspots_random, rng
    )
    # 3. Calculate gold values based on distance to nearest hotspot
    gold_values = _calculate_gold_values(
        n_samples, coordinates, hotspots, hotspot_strengths, rng=rng
    )
    # 4. Convert to categorical labels
    labels = _assign_labels(gold_values, n_classes, threshold_binary)
    # (Optional) 5. Check if we have the minimum number of samples per class and redistribute if needed
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
    # 6. Generate features for all samples
    n_total_samples = len(coordinates)
    if use_new_feature_generation:
        features = _generate_features_new(
            n_total_samples, labels, n_classes, n_features, rng
        )
        # features = _generate_features_new(n_total_samples, gold_values, n_features, labels, n_classes, rng)
    else:
        features = _generate_features(n_total_samples, gold_values, n_features, rng)
    # Print summary of generated data
    print("Label distribution:")
    for i in range(n_classes):
        count = np.sum(labels == i)
        print(f"Class {i}: {count} points ({count / n_total_samples * 100:.2f}%)")
    return coordinates, features, labels


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
    cmap = get_cmap("rainbow")
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
    fig.update_layout(width=1000, height=600)
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
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"size": 10, "color": color},
                name=f"{labels_map[label]}",
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


def _split_graph(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    n_splits: int | None,
    test_size: float | None,
    calib_size: float | None,
    seed: int | None,
):
    features = x.numpy()
    labels = y.numpy()
    n_nodes = len(labels)
    # First split into train+val and test
    train_val_idx, temp_idx = train_test_split(
        np.arange(n_nodes),
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    # Initialize stratified k-fold on the train+val data
    if n_splits is None:
        raise ValueError("n_splits must be specified when should_split is True.")

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


def construct_graph(
    coordinates: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    connection_radius: float,
    add_self_loops: bool,
    n_splits: int | None = None,
    test_size: float | None = None,
    calib_size: float | None = None,
    seed: int | None = None,
    scaler: ScalerType | None = RobustScaler(),
    should_split: bool = True,
) -> tuple[Data, list[Data], Data] | Data:
    """Create graphs from geospatial data using distance matrix with a held-out test set.

    Args:
        coordinates (array-like): Coordinate points for constructing the graph
        features (array-like): Node features
        labels (array-like): Node labels for stratification
        connection_radius (float): Distance threshold to consider interconnected nodes
        n_splits (int): Number of folds for cross-validation
        test_size (float): Proportion of data to use as test set (e.g., 0.2 for 20%)
        calib_size (float): Proportion of data to use as calibration set (e.g., 0.5 for 50%)
        seed (int): Random seed for reproducibility
        scaler (ScalerType): Scaler to use for feature scaling (default: RobustScaler)
        should_split (bool): Whether to perform train/val/test split (default: True, if False, only returns the base graph without splits)

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

    edge_index, edge_attr = prepare_edge_data(
        coordinates, connection_radius, add_self_loops
    )

    base_data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        coordinates=torch.tensor(coordinates, dtype=torch.float32),
        unscaled_features=torch.tensor(
            features, dtype=torch.float32
        ),  # store for data drift detection
    )

    split_params = (n_splits, test_size, calib_size, seed)
    if should_split and all(split_params) is not None:
        fold_data, test_data = _split_graph(
            x, y, edge_index, edge_attr, n_splits, test_size, calib_size, seed
        )
        return base_data, fold_data, test_data
    return base_data


def prepare_edge_data(
    coordinates: np.ndarray,
    connection_radius: float = 150,
    add_self_loops: bool = False,
):
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
    if not add_self_loops:
        # Exclude self-loops (nodes cannot connect to themselves)
        src, dst = np.where((dist_matrix < connection_radius) & (dist_matrix > 0))
    else:
        # include self-nodes assuming current node features are also important for prediction (node own features along with its neighbors)
        src, dst = np.where(dist_matrix < connection_radius)

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
    ).unsqueeze(1)  # Shape: [num_edges, 1]

    edge_attr = inverse_distances_squared
    return edge_index, edge_attr


def export_graph_to_html(
    graph,
    coordinates: np.ndarray,
    node_indices: np.ndarray | None,
    connection_radius: float,
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
    edge_index, _ = prepare_edge_data(coordinates, connection_radius, add_self_loops)
    src, dst = edge_index
    title = f"Graph with {coordinates.shape[0]} nodes and {edge_index.shape[1]} edges and avg degree {edge_index.shape[1] / coordinates.shape[0]:.2f}"

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
    fig.write_html(filename)
    print(f"Graph exported to {filename}.")


def export_all_graphs_to_html(
    fold_data: list[Data],
    test_data: Data,
    coordinates: np.ndarray,
    connection_radius: float,
    add_self_loops: bool,
    labels_map: dict[int, str],
    save_path: str,
):
    """Export all train, validation, test, and calibration graphs to interactive HTML files.

    Args:
        fold_data (list[Data]): List of PyG Data objects for each fold (train/val splits).
        test_data (Data): PyG Data object for the test (and optionally calibration) set.
        coordinates (np.ndarray): Array of node coordinates.
        connection_radius (float): Distance threshold for connecting nodes in the graph.
        labels_map (dict[int, str]): Mapping from label indices to label names.
        save_path (str): Directory path to save the exported HTML files.

    Returns:
        None

    """
    for i, graph in enumerate(fold_data):
        node_indices = graph.train_mask
        export_graph_to_html(
            graph,
            coordinates,
            node_indices,
            connection_radius=connection_radius,
            add_self_loops=add_self_loops,
            save_path=save_path,
            labels_map=labels_map,
            dataset_idx=i + 1,
            dataset_tag="train",
        )
        node_indices = graph.val_mask
        export_graph_to_html(
            graph,
            coordinates,
            node_indices,
            connection_radius=connection_radius,
            add_self_loops=add_self_loops,
            save_path=save_path,
            labels_map=labels_map,
            dataset_idx=i + 1,
            dataset_tag="val",
        )
    node_indices = test_data.test_mask
    export_graph_to_html(
        test_data,
        coordinates,
        node_indices,
        connection_radius=connection_radius,
        add_self_loops=add_self_loops,
        save_path=save_path,
        labels_map=labels_map,
        dataset_tag="test",
    )
    node_indices = test_data.calib_mask
    export_graph_to_html(
        test_data,
        coordinates,
        node_indices,
        connection_radius=connection_radius,
        add_self_loops=add_self_loops,
        save_path=save_path,
        labels_map=labels_map,
        dataset_tag="calib",
    )


def no_coordinate_overlap(coords1: torch.Tensor, coords2: torch.Tensor):
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
    features_scaled = scale_data(data=features, scaler=scaler)
    silhouette = silhouette_score(features_scaled, labels)

    fig, axes = pyplot.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"t-SNE with Different Perplexity Values \n The Silhouette Coefficient {silhouette:.2f}",
        fontsize=16,
    )

    perplexities = [5, 30, 50, 100]
    cmap = get_cmap("viridis")
    # colors = cmap(np.linspace(0, 1, unique_classes))
    for i, perplexity in enumerate(perplexities):
        row, col = i // 2, i % 2
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=perplexity, max_iter=1000
        )
        features_tsne = tsne.fit_transform(features_scaled)

        axes[row, col].scatter(
            features_tsne[:, 0],
            features_tsne[:, 1],
            c=labels,
            cmap=cmap,
            alpha=0.7,
            s=30,
        )
        axes[row, col].set_title(f"t-SNE (perplexity={perplexity})")
        axes[row, col].set_xlabel("t-SNE 1")
        axes[row, col].set_ylabel("t-SNE 2")
        # Create legend
        legend_elements = []
        for class_idx, class_name in enumerate(class_names):
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=cmap(class_idx / len(class_names)),
                    markersize=8,
                    label=class_name,
                )
            )

        axes[row, col].legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.15, 1),
            fontsize=8,
        )

    pyplot.tight_layout()
    if save_path:
        pyplot.savefig(save_path, bbox_inches="tight", dpi=300)


def scaler_setup(params: dict):
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
