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
import torch.nn.functional as F
from matplotlib import pyplot
from scipy.spatial import distance_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch_geometric.data import Data


def _generate_coordinates(
    n_samples: int,
    spacing: float,
    depth: float,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    # Calculate the grid size based on n_samples and spacing
    grid_size = int(np.sqrt(n_samples)) + 1
    area_size = grid_size * spacing

    # Define x_range and y_range based on spacing and n_samples
    x_range = (
        (0, area_size)
        if x_range is None
        else (
            x_range[0] + area_size,
            x_range[1] + area_size,
        )  # to avoid overlap with existing data
    )
    y_range = (
        (0, area_size)
        if y_range is None
        else (
            y_range[0] + area_size,
            y_range[1] + area_size,
        )  # to avoid overlap with existing data
    )

    # First create a grid
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)

    # Create all possible grid points
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    # Add some random jitter to make it more realistic (not exactly on grid)
    jitter = rng.uniform(0.3, 0.6, 1) * spacing  # % of spacing for natural randomness
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

    return coordinates, x_range, y_range


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
    print(f"{n_hotspots} hotspots with respective strengths {hotspot_strengths}")
    return hotspots, hotspot_strengths


def _calculate_gold_values(
    n_samples: int,
    coordinates: np.ndarray,
    hotspots: np.ndarray,
    hotspot_strengths: np.ndarray,
    exp_decay_factor: float = 0.005,
    noise_level: float = 0.05,
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

    # Clip values to 0-1 range
    gold_values = np.clip(a=gold_values, a_min=0, a_max=1)
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
    hotspots: np.ndarray,
    hotspot_strengths: np.ndarray,
    gold_values: np.ndarray,
    exp_decay_factor: float,
    noise_level: float,
    rng: np.random.Generator = np.random.default_rng(42),
):
    n_hotspots = hotspots.shape[0]

    # Calculate class counts and needed samples
    class_counts = np.bincount(labels, minlength=n_classes)
    samples_needed = np.maximum(0, min_samples_per_class - class_counts)

    # If no samples needed, return early
    if not np.any(samples_needed > 0):
        return labels, coordinates, gold_values

    # Pre-calculate distance ranges for each class
    if n_classes == 2:
        # Binary case: class 0 (far), class 1 (near)
        class_dist_ranges = [(300, 600), (0, 200)]
    else:
        # Multi-class: evenly spaced distance ranges
        max_distance = 600
        class_ranges = np.linspace(0, max_distance, n_classes + 1)
        class_dist_ranges = [
            (class_ranges[i], class_ranges[i + 1]) for i in range(n_classes)
        ]

    # Generate new samples for each class that needs them
    new_coordinates = []
    new_gold_values = []
    new_labels = []

    for class_idx in range(n_classes):
        if samples_needed[class_idx] <= 0:
            continue

        print(
            f"\nAdding {samples_needed[class_idx]} more samples for class {class_idx}"
        )
        min_dist, max_dist = class_dist_ranges[class_idx]

        # Generate all needed samples at once
        hotspot_indices = rng.integers(0, n_hotspots, size=samples_needed[class_idx])
        angles = rng.uniform(0, 2 * np.pi, size=samples_needed[class_idx])
        distances = rng.uniform(min_dist, max_dist, size=samples_needed[class_idx])

        # Convert to Cartesian coordinates
        dx = distances * np.cos(angles)
        dy = distances * np.sin(angles)
        dz = np.zeros_like(distances)  # Assuming 2D movement for simplicity

        # Calculate new positions
        new_x = hotspots[hotspot_indices, 0] + dx
        new_y = hotspots[hotspot_indices, 1] + dy
        new_z = hotspots[hotspot_indices, 2] + dz

        # Clip to bounds
        new_x = np.clip(new_x, x_range[0], x_range[1])
        new_y = np.clip(new_y, y_range[0], y_range[1])
        new_z = np.clip(new_z, depth, 0)

        # Calculate gold values
        pts = np.column_stack([new_x, new_y, new_z])
        distances = np.min(
            np.sqrt(np.sum((pts[:, np.newaxis] - hotspots) ** 2, axis=2)), axis=1
        )
        nearest_idx = np.argmin(
            np.sqrt(np.sum((pts[:, np.newaxis] - hotspots) ** 2, axis=2)), axis=1
        )
        strengths = hotspot_strengths[nearest_idx]

        gold_vals = strengths * np.exp(-distances * exp_decay_factor)
        gold_vals += rng.normal(0, noise_level, size=len(gold_vals))
        gold_vals = np.clip(gold_vals, 0, 1)

        new_coordinates.append(pts)
        new_gold_values.append(gold_vals)
        new_labels.append(np.full(samples_needed[class_idx], class_idx))

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

    return features


def generate_mineral_data(
    n_samples: int = 500,
    spacing: float = 50,
    depth: float = -500.0,
    n_features: int = 10,
    n_classes: int = 5,
    threshold_binary: float = 0.3,
    min_samples_per_class: int | None = 15,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    n_hotspots: int = 10,
    n_hotspots_random: bool = True,
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
    min_samples_per_class : int, optional
        Minimum number of samples required for each class. If None, no minimum is enforced.
    x_range: tuple[float, float], optional
        Range for x coordinates (Easting). If None, defaults to (0, area_size)
    y_range: tuple[float, float], optional
        Range for y coordinates (Northing). If None, defaults to (0, area_size)
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
    coordinates, x_range, y_range = _generate_coordinates(
        n_samples, spacing, depth, x_range, y_range, rng
    )
    # 2. Create mineralization hotspots and their strengths (mineralization centers)
    hotspots, hotspot_strengths = _create_hotspots(
        depth, x_range, y_range, n_hotspots, n_hotspots_random, rng
    )
    # 3. Calculate gold values based on distance to nearest hotspot
    noise_level = rng.uniform(0.01, 0.1)
    exp_decay_factor = rng.uniform(0.001, 0.01)
    gold_values = _calculate_gold_values(
        n_samples,
        coordinates,
        hotspots,
        hotspot_strengths,
        exp_decay_factor=exp_decay_factor,
        noise_level=noise_level,
        rng=rng,
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
            hotspots,
            hotspot_strengths,
            gold_values,
            exp_decay_factor,
            noise_level,
            rng,
        )
    # 6. Generate features for all samples
    n_total_samples = len(coordinates)
    features = _generate_features(n_total_samples, gold_values, n_features, rng)
    # Print summary of generated data
    print("\nLabel distribution:")
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
    cmap = pyplot.get_cmap("rainbow")
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


def construct_graph(
    coordinates: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    connection_radius: float,
    n_splits: int,
    test_size: float,
    calib_size: float,
    seed: int,
    scaler: ScalerType = RobustScaler(),
    should_split: bool = True,
) -> Data | tuple[Data, list[Data], Data]:
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
    scaled_features = scale_data(data=features, scaler=scaler)
    x = torch.tensor(scaled_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    edge_index, edge_attr = prepare_edge_data(coordinates, connection_radius)

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

    if should_split:
        fold_data, test_data = _split_graph(
            x, y, edge_index, edge_attr, n_splits, test_size, calib_size, seed
        )
        return base_data, fold_data, test_data
    return base_data


def _split_graph(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    n_splits: int,
    test_size: float,
    calib_size: float,
    seed: int,
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


def prepare_edge_data(coordinates: np.ndarray, connection_radius: float = 150):
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
    coordinates: np.ndarray,
    node_indices: np.ndarray,
    connection_radius: float,
    save_path: str,
    labels_map: dict[int, str],
    dataset_idx: int | None = None,
    dataset_tag: str = "train",
    filename="graph.html",
):
    """Export the graph to an interactive HTML file using Plotly."""
    edge_index, _ = prepare_edge_data(coordinates[node_indices], connection_radius)
    src, dst = edge_index
    fig = visualize_graph(
        coordinates[node_indices],
        graph.y[node_indices].numpy(),
        np.array(src),
        np.array(dst),
        labels_map=labels_map,
    )
    if dataset_idx is not None:
        filename = f"{save_path}/{dataset_tag}_graph_{dataset_idx}.html"
    else:
        filename = f"{save_path}/{dataset_tag}_graph.html"
    fig.write_html(filename)
    print(f"Graph exported to {filename}")


def export_all_graphs_to_html(
    fold_data: list[Data],
    test_data: Data,
    coordinates: np.ndarray,
    connection_radius: float,
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
        save_path=save_path,
        labels_map=labels_map,
        dataset_tag="calib",
    )


def connect_graphs_preserve_weights(
    data1, data2, threshold=0.5, k=None, similarity_metric="cosine"
):
    """Connect two graphs while preserving their original edge weights.

    Cross-graph edges are weighted by feature similarity.

    Args:
        data1: PyG Data object (with edge_attr if weighted)
        data2: PyG Data object (with edge_attr if weighted)
        threshold: Min similarity for cross-graph edges (ignored if k is given)
        k: Top-k most similar nodes to connect per node (optional)
        similarity_metric: "cosine" (default), "euclidean", or "dot"

    Returns:
        Combined Data object with edge weights preserved.

    """
    # Check if original graphs have edge weights
    orig_weights1 = (
        data1.edge_attr
        if hasattr(data1, "edge_attr")
        else torch.ones(data1.edge_index.size(1))
    )
    orig_weights2 = (
        data2.edge_attr
        if hasattr(data2, "edge_attr")
        else torch.ones(data2.edge_index.size(1))
    )

    # Compute similarity for cross-graph connections
    if similarity_metric == "cosine":
        sim_matrix = F.cosine_similarity(
            data1.x.unsqueeze(1), data2.x.unsqueeze(0), dim=-1
        )
    elif similarity_metric == "euclidean":
        sim_matrix = -torch.cdist(
            data1.x, data2.x
        )  # Negative distance (higher = more similar)
    elif similarity_metric == "dot":
        sim_matrix = data1.x @ data2.x.T
    else:
        raise ValueError(f"Unknown metric: {similarity_metric}")

    # Get cross-graph connections (src, dst) and weights
    if k is not None:
        topk_sim, topk_idx = torch.topk(sim_matrix, k=k, dim=1)
        src = torch.arange(data1.num_nodes).repeat_interleave(k)
        dst = topk_idx.flatten()
        cross_weights = topk_sim.flatten()
    else:
        mask = sim_matrix > threshold
        src, dst = torch.where(mask)
        cross_weights = sim_matrix[mask]

    # Offset node indices for data2
    offset = data1.num_nodes
    dst += offset

    # Combine node features
    x = torch.cat([data1.x, data2.x], dim=0)
    y = torch.cat([data1.y, data2.y], dim=0)

    # Combine edge indices
    edge_index = torch.cat(
        [data1.edge_index, data2.edge_index + offset, torch.stack([src, dst], dim=0)],
        dim=1,
    )

    # Combine edge weights
    edge_attr = torch.cat([orig_weights1, orig_weights2, cross_weights])

    return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
