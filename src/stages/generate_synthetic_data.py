"""Generate synthetic mineral exploration data and create graph datasets.

This module provides functions to:
- Generate synthetic mineral exploration data with realistic features
- Scale and preprocess the generated data
- Construct graph datasets for machine learning
- Export interactive 3D visualizations of the graphs
"""

import os

import numpy as np
import torch
import yaml
from scipy.spatial import distance_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch_geometric.data import Data

from src.utilities.utils import visualize_graph

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
    jitter = spacing * rng.uniform(
        0.3, 0.6, 1
    )  # % of spacing for natural randomness
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
        bins = np.linspace(0, 1, n_classes, endpoint=False)[
            1:
        ]  # n_classes-1 bin edges
        labels = np.digitize(gold_values, bins)  # 0 to n_classes-1

    # 5. Check if we have the minimum number of samples per class
    # Add samples for underrepresented classes if needed
    class_counts = [np.sum(labels == i) for i in range(n_classes)]

    for class_idx in range(n_classes):
        samples_needed = max(0, min_samples_per_class - class_counts[class_idx])

        if samples_needed > 0:
            print(
                f"\nAdding {samples_needed} more samples for class {class_idx}"
            )

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
                coordinates = np.vstack(
                    [coordinates, np.array(new_coordinates)]
                )
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
    print(
        f"Generated {n_total_samples} samples with {actual_n_features} features."
    )
    print(
        f"{n_hotspots} hotspots with respective strengths {hotspot_strengths}"
    )
    print("\nLabel distribution:")
    for i in range(n_classes):
        count = np.sum(labels == i)
        print(
            f"Class {i}: {count} points ({count / n_total_samples * 100:.2f}%)"
        )

    return coordinates, features, labels


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
    edge_index, _ = prepare_edge_data(
        coordinates[node_indices], connection_radius
    )
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


def main():
    """Execute the main data generation workflow.

    Reads parameters from params.yaml, generates synthetic mineral exploration data,
    constructs graph datasets, and exports interactive visualizations.
    """
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # Load parameters from YAML file
    N_SAMPLES = params["data"]["n_samples"]
    SPACING = params["data"]["spacing"]
    DEPTH = params["data"]["depth"]
    N_FEATURES = params["data"]["n_features"]
    N_CLASSES = params["data"]["n_classes"]
    THRESHOLD_BINARY = params["data"]["threshold_binary"]
    MIN_SAMPLES_PER_ClASS = params["data"]["min_samples_per_class"]
    connection_radius = params["data"]["connection_radius"]
    N_SPLITS = params["data"]["n_splits"]
    TEST_SIZE = params["data"]["test_size"]
    CALIB_SIZE = params["data"]["calib_size"]
    SEED = params["data"]["seed"]

    # Generate synthetic data
    coordinates, features, labels = generate_mineral_data(
        n_samples=N_SAMPLES,
        spacing=SPACING,
        depth=DEPTH,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        threshold_binary=THRESHOLD_BINARY,
        min_samples_per_class=MIN_SAMPLES_PER_ClASS,
        seed=SEED,
    )
    fold_data, test_data = construct_graph(
        coordinates,
        scale_data(data=features, scaler=StandardScaler()),
        labels,
        connection_radius=connection_radius,
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        calib_size=CALIB_SIZE,
        seed=SEED,
    )
    # Save the generated data
    dataset_path = "results/data"
    os.makedirs(dataset_path, exist_ok=True)
    torch.save(fold_data, os.path.join(dataset_path, "fold_data.pt"))
    torch.save(test_data, os.path.join(dataset_path, "test_data.pt"))
    # export the interactive 3D plots
    for i, graph in enumerate(fold_data):
        node_indices = graph.train_mask
        export_graph_to_html(
            graph,
            coordinates,
            node_indices,
            connection_radius=connection_radius,
            save_path=dataset_path,
            dataset_idx=i + 1,
            dataset_tag="train",
        )
        node_indices = graph.val_mask
        export_graph_to_html(
            graph,
            coordinates,
            node_indices,
            connection_radius=connection_radius,
            save_path=dataset_path,
            dataset_idx=i + 1,
            dataset_tag="val",
        )
    node_indices = test_data.test_mask
    export_graph_to_html(
        test_data,
        coordinates,
        node_indices,
        connection_radius=connection_radius,
        save_path=dataset_path,
        dataset_tag="test",
    )
    node_indices = test_data.calib_mask
    export_graph_to_html(
        test_data,
        coordinates,
        node_indices,
        connection_radius=connection_radius,
        save_path=dataset_path,
        dataset_tag="calib",
    )


if __name__ == "__main__":
    main()
