import os

import numpy as np
import torch
import yaml
from scipy.spatial import distance_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data


def load_data(
    origin: tuple,
    x_spacing: float,
    y_spacing: float,
    z_spacing: float,
    x_length: float,
    y_length: float,
    z_depth: float,
    n_features: int,
    n_classes: int,
    seed: int,
):
    """
    Generate synthetic drill hole data on a regular grid.

    Parameters:
    -----------
    origin : tuple
        (easting, northing, depth) of the starting point
    x_spacing : float
        Spacing between points in easting direction (meters)
    y_spacing : float
        Spacing between points in northing direction (meters)
    z_spacing : float
        Spacing between points in depth direction (meters)
    x_length : float
        Total length in easting direction (meters)
    y_length : float
        Total length in northing direction (meters)
    z_depth : float
        Total depth to sample (meters)
    n_features : int
        Number of features to generate for each point
    n_classes: int
        Number of classes
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    coordinates : np.ndarray
        Array of shape (n_points, 3) containing (easting, northing, depth)
    features : np.ndarray
        Array of shape (n_points, n_features) with values in [0, 1]
    labels : np.ndarray
        Array of shape (n_points,) with integer labels [0, 1, 2, 3, 4]
    """
    np.random.seed(seed)

    # Calculate number of points in each direction
    nx = int(x_length / x_spacing) + 1
    ny = int(y_length / y_spacing) + 1
    nz = int(z_depth / z_spacing) + 1

    # Generate coordinate grids
    x = np.linspace(origin[0], origin[0] + x_length, nx)
    y = np.linspace(origin[1], origin[1] + y_length, ny)
    z = np.linspace(origin[2], origin[2] - z_depth, nz)  # Negative for depth

    # Create 3D grid
    X, Y, Z = np.meshgrid(x, y, z)

    # Reshape to (n_points, 3)
    coordinates = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    n_points = len(coordinates)

    # Generate synthetic features
    features = np.zeros((n_points, n_features))

    # Generate each feature with some spatial correlation
    for i in range(n_features):
        # Create base feature with spatial correlation
        feature = (
            np.sin(coordinates[:, 0] / (50 * (i + 1)))
            + np.cos(coordinates[:, 1] / (50 * (i + 1)))
            + np.exp(-coordinates[:, 2] / (30 * (i + 1)))
        )
        # Add some random noise
        feature += np.random.randn(n_points) * 0.1
        features[:, i] = feature

    # Scale features to [0, 1]
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Generate labels based on feature combinations and spatial patterns
    label_prob = (
        0.3 * np.sin(coordinates[:, 0] / 50) * np.cos(coordinates[:, 1] / 50)
        + 0.3 * np.exp(-coordinates[:, 2] / 50)
        + 0.4 * np.mean(features, axis=1)
    )
    # Scale to [0, 1]
    label_prob = (label_prob - label_prob.min()) / (label_prob.max() - label_prob.min())
    # Convert to 5 classes [0, 1, 2, 3, 4]
    labels = np.digitize(
        label_prob,
        bins=[(1 + i) * (1 / n_classes) for i in range(n_classes - 1)],
    )

    # Print summary
    print("Generated data shape:")
    print(f"Coordinates: {coordinates.shape}")
    print(f"Features: {features.shape}")
    print(f"Labels: {labels.shape}")
    print("\nFeature statistics:")
    print(f"Min values: {features.min(axis=0)}")
    print(f"Max values: {features.max(axis=0)}")
    print("\nLabel distribution:")
    for i in range(n_classes):
        count = np.sum(labels == i)
        print(f"Class {i}: {count} points ({count/n_points*100:.1f}%)")

    return coordinates, features, labels


def construct_graph(
    coordinates,
    features,
    labels,
    d_threshold: float,
    edge_dim: int,
    n_splits: int,
    test_size: float,
    calib_size: int,
    seed: int,
):
    """
    Create graphs from geospatial data using distance matrix with a held-out
    test set, an optional held-out calibration set, and stratified k-fold splits
    for the remaining data

    Parameters:
    -----------
    coordinates (array-like): Coordinate points for constructing the graph
    features (array-like): Node features
    labels (array-like): Node labels for stratification
    n_splits (int): Number of folds for cross-validation
    test_size (float): Proportion of data to use as test set (e.g., 0.2 for 20%)
    calib_size (float): Proportion of data to use as calibration set
        (e.g., 0.5 for 50%)
    d_threshold (float): Distance threshold to consider interconnected nodes
    seed (int): Random seed for reproducibility

    Returns:
    --------
    tuple containing:
        - list of Data: PyG Data objects for each fold (train/val splits)
        - Data: Single PyG Data object for test (and an optional calibration) set
    """
    # Convert features and labels to torch tensors
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    edge_index, edge_attr = prepare_edge_data(coordinates, d_threshold, edge_dim)

    n_nodes = len(labels)
    # First split into train+val and test
    train_val_idx, temp_idx = train_test_split(
        np.arange(n_nodes), test_size=test_size, stratify=labels, random_state=seed
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
            x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, test_mask=test_mask
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


def prepare_edge_data(coordinates, d_threshold: float, edge_dim: int):
    """Prepares edge connectivity and attributes for a graph neural network from coordinate data.
    This function computes pairwise distances between points and creates edge connections
    based on a distance threshold, making the resulting graph undirected. It also
    generates edge attributes including both raw distances and inverse distances.
    Args:
        coordinates (numpy.ndarray): Array of point coordinates with shape [num_nodes, num_dimensions]
        d_threshold (float, optional): Maximum distance threshold for creating edges. Defaults to 5.0.
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
    src, dst = np.where((dist_matrix < d_threshold) & (dist_matrix > 0))
    # # or alternatively, include self-nodes assuming current node features are also
    # # important for prediction (node own features along with its neighbors)
    # src, dst = np.where(dist_matrix < d_threshold)
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
    raw_distances = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(
        1
    )  # Shape: [num_edges, 1]
    edge_attr = raw_distances
    if edge_dim == 2:
        inverse_distances_squared = (
            1.0 / (edge_distances + 1e-6) ** 2
        )  # Avoid division by zero that matters for self-nodes
        inverse_distances_squared = torch.tensor(
            inverse_distances_squared, dtype=torch.float
        ).unsqueeze(1)  # Shape: [num_edges, 1]
        edge_attr = torch.cat(
            [inverse_distances_squared, raw_distances], dim=1
        )  # Shape: [num_edges, 2]
    return edge_index, edge_attr


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # Load parameters from YAML file
    ORIGIN = tuple(params["data"]["origin"])
    X_SPACING = params["data"]["x_spacing"]
    Y_SPACING = params["data"]["y_spacing"]
    Z_SPACING = params["data"]["z_spacing"]
    X_LENGTH = params["data"]["x_length"]
    Y_LENGTH = params["data"]["y_length"]
    Z_DEPTH = params["data"]["z_depth"]
    N_FEATURES = params["data"]["n_features"]
    N_CLASSES = params["data"]["n_classes"]
    D_THRESHOLD = params["data"]["d_threshold"]
    EDGE_DIM = params["data"]["edge_dim"]
    N_SPLITS = params["data"]["n_splits"]
    TEST_SIZE = params["data"]["test_size"]
    CALIB_SIZE = params["data"]["calib_size"]
    SEED = params["data"]["seed"]

    # Generate synthetic data
    coordinates, features, labels = load_data(
        origin=ORIGIN,
        x_spacing=X_SPACING,
        y_spacing=Y_SPACING,
        z_spacing=Z_SPACING,
        x_length=X_LENGTH,
        y_length=Y_LENGTH,
        z_depth=Z_DEPTH,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        seed=SEED,
    )
    fold_data, test_data = construct_graph(
        coordinates,
        features,
        labels,
        d_threshold=D_THRESHOLD,
        edge_dim=EDGE_DIM,
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


if __name__ == "__main__":
    main()
