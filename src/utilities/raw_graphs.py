"""Bind synthetic graph artifacts to canonical observations and split provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data

from src.utilities.data_utils import partition_indices
from src.utilities.general_utils import load_data
from src.utilities.graph_protocol import graph_digest, verify_graph
from src.utilities.preprocessing import (
    FittedPreprocessing,
    RawDataset,
    synthetic_sample_ids,
)
from src.utilities.spatial_graph import build_spatial_graph

RAW_SCHEMA_VERSION = 1


def synthetic_schema(base_params: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve the ordered feature identifiers and class mapping for synthetic data."""
    features = tuple(f"Feature_{i}" for i in range(base_params["n_features"]))
    classes = base_params["class_names"] or [
        f"Class {i}" for i in range(base_params["n_classes"])
    ]
    if len(classes) != base_params["n_classes"]:
        raise ValueError(
            "Synthetic class mapping must match the configured class count."
        )
    return features, tuple(classes)


def synthetic_raw(
    features: NDArray,
    coordinates: NDArray,
    labels: NDArray,
    base_params: dict,
    *,
    parent_sha256: str | None = None,
) -> RawDataset:
    """Assign reproducible batch IDs to newly generated synthetic rows."""
    feature_names, class_names = synthetic_schema(base_params)
    if not np.issubdtype(np.asarray(labels).dtype, np.integer):
        raise ValueError("Synthetic labels must contain integer class indices.")
    columns = (
        np.asarray(features, dtype=np.float64),
        np.asarray(coordinates, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )
    digest = hashlib.sha256(b"mineral-synthetic-batch-v1\n")
    digest.update((parent_sha256 or "bootstrap").encode("ascii"))
    for values, dtype in zip(columns, ("<f8", "<f8", "<i8"), strict=True):
        digest.update(json.dumps(values.shape).encode("ascii"))
        digest.update(values.astype(dtype, copy=False).tobytes())
    return RawDataset(
        features=columns[0],
        coordinates=columns[1],
        labels=columns[2],
        feature_names=feature_names,
        class_names=class_names,
        sample_ids=synthetic_sample_ids(digest.hexdigest(), len(features)),
    )


def stamp_raw_graph(
    graph: Data, raw: RawDataset, *, source_parent: str | None = None
) -> None:
    """Attach canonical observations and versioned metadata to a graph for storage."""
    graph.x = torch.tensor(raw.features)
    graph.unscaled_features = graph.x
    graph.coordinates = torch.tensor(raw.coordinates)
    graph.raw_schema_version = RAW_SCHEMA_VERSION
    graph.dataset_sha256 = raw.dataset_sha256
    graph.feature_names = raw.feature_names
    graph.class_names = raw.class_names
    graph.sample_ids = raw.sample_ids
    if source_parent is not None:
        graph.source_parent_sha256 = source_parent


def read_raw_graph(graph: Data) -> RawDataset:
    """Recover a verified raw dataset from a trusted locally produced graph."""
    version = getattr(graph, "raw_schema_version", None)
    if (
        not isinstance(graph, Data)
        or type(version) is not int
        or version != RAW_SCHEMA_VERSION
    ):
        raise ValueError(
            "Legacy or incompatible raw graph. Regenerate synthetic data for M1.3."
        )
    try:
        raw = RawDataset(
            features=graph.x.numpy(),
            coordinates=graph.coordinates.numpy(),
            labels=graph.y.numpy(),
            feature_names=graph.feature_names,
            class_names=graph.class_names,
            sample_ids=graph.sample_ids,
        )
        if graph.dataset_sha256 != raw.dataset_sha256:
            raise ValueError(
                "Raw dataset digest does not match stored rows and schema."
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid canonical raw graph. Regenerate synthetic data: {exc}"
        ) from exc
    return raw


def _split_digest(raw: RawDataset, graph: Data) -> str:
    identity = (
        raw.dataset_sha256,
        graph.split_type,
        getattr(graph, "fold", None),
        graph.original_node_indices.tolist(),
    )
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def build_raw_graph(
    raw: RawDataset,
    protocol: dict,
    role: str,
    *,
    indices: NDArray[np.int64] | None = None,
    fold: int | None = None,
    source_parent: str | None = None,
) -> Data:
    """Build and stamp one verified population without accessing other coordinates."""
    if indices is not None:
        if (
            indices.dtype != np.int64
            or indices.ndim != 1
            or not len(indices)
            or indices.min() < 0
            or indices.max() >= len(raw.sample_ids)
            or np.unique(indices).size != len(indices)
        ):
            raise ValueError(
                "Partition needs unique, non-empty, in-bounds int64 indices."
            )
        coordinates = raw.coordinates[indices]
        sample_ids = tuple(raw.sample_ids[index] for index in indices)
        labels = raw.labels[indices]
    else:
        coordinates, sample_ids, labels = raw.coordinates, raw.sample_ids, raw.labels
    graph = build_spatial_graph(
        coordinates,
        sample_ids,
        **protocol["connectivity"],
        affinity_scale=protocol["affinity"]["scale"],
        add_self_loops=protocol["self_loops"]["enabled"],
        loop_affinity=protocol["self_loops"]["stored_affinity"],
    )
    graph.y = torch.tensor(labels)
    graph.split_type = role
    if indices is None:
        stamp_raw_graph(graph, raw, source_parent=source_parent)
    else:
        graph.original_node_indices = torch.tensor(indices)
        graph.fold = fold
        graph.raw_schema_version = RAW_SCHEMA_VERSION
        graph.dataset_sha256 = raw.dataset_sha256
        graph.x = torch.tensor(raw.features[indices])
        graph.split_sha256 = _split_digest(raw, graph)
    graph.graph_protocol = protocol
    graph.graph_sha256 = graph_digest(graph)
    return graph


def _verify_membership(
    row_count: int,
    folds: list[tuple[set[int], set[int]]],
    test_rows: set[int],
    calibration_rows: set[int],
) -> set[int]:
    if test_rows & calibration_rows:
        raise ValueError("Calibration and test partitions overlap.")
    development = set(range(row_count)) - (test_rows | calibration_rows)
    validated_rows: set[int] = set()
    for train_rows, val_rows in folds:
        if (
            train_rows & val_rows
            or train_rows | val_rows != development
            or validated_rows & val_rows
        ):
            raise ValueError(
                "Fold partitions overlap or do not match the development pool."
            )
        validated_rows.update(val_rows)
    if validated_rows != development:
        raise ValueError("Validation folds do not cover the development pool.")
    return development


def build_cycle_graphs(
    raw: RawDataset, protocol: dict, base_params: dict
) -> tuple[Data, list[tuple[Data, Data]], Data, Data]:
    """Establish membership first and build each population independently."""
    folds, test, calibration = partition_indices(
        raw.labels,
        base_params["n_splits"],
        base_params.get("test_size"),
        base_params.get("calib_size"),
        base_params.get("seed", 42),
    )
    _verify_membership(
        len(raw.sample_ids),
        [(set(train), set(val)) for train, val in folds],
        set(test),
        set(calibration),
    )
    return (
        build_raw_graph(raw, protocol, "overview"),
        [
            (
                build_raw_graph(raw, protocol, "train", indices=train, fold=index),
                build_raw_graph(raw, protocol, "val", indices=validation, fold=index),
            )
            for index, (train, validation) in enumerate(folds)
        ],
        build_raw_graph(raw, protocol, "test", indices=test),
        build_raw_graph(raw, protocol, "calib", indices=calibration),
    )


def _checked_split(
    raw: RawDataset, graph: Data, role: str, fold: int | None, protocol: dict
) -> set[int]:
    version = getattr(graph, "raw_schema_version", None)
    if (
        not isinstance(graph, Data)
        or type(version) is not int
        or version != RAW_SCHEMA_VERSION
    ):
        raise ValueError(
            "Legacy or incompatible graph partition. Regenerate synthetic data for M1.3."
        )
    if (
        getattr(graph, "dataset_sha256", None) != raw.dataset_sha256
        or getattr(graph, "split_type", None) != role
        or getattr(graph, "fold", None) != fold
    ):
        raise ValueError(
            "Graph partition provenance does not match the parent dataset or split."
        )
    verify_graph(graph, protocol, role)
    indices = getattr(graph, "original_node_indices", None)
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int64
        or indices.ndim != 1
        or not indices.numel()
        or indices.min() < 0
        or indices.max() >= len(raw.sample_ids)
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError(
            "Graph partition requires unique, in-bounds int64 parent row indices."
        )
    rows = indices.numpy()
    if not np.array_equal(
        graph.coordinates.numpy(), raw.coordinates[rows]
    ) or graph.sample_ids != tuple(raw.sample_ids[index] for index in rows):
        raise ValueError(
            "Graph partition row order, coordinates or sample IDs do not match the raw parent."
        )
    if (
        getattr(graph, "split_sha256", None) != _split_digest(raw, graph)
        or graph.y is None
        or graph.y.dtype != torch.int64
        or not np.array_equal(graph.y.numpy(), raw.labels[rows])
        or graph.x is None
        or graph.x.dtype != torch.float64
        or not np.array_equal(graph.x.numpy(), raw.features[rows])
    ):
        raise ValueError(
            "Graph partition row order, features or labels do not match the raw parent."
        )
    return set(indices.tolist())


def load_cycle_data(
    directory: Path,
    protocol: dict,
) -> tuple[RawDataset, list[tuple[Data, Data]], Data, Data]:
    """Load a base cycle and verify that every fold keeps held-out rows separate."""
    base = load_data(directory / "base_data.pt", "Base")
    raw = read_raw_graph(base)
    verify_graph(base, protocol, "overview")
    folds = load_data(directory / "fold_data.pt", "Train-Validation Fold")
    test = load_data(directory / "test_data.pt", "Test")
    calibration = load_data(directory / "calib_data.pt", "Calibration")
    test_rows = _checked_split(raw, test, "test", None, protocol)
    calibration_rows = _checked_split(raw, calibration, "calib", None, protocol)
    if not isinstance(folds, tuple | list) or not folds:
        raise ValueError("At least one train-validation fold is required.")
    verified = []
    for fold, pair in enumerate(folds):
        if not isinstance(pair, tuple | list) or len(pair) != 2:
            raise ValueError(
                "Each fold must contain training and validation graphs. Regenerate the split."
            )
        training, validation = pair
        verified.append(
            (
                _checked_split(raw, training, "train", fold, protocol),
                _checked_split(raw, validation, "val", fold, protocol),
            )
        )
    _verify_membership(len(raw.sample_ids), verified, test_rows, calibration_rows)
    return raw, folds, test, calibration


def build_development_graph(
    raw: RawDataset,
    folds: list[tuple[Data, Data]],
    test: Data,
    calibration: Data,
    protocol: dict,
) -> Data:
    """Reconstruct the development pool in parent order from verified graphs."""
    development = _verify_membership(
        len(raw.sample_ids),
        [
            (
                set(training.original_node_indices.tolist()),
                set(validation.original_node_indices.tolist()),
            )
            for training, validation in folds
        ],
        set(test.original_node_indices.tolist()),
        set(calibration.original_node_indices.tolist()),
    )
    return build_raw_graph(
        raw,
        protocol,
        "development",
        indices=np.array(sorted(development), dtype=np.int64),
    )


def model_graph(
    graph: Data, raw: RawDataset, state: FittedPreprocessing, settings: dict
) -> Data:
    """Copy verified topology and prepare features and explicit model loop weights."""
    rows = getattr(graph, "original_node_indices", None)
    features = raw.features if rows is None else raw.features[rows.numpy()]
    prepared = graph.clone()
    prepared.x = torch.from_numpy(
        state.transform_model_input(features, raw.feature_names)
    )
    loop_affinity = settings["effective_loop_affinity"]
    if loop_affinity is not None:
        src, dst = prepared.edge_index
        prepared.edge_weight[src == dst] = loop_affinity
        prepared.edge_attr = prepared.edge_weight[:, None]
    return prepared
