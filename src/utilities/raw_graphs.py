"""Bind synthetic graph artifacts to canonical observations and split provenance."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data

from src.utilities.general_utils import load_data
from src.utilities.preprocessing import (
    FittedPreprocessing,
    RawDataset,
    synthetic_sample_ids,
)

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


def stamp_partitions(
    raw: RawDataset, folds: Sequence[tuple[Data, Data]], test: Data, calibration: Data
) -> None:
    """Bind stored split graphs to their parent and exact row ordering."""
    for graph in (test, calibration, *(graph for pair in folds for graph in pair)):
        graph.raw_schema_version = RAW_SCHEMA_VERSION
        graph.dataset_sha256 = raw.dataset_sha256
        graph.x = torch.tensor(raw.features[graph.original_node_indices.numpy()])
        graph.split_sha256 = _split_digest(raw, graph)


def _checked_split(
    raw: RawDataset, graph: Data, role: str, fold: int | None
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
) -> tuple[RawDataset, list[tuple[Data, Data]], Data, Data]:
    """Load a base cycle and verify that every fold keeps held-out rows separate."""
    raw = read_raw_graph(load_data(directory / "base_data.pt", "Base"))
    folds = load_data(directory / "fold_data.pt", "Train-Validation Fold")
    test = load_data(directory / "test_data.pt", "Test")
    calibration = load_data(directory / "calib_data.pt", "Calibration")
    test_rows = _checked_split(raw, test, "test", None)
    calibration_rows = _checked_split(raw, calibration, "calib", None)
    if test_rows & calibration_rows:
        raise ValueError("Calibration and test partitions overlap.")
    held_out = test_rows | calibration_rows
    development = set(range(len(raw.sample_ids))) - held_out
    if not isinstance(folds, tuple | list) or not folds:
        raise ValueError("At least one train-validation fold is required.")
    validated_rows = set()
    for fold, pair in enumerate(folds):
        if not isinstance(pair, tuple | list) or len(pair) != 2:
            raise ValueError(
                "Each fold must contain training and validation graphs. Regenerate the split."
            )
        training, validation = pair
        train_rows = _checked_split(raw, training, "train", fold)
        val_rows = _checked_split(raw, validation, "val", fold)
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
    return raw, folds, test, calibration


def model_graph(graph: Data, raw: RawDataset, state: FittedPreprocessing) -> Data:
    """Copy a verified graph and replace only its model input matrix."""
    rows = getattr(graph, "original_node_indices", None)
    features = raw.features if rows is None else raw.features[rows.numpy()]
    prepared = graph.clone()
    prepared.x = torch.from_numpy(
        state.transform_model_input(features, raw.feature_names)
    )
    return prepared
