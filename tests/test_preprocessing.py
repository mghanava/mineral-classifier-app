"""Exercise numerical preprocessing and its model and dataset bindings."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utilities.preprocessing import (
    RawDataset,
    TrainingPartition,
    fit_preprocessing,
    load_preprocessing,
    save_preprocessing,
    synthetic_sample_ids,
)

FEATURES = ("copper", "constant")
CLASSES = ("background", "deposit")


@pytest.fixture
def raw() -> RawDataset:
    """Supply three training rows and two held-out rows with stable IDs."""
    return RawDataset(
        features=np.array([[0.125, 5], [1.5, 5], [4.0, 5], [35.0, 6], [-28.0, 4]]),
        coordinates=np.arange(15, dtype=np.float64).reshape(-1, 3),
        feature_names=FEATURES,
        class_names=CLASSES,
        sample_ids=synthetic_sample_ids("base", 3) + synthetic_sample_ids("new", 2),
        labels=np.array([0, 1, 0, 1, 0], dtype=np.int64),
    )


@pytest.fixture
def fold(raw: RawDataset) -> TrainingPartition:
    """Select the fold's three training rows."""
    return TrainingPartition(raw.dataset_sha256, 0, np.array([0, 1, 2], dtype=np.int64))


@pytest.mark.parametrize(
    ("estimator", "options"),
    [
        (RobustScaler, {}),
        (
            RobustScaler,
            {"quantile_range": [10, 90], "unit_variance": True, "copy": False},
        ),
        (RobustScaler, {"with_centering": False}),
        (RobustScaler, {"with_scaling": False}),
        (RobustScaler, {"with_centering": False, "with_scaling": False}),
        (StandardScaler, {}),
        (StandardScaler, {"with_mean": False}),
        (StandardScaler, {"with_std": False}),
        (StandardScaler, {"with_mean": False, "with_std": False, "copy": False}),
        (MinMaxScaler, {"feature_range": [-2, 3], "clip": False, "copy": False}),
        (MinMaxScaler, {"feature_range": [-2, 3], "clip": True}),
        (None, {}),
    ],
)
def test_sklearn_parity_and_json_round_trip(
    raw: RawDataset,
    fold: TrainingPartition,
    tmp_path: Path,
    estimator: type[RobustScaler] | type[StandardScaler] | type[MinMaxScaler] | None,
    options: dict,
) -> None:
    """Match sklearn including constant columns and held-out out-of-range rows."""
    scaler_type = estimator.__name__ if estimator else None
    state = fit_preprocessing(raw, fold, scaler_type, options)
    sklearn_options = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in options.items()
    }
    original = raw.features.copy()
    expected = (
        estimator(**sklearn_options)
        .fit(original[fold.training_indices])
        .transform(original.copy())
        if estimator
        else original
    )
    for key in ("quantile_range", "feature_range"):
        if key in options:
            assert isinstance(options[key], list)

    weights_path = tmp_path / "model.pt"
    weights_path.write_bytes(b"checkpoint fixture")
    sidecar = tmp_path / "preprocessing.json"
    save_preprocessing(sidecar, state, weights_path)
    loaded = load_preprocessing(
        sidecar,
        weights_path,
        feature_names=FEATURES,
        class_names=CLASSES,
        partition=fold,
    )

    assert loaded == state
    transformed = loaded.transform(raw.features, FEATURES)
    model_input = loaded.transform_model_input(raw.features, FEATURES)
    assert transformed.dtype == np.float64
    assert model_input.dtype == np.float32
    np.testing.assert_allclose(transformed, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(model_input, expected.astype(np.float32), rtol=1e-6)
    np.testing.assert_array_equal(raw.features, original)
    assert not np.shares_memory(transformed, raw.features)


def test_fit_ignores_held_out_rows_and_has_independent_fold_state(
    raw: RawDataset, fold: TrainingPartition
) -> None:
    """Change held-out values and fit another fold without mutating the first."""
    state = fit_preprocessing(raw, fold, "StandardScaler")
    expected_center = raw.features[fold.training_indices].mean(axis=0)
    np.testing.assert_array_equal(state.center, expected_center)

    modified = raw.features.copy()
    modified[3:] = 1e20
    outlier_dataset = replace(raw, features=modified)
    same_rows = replace(fold, dataset_sha256=outlier_dataset.dataset_sha256)
    outlier_state = fit_preprocessing(outlier_dataset, same_rows, "StandardScaler")
    assert outlier_state.dataset_sha256 != state.dataset_sha256
    assert outlier_state.partition_sha256 != state.partition_sha256
    assert outlier_state.center == state.center
    assert outlier_state.scale == state.scale

    before = state.transform(raw.features, FEATURES)
    other_fold = replace(
        fold, fold_index=1, training_indices=np.array([1, 3, 4], dtype=np.int64)
    )
    other_state = fit_preprocessing(raw, other_fold, "StandardScaler")
    assert other_state.center != state.center
    assert other_state.partition_sha256 != state.partition_sha256
    np.testing.assert_array_equal(state.transform(raw.features, FEATURES), before)


def test_frozen_prediction_row_is_batch_invariant(
    raw: RawDataset, fold: TrainingPartition, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transform the same row with different neighbors without calling fit."""
    state = fit_preprocessing(raw, fold, "MinMaxScaler", {"copy": False})

    def unexpected_fit(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Transform attempted to fit an estimator.")

    monkeypatch.setattr(MinMaxScaler, "fit", unexpected_fit)
    first = np.array([[2.0, 5.0], [-100.0, 5.0]])
    second = np.array([[1000.0, 12.0], [2.0, 5.0], [50.0, 12.0]])
    np.testing.assert_array_equal(
        state.transform_model_input(first, FEATURES)[0],
        state.transform_model_input(second, FEATURES)[1],
    )
    np.testing.assert_array_equal(first, [[2.0, 5.0], [-100.0, 5.0]])
    np.testing.assert_array_equal(second, [[1000.0, 12.0], [2.0, 5.0], [50.0, 12.0]])


@pytest.mark.parametrize(
    ("features", "names", "message"),
    [
        (np.ones((2, 2)), FEATURES[::-1], "Feature order mismatch"),
        (np.ones((2, 1)), FEATURES, "column count"),
        (np.ones(2), FEATURES, "two-dimensional float64"),
        (np.ones((2, 2), dtype=np.float32), FEATURES, "two-dimensional float64"),
        (np.array([[np.nan, 5.0]]), FEATURES, "finite values"),
    ],
)
def test_transform_rejects_invalid_raw_inputs(
    raw: RawDataset,
    fold: TrainingPartition,
    features: np.ndarray,
    names: tuple[str, ...],
    message: str,
) -> None:
    """Check exact feature order, shape, canonical dtype and finite values."""
    state = fit_preprocessing(raw, fold, None)
    with pytest.raises(ValueError, match=message):
        state.transform(features, names)


def test_float32_overflow_is_rejected_at_model_boundary(
    raw: RawDataset, fold: TrainingPartition
) -> None:
    """Retain finite float64 values until conversion to model inputs."""
    state = fit_preprocessing(raw, fold, None)
    features = np.array([[1e40, 5.0]])
    np.testing.assert_array_equal(state.transform(features, FEATURES), features)
    with pytest.raises(ValueError, match="non-finite float32"):
        state.transform_model_input(features, FEATURES)


def test_dataset_and_partition_contract(
    raw: RawDataset, fold: TrainingPartition
) -> None:
    """Verify identifier stability, immutable copies and partition alignment."""
    features = raw.features.copy()
    features[0, 0] = 1.0000000000001
    recreated = replace(raw, features=features)
    features[0, 0] = 1.0
    assert recreated.features[0, 0] == 1.0000000000001
    assert not recreated.features.flags.writeable
    assert not recreated.coordinates.flags.writeable
    assert not recreated.labels.flags.writeable
    assert recreated.sample_ids[:3] == synthetic_sample_ids("base", 3)
    assert recreated.dataset_sha256 != raw.dataset_sha256
    assert replace(raw).dataset_sha256 == raw.dataset_sha256

    with pytest.raises(ValueError, match="raw dataset provenance"):
        fit_preprocessing(recreated, fold, None)
    out_of_bounds = replace(fold, training_indices=np.array([6], dtype=np.int64))
    with pytest.raises(ValueError, match="outside the raw dataset"):
        fit_preprocessing(raw, out_of_bounds, None)
    with pytest.raises(ValueError, match="unique and non-negative"):
        replace(fold, training_indices=np.array([1, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="int64"):
        replace(fold, training_indices=np.array([1], dtype=np.int32))
    with pytest.raises(ValueError, match="unique names"):
        replace(raw, sample_ids=("duplicate",) * 5)
    with pytest.raises(ValueError, match="known class indices"):
        replace(raw, labels=np.array([0, 1, 2, 1, 0], dtype=np.int64))
    unlabeled = replace(raw, labels=None)
    with pytest.raises(ValueError, match="labeled training dataset"):
        fit_preprocessing(
            unlabeled, replace(fold, dataset_sha256=unlabeled.dataset_sha256), None
        )


@pytest.fixture
def artifact_paths(
    raw: RawDataset, fold: TrainingPartition, tmp_path: Path
) -> tuple[Path, Path]:
    """Save a valid sidecar and matching model bytes."""
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"original checkpoint")
    sidecar = tmp_path / "preprocessing.json"
    save_preprocessing(sidecar, fit_preprocessing(raw, fold, "StandardScaler"), weights)
    return sidecar, weights


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("scaler_type", "unknown"),
        ("scale", [0, 1]),
        ("scale", [1]),
        ("scale", [True, 1]),
        ("scale", [float("nan"), 1]),
        ("center", [0, float("inf")]),
        ("center", ["0", 1]),
        ("fold_index", False),
        ("dataset_sha256", "invalid"),
        ("clip_bounds", [-1, 1]),
    ],
)
def test_rejects_malformed_numeric_state(
    artifact_paths: tuple[Path, Path], key: str, value: object
) -> None:
    """Reject invalid JSON coefficient types and incompatible state layouts."""
    sidecar, weights = artifact_paths
    artifact = json.loads(sidecar.read_text(encoding="utf-8"))
    artifact["state"][key] = value
    sidecar.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed preprocessing artifact"):
        load_preprocessing(
            sidecar, weights, feature_names=FEATURES, class_names=CLASSES
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{", "Cannot read preprocessing artifact"),
        ('{"version": 1, "version": 1}', "Duplicate JSON keys"),
        ("{}", "Unsupported or legacy"),
        ('{"version": true}', "Unsupported or legacy"),
        ('{"version": 2}', "Unsupported or legacy"),
        ('{"version": 1}', "Malformed preprocessing artifact"),
    ],
)
def test_rejects_malformed_or_legacy_sidecar(
    artifact_paths: tuple[Path, Path], content: str, message: str
) -> None:
    """Fail closed on invalid envelopes and unsupported versions."""
    sidecar, weights = artifact_paths
    sidecar.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_preprocessing(
            sidecar, weights, feature_names=FEATURES, class_names=CLASSES
        )


def test_rejects_missing_files_and_bundle_mismatches(
    artifact_paths: tuple[Path, Path], fold: TrainingPartition
) -> None:
    """Require matching weights, expected schema and optional split provenance."""
    sidecar, weights = artifact_paths
    with pytest.raises(ValueError, match="Missing preprocessing artifact"):
        load_preprocessing(
            sidecar.with_name("missing.json"),
            weights,
            feature_names=FEATURES,
            class_names=CLASSES,
        )
    with pytest.raises(ValueError, match="Feature order mismatch"):
        load_preprocessing(
            sidecar, weights, feature_names=FEATURES[::-1], class_names=CLASSES
        )
    with pytest.raises(ValueError, match="Class mapping mismatch"):
        load_preprocessing(
            sidecar, weights, feature_names=FEATURES, class_names=CLASSES[::-1]
        )
    with pytest.raises(ValueError, match="expected provenance"):
        load_preprocessing(
            sidecar,
            weights,
            feature_names=FEATURES,
            class_names=CLASSES,
            partition=replace(fold, fold_index=1),
        )
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Model weights digest mismatch"):
        load_preprocessing(
            sidecar, weights, feature_names=FEATURES, class_names=CLASSES
        )
    weights.unlink()
    with pytest.raises(ValueError, match="Cannot read model weights"):
        load_preprocessing(
            sidecar, weights, feature_names=FEATURES, class_names=CLASSES
        )
