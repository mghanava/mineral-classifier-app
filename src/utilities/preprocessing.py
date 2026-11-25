"""Fit fold-local feature scalers and persist their numerical transform state."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ARTIFACT_VERSION = 2
_SCALERS = {
    "RobustScaler": RobustScaler,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, list | tuple) or not values:
        raise ValueError(f"{name} must be a non-empty ordered sequence.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique names.")
    return tuple(values)


def _check_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _check_fold(fold_index: int | None) -> None:
    if fold_index is not None and (type(fold_index) is not int or fold_index < 0):
        raise ValueError(
            "Fold index must be zero-based or null for the complete development pool."
        )


def _matrix(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float64
        or values.ndim != 2
        or not all(values.shape)
    ):
        raise ValueError(f"{name} must be a non-empty two-dimensional float64 array.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _int_vector(values: NDArray[np.int64], name: str) -> NDArray[np.int64]:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.int64
        or values.ndim != 1
        or values.size == 0
    ):
        raise ValueError(f"{name} must be a non-empty one-dimensional int64 array.")
    return values


def _immutable_array(values: NDArray) -> NDArray:
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _update_array_digest(digest: Any, values: NDArray, dtype: str) -> None:
    # Include shape and normalize byte order for stable IDs across platforms.
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.astype(dtype, copy=False).tobytes())


def synthetic_sample_ids(batch_id: str, row_count: int) -> tuple[str, ...]:
    """Create IDs in generation order for one immutable synthetic batch.

    Args:
        batch_id: Unique generation identifier persisted with the batch.
        row_count: Number of rows in the batch.

    Returns:
        A tuple of stable IDs to retain when appending or selecting rows.

    Raises:
        ValueError: The batch ID or row count is invalid.

    Example:
        ```python
        ids = synthetic_sample_ids("bootstrap-run-12", 2)
        ```

    """
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("Synthetic batch ID must be a non-empty string.")
    if type(row_count) is not int or row_count < 1:
        raise ValueError("Synthetic batch row count must be a positive integer.")
    return tuple(f"{batch_id}:{index}" for index in range(row_count))


@dataclass(frozen=True, eq=False)
class RawDataset:
    """Represent aligned raw observations without graph or model state.

    The digest covers row order and values. Copies backed by immutable bytes
    keep later mutations of the caller's arrays from invalidating provenance.

    Args:
        features: Finite float64 matrix of raw feature values.
        coordinates: Finite float64 matrix of three spatial coordinates per row.
        feature_names: Ordered names for feature columns.
        class_names: Class names in integer-label order.
        sample_ids: Unique persistent identifiers aligned with the rows.
        labels: Aligned int64 class labels, or None for unlabeled observations.
            Defaults to None.

    Attributes:
        features: Canonical, read-only raw features.
        coordinates: Canonical, read-only coordinates.
        feature_names: Ordered, unique feature names.
        class_names: Ordered, unique class names.
        sample_ids: Persistent identifiers in dataset order.
        labels: Canonical, read-only labels when supplied.
        dataset_sha256: Digest of the canonical observations and schema.

    Example:
        ```python
        raw = RawDataset(
            np.array([[0.8]], dtype=np.float64),
            np.array([[1.0, 2.0, -3.0]], dtype=np.float64),
            ("copper",),
            ("background", "deposit"),
            synthetic_sample_ids("bootstrap-run-12", 1),
            np.array([0], dtype=np.int64),
        )
        ```

    """

    features: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    labels: NDArray[np.int64] | None = None
    dataset_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate alignment and bind the canonical rows to their digest."""
        feature_names = _names(self.feature_names, "Feature names")
        class_names = _names(self.class_names, "Class names")
        sample_ids = _names(self.sample_ids, "Sample IDs")
        features = _matrix(self.features, "Raw features")
        coordinates = _matrix(self.coordinates, "Raw coordinates")
        row_count, feature_count = features.shape
        if feature_count != len(feature_names):
            raise ValueError("Raw feature columns do not match the feature names.")
        if coordinates.shape != (row_count, 3) or len(sample_ids) != row_count:
            raise ValueError(
                "Coordinates and sample IDs must align with raw feature rows."
            )
        if self.labels is not None:
            labels = _int_vector(self.labels, "Labels")
            if labels.size != row_count or np.any(
                (labels < 0) | (labels >= len(class_names))
            ):
                raise ValueError(
                    "Labels must align with the rows and use known class indices."
                )
            object.__setattr__(self, "labels", _immutable_array(labels))

        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "class_names", class_names)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "features", _immutable_array(features))
        object.__setattr__(self, "coordinates", _immutable_array(coordinates))

        metadata = (feature_names, class_names, sample_ids, self.labels is not None)
        digest = hashlib.sha256(b"mineral-raw-dataset-v1\n")
        digest.update(json.dumps(metadata, separators=(",", ":")).encode("ascii"))
        _update_array_digest(digest, self.features, "<f8")
        _update_array_digest(digest, self.coordinates, "<f8")
        if self.labels is not None:
            _update_array_digest(digest, self.labels, "<i8")
        object.__setattr__(self, "dataset_sha256", digest.hexdigest())


@dataclass(frozen=True, eq=False)
class TrainingPartition:
    """Bind ordered training rows to their exact raw parent dataset.

    The fitted-state artifact records the partition digest. The partition
    indices themselves remain in the dataset artifacts.

    Args:
        dataset_sha256: Digest of the raw parent dataset.
        fold_index: Zero-based fold index or None for the development pool.
        training_indices: Unique int64 row indices in training order.

    Attributes:
        dataset_sha256: Digest of the raw parent dataset.
        fold_index: Zero-based fold index or None for the development pool.
        training_indices: Read-only training row indices.
        partition_sha256: Digest of the parent, fold and ordered training indices.

    Example:
        ```python
        fold = TrainingPartition(
            raw.dataset_sha256, 0, np.array([0], dtype=np.int64)
        )
        ```

    """

    dataset_sha256: str
    fold_index: int | None
    training_indices: NDArray[np.int64]
    partition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate indices and bind them to the parent and fold."""
        _check_digest(self.dataset_sha256, "Dataset provenance")
        _check_fold(self.fold_index)
        indices = _int_vector(self.training_indices, "Training indices")
        if indices.min() < 0 or np.unique(indices).size != indices.size:
            raise ValueError("Training indices must be unique and non-negative.")
        object.__setattr__(self, "training_indices", _immutable_array(indices))

        domain = (
            b"mineral-development-pool-v1\n"
            if self.fold_index is None
            else b"mineral-training-partition-v1\n"
        )
        digest = hashlib.sha256(domain)
        digest.update(f"{self.dataset_sha256}:{self.fold_index}:".encode("ascii"))
        _update_array_digest(digest, self.training_indices, "<i8")
        object.__setattr__(self, "partition_sha256", digest.hexdigest())


def _numeric_tuple(values: Sequence[float], count: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, list | tuple) or len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} values.")
    try:
        valid = all(
            type(value) in (int, float) and np.isfinite(float(value))
            for value in values
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must contain finite numbers.")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class FittedPreprocessing:
    """Describe a training population's immutable numerical feature transform.

    Fit estimators through `fit_preprocessing`. The persisted numeric state
    defines transform behavior independently of estimator deserialization.

    Args:
        scaler_type: Supported scikit-learn scaler name, or None for identity.
        feature_names: Ordered feature names used for fitting.
        class_names: Class names in integer-label order.
        dataset_sha256: Digest of the raw parent dataset.
        partition_sha256: Digest of the exact training partition.
        fold_index: Zero-based fold index or None for the development pool.
        center: Optional subtraction vector for Standard or Robust scaling.
            Defaults to None.
        scale: Optional divisor for Standard or Robust scaling, or required
            multiplier for MinMax scaling. Defaults to None.
        offset: Required addition vector for MinMax scaling. Defaults to None.
        clip_bounds: MinMax clipping bounds when enabled. Defaults to None.

    Attributes:
        scaler_type: Fitted transform kind.
        feature_names: Expected feature order.
        class_names: Expected class mapping.
        dataset_sha256: Bound dataset provenance.
        partition_sha256: Bound training partition provenance.
        fold_index: Zero-based fitted fold or None for the development pool.
        center: Fitted subtraction vector, when used.
        scale: Fitted divisor or multiplier, when used.
        offset: Fitted MinMax offset, when used.
        clip_bounds: Fitted output clipping bounds, when used.

    Example:
        ```python
        state = fit_preprocessing(raw, fold, "StandardScaler")
        ```

    """

    scaler_type: str | None
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    dataset_sha256: str
    partition_sha256: str
    fold_index: int | None
    center: tuple[float, ...] | None = None
    scale: tuple[float, ...] | None = None
    offset: tuple[float, ...] | None = None
    clip_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the fitted numerical state."""
        if self.scaler_type is not None and (
            not isinstance(self.scaler_type, str) or self.scaler_type not in _SCALERS
        ):
            raise ValueError(f"Unsupported scaler type {self.scaler_type!r}.")
        feature_names = _names(self.feature_names, "Feature names")
        class_names = _names(self.class_names, "Class names")
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "class_names", class_names)
        _check_digest(self.dataset_sha256, "Dataset provenance")
        _check_digest(self.partition_sha256, "Training partition provenance")
        _check_fold(self.fold_index)

        for name in ("center", "scale", "offset", "clip_bounds"):
            values = getattr(self, name)
            if values is not None:
                count = 2 if name == "clip_bounds" else len(feature_names)
                object.__setattr__(self, name, _numeric_tuple(values, count, name))

        if self.scale is not None and min(self.scale) <= 0:
            raise ValueError("Fitted scale values must be positive.")
        if self.clip_bounds is not None and self.clip_bounds[0] >= self.clip_bounds[1]:
            raise ValueError("MinMax clipping bounds must be increasing.")
        if self.scaler_type == "MinMaxScaler":
            if self.center is not None or self.scale is None or self.offset is None:
                raise ValueError(
                    "MinMax fitted state requires scale and offset only, with optional clipping."
                )
        elif self.offset is not None or self.clip_bounds is not None:
            raise ValueError(
                "Offset and clipping are only valid for MinMax fitted state."
            )
        elif self.scaler_type is None and (
            self.center is not None or self.scale is not None
        ):
            raise ValueError("Disabled scaling cannot contain fitted coefficients.")

    def transform(
        self, features: NDArray[np.float64], feature_names: Sequence[str]
    ) -> NDArray[np.float64]:
        """Apply the frozen transform in float64 without modifying raw inputs.

        Args:
            features: Canonical raw features in float64 precision.
            feature_names: Ordered names for the input feature columns.

        Returns:
            A new float64 matrix in the fitted feature space.

        Raises:
            ValueError: The feature schema mismatches or values are invalid.

        Example:
            ```python
            transformed = state.transform(
                raw.features, raw.feature_names
            )
            ```

        """
        if _names(feature_names, "Input feature names") != self.feature_names:
            raise ValueError(
                f"Feature order mismatch. Expected {self.feature_names!r}."
            )
        matrix = _matrix(features, "Raw features")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                "Raw feature column count does not match fitted preprocessing."
            )

        result = matrix.copy()
        # Preserve the order of the operations used by each estimator.
        with np.errstate(over="ignore", invalid="ignore"):
            if self.scaler_type == "MinMaxScaler":
                result *= self.scale
                result += self.offset
                if self.clip_bounds is not None:
                    np.clip(result, *self.clip_bounds, out=result)
            else:
                if self.center is not None:
                    result -= self.center
                if self.scale is not None:
                    result /= self.scale
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced non-finite transformed features.")
        return result

    def transform_model_input(
        self, features: NDArray[np.float64], feature_names: Sequence[str]
    ) -> NDArray[np.float32]:
        """Transform raw features and cast them at the model-input boundary.

        Args:
            features: Canonical raw features in float64 precision.
            feature_names: Ordered names for the input feature columns.

        Returns:
            A new float32 matrix ready for the model.

        Raises:
            ValueError: Features are invalid or cannot be represented as finite floats.

        Example:
            ```python
            model_input = state.transform_model_input(
                raw.features, raw.feature_names
            )
            ```

        """
        with np.errstate(over="ignore"):
            model_input = self.transform(features, feature_names).astype(np.float32)
        if not np.isfinite(model_input).all():
            raise ValueError("Preprocessing produced non-finite float32 model inputs.")
        return model_input


def _fitted_vector(values: NDArray | None) -> tuple[float, ...] | None:
    return None if values is None else tuple(values.tolist())


def fit_preprocessing(
    dataset: RawDataset,
    partition: TrainingPartition,
    scaler_type: str | None,
    scaler_params: Mapping[str, Any] | None = None,
) -> FittedPreprocessing:
    """Fit a new scaler using only the designated training rows.

    The selected indices must refer to the same raw parent. The caller
    designates eligible training rows through the partition contract.

    Args:
        dataset: Canonical raw dataset containing the training rows.
        partition: Ordered training indices and their parent binding.
        scaler_type: Supported scaler name, or None for disabled scaling.
        scaler_params: Scikit-learn constructor options. Defaults to None.

    Returns:
        Frozen numerical preprocessing bound to the training partition.

    Raises:
        ValueError: Partition, labels, scaling options, or fitted state are invalid.

    Example:
        ```python
        state = fit_preprocessing(
            raw, fold, "RobustScaler", {"quantile_range": [25, 75]}
        )
        ```

    """
    if dataset.dataset_sha256 != partition.dataset_sha256:
        raise ValueError(
            "Training partition does not match the raw dataset provenance."
        )
    if partition.training_indices.max() >= dataset.features.shape[0]:
        raise ValueError("Training index is outside the raw dataset.")
    if dataset.labels is None:
        raise ValueError("Preprocessing fit requires a labeled training dataset.")

    options = dict(scaler_params) if scaler_params is not None else {}
    center = scale = offset = clip_bounds = None
    if scaler_type is None:
        if options:
            raise ValueError("Disabled scaling requires empty scaler parameters.")
    else:
        if not isinstance(scaler_type, str) or scaler_type not in _SCALERS:
            raise ValueError(f"Unsupported scaler type {scaler_type!r}.")
        for key in ("quantile_range", "feature_range"):
            if isinstance(options.get(key), list):
                options[key] = tuple(options[key])
        try:
            scaler = _SCALERS[scaler_type](**options)
            scaler.fit(dataset.features[partition.training_indices])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot fit {scaler_type}: {exc}") from exc
        scale = _fitted_vector(scaler.scale_)
        if isinstance(scaler, MinMaxScaler):
            offset = _fitted_vector(scaler.min_)
            if scaler.clip:
                lower, upper = scaler.feature_range
                clip_bounds = (float(lower), float(upper))
        elif isinstance(scaler, StandardScaler):
            if scaler.with_mean:
                center = _fitted_vector(scaler.mean_)
        else:
            center = _fitted_vector(scaler.center_)

    return FittedPreprocessing(
        scaler_type=scaler_type,
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
        dataset_sha256=dataset.dataset_sha256,
        partition_sha256=partition.partition_sha256,
        fold_index=partition.fold_index,
        center=center,
        scale=scale,
        offset=offset,
        clip_bounds=clip_bounds,
    )


def _weights_digest(weights_path: Path | bytes) -> str:
    if isinstance(weights_path, bytes):
        return hashlib.sha256(weights_path).hexdigest()
    try:
        with weights_path.open("rb") as weights:
            return hashlib.file_digest(weights, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError(f"Cannot read model weights at {weights_path}: {exc}") from exc


def save_preprocessing(
    path: Path, state: FittedPreprocessing, weights_path: Path
) -> None:
    """Write fitted numerical state bound to the exact model checkpoint bytes.

    Callers can stage the sidecar and weights together before publication.

    Args:
        path: Destination for the versioned JSON sidecar.
        state: Fitted preprocessing for the model's training population.
        weights_path: Path to the existing checkpoint to bind.

    Raises:
        ValueError: The model checkpoint cannot be read.
        OSError: The JSON sidecar cannot be written.

    Example:
        ```python
        save_preprocessing(
            Path("preprocessing.json"), state, Path("model.pt")
        )
        ```

    """
    artifact = {
        "version": ARTIFACT_VERSION,
        "weights_sha256": _weights_digest(weights_path),
        "state": asdict(state),
    }
    path.write_text(
        json.dumps(artifact, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON keys.")
    return result


def load_preprocessing(
    path: Path | bytes,
    weights_path: Path | bytes,
    *,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    partition: TrainingPartition | None = None,
) -> FittedPreprocessing:
    """Validate and load numerical preprocessing without deserializing estimators.

    Check the expected feature order and class mapping for every load. When
    the original training partition is available, also check the fit provenance.

    Args:
        path: Path or byte snapshot of the versioned JSON sidecar.
        weights_path: Path or byte snapshot of the matching checkpoint.
        feature_names: Expected ordered feature names.
        class_names: Expected class names in integer-label order.
        partition: Expected fit provenance, when available. Defaults to None.

    Returns:
        Verified frozen preprocessing.

    Raises:
        ValueError: The sidecar is missing, malformed, legacy, or incompatible.

    Example:
        ```python
        state = load_preprocessing(
            Path("preprocessing.json"),
            Path("model.pt"),
            feature_names=("copper",),
            class_names=("background", "deposit"),
        )
        ```

    """
    source = "sidecar snapshot" if isinstance(path, bytes) else path
    try:
        payload = path if isinstance(path, bytes) else path.read_text(encoding="utf-8")
        artifact = json.loads(payload, object_pairs_hook=_unique_keys)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Missing preprocessing artifact at {path}. Retrain to produce a matched bundle."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read preprocessing artifact at {source}: {exc}"
        ) from exc

    if (
        not isinstance(artifact, dict)
        or type(artifact.get("version")) is not int
        or artifact["version"] != ARTIFACT_VERSION
    ):
        raise ValueError(
            "Unsupported or legacy preprocessing version. Retrain to regenerate it."
        )

    try:
        if artifact.keys() != {"version", "weights_sha256", "state"}:
            raise ValueError("Unexpected or missing artifact fields.")
        _check_digest(artifact["weights_sha256"], "Weights digest")
        state_fields = artifact["state"]
        if not isinstance(state_fields, dict) or state_fields.keys() != {
            item.name for item in fields(FittedPreprocessing)
        }:
            raise ValueError("Unexpected or missing numerical state fields.")
        state = FittedPreprocessing(**state_fields)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Malformed preprocessing artifact at {source}: {exc}"
        ) from exc

    if state.feature_names != _names(feature_names, "Expected feature names"):
        raise ValueError(
            f"Feature order mismatch. The model expects {state.feature_names!r}."
        )
    if state.class_names != _names(class_names, "Expected class names"):
        raise ValueError(
            f"Class mapping mismatch. The model expects {state.class_names!r}."
        )
    if partition is not None and (
        state.dataset_sha256 != partition.dataset_sha256
        or state.partition_sha256 != partition.partition_sha256
        or state.fold_index != partition.fold_index
    ):
        raise ValueError(
            "Preprocessing training partition does not match expected provenance."
        )
    if artifact["weights_sha256"] != _weights_digest(weights_path):
        raise ValueError(
            "Model weights digest mismatch. Use the matching preprocessing sidecar."
        )
    return state
