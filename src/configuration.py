"""Validate fully resolved experiment settings without running the pipeline."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from src.utilities.device import validate_device_syntax
from src.utilities.graph_protocol import (
    resolve_connectivity,
    validate_affinity_scale,
    validate_model_graph_settings,
)
from src.utilities.preprocessing import validate_scaler_type
from src.utilities.reproducibility import validate_seed

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class ValidationIssue:
    """Describe a finding at a dotted YAML path with a stable code."""

    severity: Severity
    path: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Hold the ordered static findings for one configuration."""

    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the configuration has no hard errors."""
        return all(issue.severity != "error" for issue in self.issues)


@dataclass(frozen=True)
class _Field:
    kind: str
    nullable: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[str, ...] = ()


_BOOL = _Field("boolean")
_INT = _Field("integer")
_REAL = _Field("number")
_POS_INT = replace(_INT, minimum=1)
_NONNEG_REAL = replace(_REAL, minimum=0)
_PROBABILITY = replace(_REAL, minimum=0, maximum=1)
_SEED = replace(_INT, nullable=True)
_SPLIT = replace(_REAL, nullable=True)
_PAIR = _Field("pair")
_CALIBRATION_METHODS = ("temperature", "isotonic", "platt", "beta", "dirichlet")

_MODEL_FIELDS = {
    "hidden_channels": _POS_INT,
    "dropout": _PROBABILITY,
    "batch_norm": _BOOL,
}
_OPTIMIZER_FIELDS = {
    "n_epochs": _INT,
    "lr": _REAL,
    "weight_decay_adam_optimizer": _REAL,
    "factor_learning_rate_scheduler": _REAL,
    "patience_learning_rate_scheduler": _INT,
    "patience_early_stopping": _INT,
    "min_delta_early_stopping": _REAL,
}
_DRIFT_STOP_FIELDS = {
    "enabled": _BOOL,
    "confidence_threshold": _REAL,
    "min_permutations": _INT,
}
_SCHEMA = {
    "cycles": replace(_INT, minimum=0),
    "add_self_loops": _BOOL,
    "default_model": _Field("string", choices=("gcn", "gat")),
    "device": _Field("string"),
    "data": {
        "scaler_type": _Field("string", nullable=True),
        "scaler_params": _Field("mapping", nullable=True),
        "edge_affinity": {
            "method": _Field("string", choices=("unit", "inverse_quadratic")),
            "scale": replace(_REAL, nullable=True),
        },
        "base": {
            "radius": _NONNEG_REAL,
            "depth": replace(_REAL, maximum=0),
            "n_samples": _POS_INT,
            "spacing": _REAL,
            "n_features": _POS_INT,
            "n_classes": replace(_INT, minimum=2),
            "class_names": _Field("list", nullable=True),
            "threshold_binary": _REAL,
            "min_samples_per_class": replace(_INT, nullable=True, minimum=0),
            "k_nearest": replace(_POS_INT, nullable=True),
            "connection_radius": replace(_REAL, nullable=True),
            "distance_percentile": replace(
                _REAL, nullable=True, minimum=0, maximum=100
            ),
            "n_splits": replace(_INT, minimum=2),
            "test_size": _SPLIT,
            "calib_size": _SPLIT,
            "n_hotspots_max": _POS_INT,
            "hotspot_drift": replace(_NONNEG_REAL, nullable=True),
            "mineral_noise_level": _NONNEG_REAL,
            "exp_decay_factor": _REAL,
            "feature_noise_level": _NONNEG_REAL,
            "prototype_evolution_rate": replace(_NONNEG_REAL, nullable=True),
            "class_separation": _REAL,
            "class_influence": _REAL,
            "spatial_weight": _REAL,
            "interaction_strength": _REAL,
            "correlation_strength": _REAL,
            "feature_weight_range": _PAIR,
            "feature_offset_std": _NONNEG_REAL,
            "seed": _SEED,
        },
        "pred": {
            "n_samples": _POS_INT,
            "min_samples_per_class": replace(_INT, nullable=True, minimum=0),
            "seed": _SEED,
        },
    },
    "models": {
        "gat": {
            **_MODEL_FIELDS,
            "n_layers": replace(_INT, minimum=2),
            "n_heads": _POS_INT,
            "negative_slope": _REAL,
            "bias": _BOOL,
            "residual": _BOOL,
        },
        "gcn": {
            **_MODEL_FIELDS,
            "n_layers": _POS_INT,
            "improved": _BOOL,
            "normalize": _BOOL,
        },
    },
    "train": {**_OPTIMIZER_FIELDS, "seed": _SEED, "max_grad_norm": _REAL},
    "evaluate": {
        **_OPTIMIZER_FIELDS,
        "seed": _SEED,
        "initial_temperature": _REAL,
        "n_bins": _POS_INT,
        "calibration_method": _Field("string", choices=_CALIBRATION_METHODS),
        "verbose": _BOOL,
        "reg_lambda": _REAL,
        "reg_mu": _REAL,
        "eps": _REAL,
    },
    "analyze_drift": {
        "seed": _SEED,
        "n_permutations": _POS_INT,
        "use_max_sliced_wasserstein": replace(_BOOL, nullable=True),
        "n_projections": _INT,
        "use_sinkhorn_wasserstein": replace(_BOOL, nullable=True),
        "sinkhorn_wasserstein_reg": _REAL,
        "gamma": replace(_NONNEG_REAL, nullable=True),
        "early_stopping": _Field("mapping", nullable=True),
    },
}
_SCALER_OPTIONS = {
    None: {},
    "RobustScaler": {
        "quantile_range": _PAIR,
        "with_centering": _BOOL,
        "with_scaling": _BOOL,
        "unit_variance": _BOOL,
        "copy": _BOOL,
    },
    "StandardScaler": {"with_mean": _BOOL, "with_std": _BOOL, "copy": _BOOL},
    "MinMaxScaler": {"feature_range": _PAIR, "clip": _BOOL, "copy": _BOOL},
}
_CONNECTIVITY_KEYS = ("k_nearest", "connection_radius", "distance_percentile")
_TYPES = {
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "string": str,
    "pair": (list, tuple),
    "list": list,
}


def _key_path(parent: str, key: object) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{parent}.{key}" if parent else key
    return f"{parent or '$'}[{key!r}]"


class _Validator:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []
        self.values: dict[str, object] = {}

    def issue(
        self, path: str, code: str, message: str, severity: Severity = "error"
    ) -> None:
        self.issues.append(ValidationIssue(severity, path or "$", code, message))

    def check(self, path: str, code: str, validator: Callable, *args: object) -> bool:
        try:
            validator(*args)
        except ValueError as error:
            self.issue(path, code, str(error))
            return False
        return True

    def walk(
        self,
        value: object,
        schema: Mapping[str, _Field | dict],
        path: str = "",
        *,
        require_keys: bool = True,
        unknown_code: str = "unknown_key",
        unknown_message: str = "Remove this key or move it to its documented section.",
    ) -> None:
        if not isinstance(value, Mapping):
            self.issue(path, "invalid_type", "Provide a mapping for this section.")
            return
        for key, field in schema.items():
            child = _key_path(path, key)
            if key not in value:
                if require_keys:
                    self.issue(
                        child,
                        "missing_key",
                        "Supply this key in the resolved configuration.",
                    )
            elif isinstance(field, _Field):
                self.field(child, value[key], field)
            else:
                self.walk(value[key], field, child)
        for key in sorted(value.keys() - schema.keys(), key=repr):
            child = _key_path(path, key)
            message = unknown_message
            if child in ("models.gcn.add_self_loops", "models.gat.add_self_loops"):
                message = (
                    "Use the top-level add_self_loops setting for graph-owned loops."
                )
            elif child in ("data.make_edge_weight", "data.make_edge_weight_method"):
                message = "Use data.edge_affinity.method and data.edge_affinity.scale."
            self.issue(child, unknown_code, message)

    def field(self, path: str, value: object, field: _Field) -> None:
        expected = "an integer" if field.kind == "integer" else f"a {field.kind}"
        if value is None:
            if field.nullable:
                self.values[path] = value
            else:
                self.issue(path, "null_not_allowed", f"Replace null with {expected}.")
            return

        if field.kind == "mapping":
            valid = isinstance(value, Mapping)
        elif field.kind == "number":
            valid = type(value) in _TYPES["number"]
        elif field.kind == "pair":
            valid = isinstance(value, _TYPES["pair"])
        else:
            valid = type(value) is _TYPES[field.kind]
        if not valid:
            self.issue(path, "invalid_type", f"Provide {expected}.")
            return
        if field.kind == "number":
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                self.issue(
                    path,
                    "nonfinite_number",
                    "Provide a finite number supported by floating-point computation.",
                )
                return
        if field.minimum is not None and value < field.minimum:
            self.issue(
                path, "out_of_range", f"Use a value of at least {field.minimum}."
            )
            return
        if field.maximum is not None and value > field.maximum:
            self.issue(path, "out_of_range", f"Use a value of at most {field.maximum}.")
            return
        if field.choices and value not in field.choices:
            self.issue(
                path, "invalid_choice", f"Choose one of {', '.join(field.choices)}."
            )
            return
        if field.kind == "pair":
            if len(value) != 2:
                self.issue(path, "invalid_length", "Provide exactly two endpoints.")
                return
            for index, endpoint in enumerate(value):
                self.field(f"{path}[{index}]", endpoint, _REAL)
            if not all(f"{path}[{index}]" in self.values for index in range(2)):
                return
        self.values[path] = value

    def range(
        self,
        path: str,
        accepts: Callable[[int | float], bool],
        message: str,
        *,
        severity: Severity = "error",
    ) -> None:
        value = self.values.get(path)
        if isinstance(value, int | float) and not accepts(value):
            self.issue(
                path,
                "out_of_range" if severity == "error" else "unusual_value",
                message,
                severity,
            )

    def inactive(self, path: str, message: str) -> None:
        if path in self.values:
            self.issue(path, "inactive_setting", message, "info")

    def graph_and_device(self) -> None:
        values = self.values
        if "device" in values:
            self.check(
                "device", "invalid_device", validate_device_syntax, values["device"]
            )
        # Only check exclusivity once every connectivity slot has a valid value.
        if all(f"data.base.{key}" in values for key in _CONNECTIVITY_KEYS):
            base = {key: values[f"data.base.{key}"] for key in _CONNECTIVITY_KEYS}
            self.check("data.base", "connectivity_conflict", resolve_connectivity, base)
        self.range(
            "data.base.connection_radius",
            lambda x: x > 0,
            "Use a positive connection radius or null.",
        )
        method, scale = "data.edge_affinity.method", "data.edge_affinity.scale"
        if method in values and scale in values:
            self.check(
                scale,
                "affinity_scale",
                validate_affinity_scale,
                values[method],
                values[scale],
            )
        normalize = "models.gcn.normalize"
        if normalize in values:
            self.check(
                normalize,
                "gcn_normalization",
                validate_model_graph_settings,
                "gcn",
                {"normalize": values[normalize]},
            )
        for model in ("gat", "gcn"):
            self.range(
                f"models.{model}.dropout",
                lambda x: x < 1,
                "Use dropout below 1 to retain signal during training.",
                severity="warning",
            )
        self.range(
            "models.gat.negative_slope",
            lambda x: x >= 0,
            "Prefer a nonnegative LeakyReLU slope.",
            severity="warning",
        )
        if values.get("add_self_loops") is False:
            self.inactive(
                "models.gcn.improved",
                "Improved GCN loop affinity is inactive with graph self-loops disabled.",
            )

    def preprocessing(self) -> None:
        values = self.values
        method, path = "data.scaler_type", "data.scaler_params"
        if method not in values:
            return
        scaler = values[method]
        if not self.check(method, "invalid_scaler", validate_scaler_type, scaler):
            return
        options = values.get(path)
        if isinstance(options, Mapping):
            self.walk(
                options,
                _SCALER_OPTIONS[scaler],
                path,
                require_keys=False,
                unknown_code="scaler_option",
                unknown_message=(
                    "Clear the options mapping or select a scaler that supports this option."
                    if scaler is None
                    else f"Remove this option or select a scaler that supports it. {scaler} does not accept it."
                ),
            )
        quantile_path = f"{path}.quantile_range"
        if quantile_path in values:
            low, high = values[quantile_path]
            if not 0 <= low <= high <= 100:
                self.issue(
                    quantile_path,
                    "out_of_range",
                    "Use ordered endpoints between 0 and 100.",
                )
            elif (
                values.get(f"{path}.unit_variance") is True
                and values.get(f"{path}.with_scaling") is not False
                and (low == high or low == 0 or high == 100)
            ):
                self.issue(
                    quantile_path,
                    "invalid_scaler_range",
                    "With unit variance and scaling enabled, use distinct endpoints strictly inside (0, 100) for a finite positive scale.",
                )
            elif low == high:
                self.issue(
                    quantile_path,
                    "unusual_value",
                    "Equal quantile endpoints are supported. Use distinct endpoints to measure a quantile spread.",
                    "warning",
                )
        feature_path = f"{path}.feature_range"
        if feature_path in values:
            low, high = values[feature_path]
            if not low < high:
                self.issue(
                    feature_path,
                    "out_of_range",
                    "Use a lower endpoint strictly below the upper endpoint.",
                )

    def synthetic(self) -> None:
        values = self.values
        base = "data.base"
        names_path = f"{base}.class_names"
        names = values.get(names_path)
        class_count = values.get(f"{base}.n_classes")
        if names:
            if class_count is not None and len(names) != class_count:
                self.issue(
                    names_path,
                    "invalid_length",
                    "Supply one name per class, an empty list, or null.",
                )
            seen = set()
            for index, name in enumerate(names):
                path = f"{names_path}[{index}]"
                if not isinstance(name, str) or not name.strip():
                    self.issue(
                        path, "invalid_class_name", "Provide a nonblank class name."
                    )
                elif name in seen:
                    self.issue(
                        path,
                        "duplicate_class_name",
                        "Use a unique name for each class.",
                    )
                else:
                    seen.add(name)
        weight_path = f"{base}.feature_weight_range"
        if weight_path in values:
            low, high = values[weight_path]
            if low > high:
                self.issue(
                    weight_path,
                    "out_of_range",
                    "Use a lower endpoint at or below the upper endpoint.",
                )

        if class_count == 2:
            self.range(
                f"{base}.threshold_binary",
                lambda x: 0 <= x <= 1,
                "Prefer a binary threshold within the normalized signal range [0, 1].",
                severity="warning",
            )
        elif class_count is not None:
            self.inactive(
                f"{base}.threshold_binary",
                "This threshold applies only to two-class simulation.",
            )
        for key, predicate, message in (
            (
                "radius",
                lambda x: x > 0,
                "A zero radius collapses XY placement. Check realized coordinates.",
            ),
            ("depth", lambda x: x < 0, "A zero depth generates planar coordinates."),
            (
                "spacing",
                lambda x: x >= 0,
                "Negative spacing imposes no minimum separation.",
            ),
            (
                "exp_decay_factor",
                lambda x: x >= 0,
                "Use a nonnegative factor for distance decay.",
            ),
            (
                "class_separation",
                lambda x: x >= 0,
                "A negative separation reverses the prototype directions.",
            ),
            (
                "spatial_weight",
                lambda x: 0 <= x <= 1,
                "Use a weight in [0, 1] for a convex blend.",
            ),
            (
                "correlation_strength",
                lambda x: 0 <= x <= 1,
                "Prefer a covariance blend in [0, 1]. Values above 1 can produce an invalid covariance.",
            ),
        ):
            self.range(f"{base}.{key}", predicate, message, severity="warning")
        if values.get(f"{base}.n_features") == 1:
            self.issue(
                f"{base}.n_features",
                "diagnostic_feasibility",
                "The simulator supports one feature, but the two-component t-SNE diagnostic requires sufficient feature dimensions.",
                "warning",
            )
        target = values.get(f"{base}.n_samples")
        if target is not None and target <= 100:
            self.issue(
                f"{base}.n_samples",
                "population_feasibility",
                "The bootstrap t-SNE diagnostic requires more than 100 realized rows. Check the population after placement and class top-up.",
                "warning",
            )
        folds = values.get(f"{base}.n_splits")
        floor = values.get(f"{base}.min_samples_per_class")
        if (
            folds is not None
            and f"{base}.min_samples_per_class" in values
            and (floor or 0) < folds
        ):
            self.issue(
                f"{base}.min_samples_per_class",
                "population_feasibility",
                "The requested floor alone may not support all stratified folds. Check realized class counts after holdout.",
                "warning",
            )

    def partitions_and_seeds(self) -> None:
        values = self.values
        holdout = "data.base.test_size"
        calibration = "data.base.calib_size"
        for path in (holdout, calibration):
            self.range(
                path,
                lambda x: x >= 1 if type(x) is int else 0 < x < 1,
                "Use null, a positive integer count, or a fraction strictly between 0 and 1.",
            )
        test_size, calib_size = values.get(holdout), values.get(calibration)
        if (
            type(test_size) is int
            and type(calib_size) is int
            and test_size > 0
            and calib_size >= test_size
        ):
            self.issue(
                calibration,
                "partition_conflict",
                "Choose a calibration count smaller than the integer holdout count.",
            )
        for group in ("data.base", "data.pred", "train", "evaluate", "analyze_drift"):
            path = f"{group}.seed"
            if path not in values:
                continue
            seed = values[path]
            maximum = None if group == "data.pred" else 2**32 - 1
            try:
                validate_seed(seed, path, maximum=maximum)
            except ValueError as error:
                self.issue(path, "invalid_seed", str(error))
            if seed is None:
                self.issue(
                    path,
                    "unseeded_execution",
                    "Explicit null leaves this setting unseeded when its random operations run.",
                    "info",
                )

    def optimizer(self, group: str) -> None:
        self.range(
            f"{group}.lr",
            lambda x: x >= 0,
            "Adam requires a nonnegative learning rate.",
        )
        self.range(
            f"{group}.weight_decay_adam_optimizer",
            lambda x: x >= 0,
            "Adam requires nonnegative weight decay.",
        )
        self.range(
            f"{group}.factor_learning_rate_scheduler",
            lambda x: x < 1,
            "ReduceLROnPlateau requires a factor below 1.",
        )
        for key, predicate, message in (
            ("lr", lambda x: x > 0, "A zero learning rate prevents optimizer updates."),
            (
                "factor_learning_rate_scheduler",
                lambda x: x > 0,
                "A nonpositive factor can reduce the learning rate to zero.",
            ),
            (
                "patience_learning_rate_scheduler",
                lambda x: x >= 0,
                "Negative scheduler patience can reduce the rate immediately.",
            ),
            (
                "patience_early_stopping",
                lambda x: x > 0,
                "Nonpositive early-stopping patience can stop after the first epoch.",
            ),
            (
                "min_delta_early_stopping",
                lambda x: x >= 0,
                "A negative minimum delta can count deterioration as improvement.",
            ),
        ):
            self.range(f"{group}.{key}", predicate, message, severity="warning")
        budget = self.values.get(f"{group}.n_epochs")
        if budget is not None and budget > 0:
            for key, minimum, overhead in (
                ("patience_learning_rate_scheduler", 0, 2),
                ("patience_early_stopping", 1, 1),
            ):
                path = f"{group}.{key}"
                patience = self.values.get(path)
                if (
                    patience is not None
                    and patience >= minimum
                    and patience + overhead > budget
                ):
                    self.issue(
                        path,
                        "unreachable_budget",
                        f"This control cannot trigger within the {budget}-epoch budget.",
                        "warning",
                    )

    def training_and_calibration(self) -> None:
        self.range(
            "train.n_epochs",
            lambda x: x >= 1,
            "Use at least one training epoch for the development refit.",
        )
        self.range(
            "train.max_grad_norm",
            lambda x: x > 0,
            "Use a positive clipping norm to avoid zeroing or reversing gradients.",
            severity="warning",
        )
        self.optimizer("train")

        method = self.values.get("evaluate.calibration_method")
        if method is None:
            return
        if method == "temperature":
            path = "evaluate.initial_temperature"
            if type(self.values.get(path)) is int:
                self.issue(
                    path,
                    "invalid_type",
                    "Use a floating-point value, such as 1.0, for Torch's trainable temperature.",
                )
            else:
                self.range(
                    path,
                    lambda x: x != 0,
                    "Use a nonzero initial temperature to avoid division by zero.",
                )
                self.range(
                    path,
                    lambda x: x >= 0,
                    "A negative temperature reverses the ordering of logits.",
                    severity="warning",
                )
        else:
            self.inactive(
                "evaluate.initial_temperature",
                "This setting applies only to temperature calibration.",
            )
        if method != "dirichlet":
            self.inactive(
                "evaluate.reg_mu",
                "This bias penalty applies only to Dirichlet calibration.",
            )
        else:
            self.range(
                "evaluate.reg_mu",
                lambda x: x >= 0,
                "Use a nonnegative bias regularization penalty.",
                severity="warning",
            )
        if method in ("platt", "beta"):
            self.range(
                "evaluate.eps",
                lambda x: 0 < x < 1,
                "Use epsilon strictly between 0 and 1 to keep logarithm inputs positive.",
            )
            self.range(
                "evaluate.eps",
                lambda x: x < 0.5 or x >= 1,
                "Use epsilon below 0.5 to preserve the probability range.",
                severity="warning",
            )
        else:
            self.inactive(
                "evaluate.eps",
                "This probability clamp applies only to Platt and beta calibration.",
            )
        if method == "isotonic":
            for key in (*_OPTIMIZER_FIELDS, "verbose", "reg_lambda", "seed"):
                self.inactive(
                    f"evaluate.{key}",
                    "Isotonic fitting does not use parametric fitting controls.",
                )
        else:
            self.range(
                "evaluate.n_epochs",
                lambda x: x > 0,
                "Use a positive epoch budget to fit parametric calibration.",
                severity="warning",
            )
            self.range(
                "evaluate.reg_lambda",
                lambda x: x >= 0,
                "Use a nonnegative weight regularization penalty.",
                severity="warning",
            )
            self.optimizer("evaluate")

    def drift(self) -> None:
        values = self.values
        base = "analyze_drift"
        sliced = values.get(f"{base}.use_max_sliced_wasserstein")
        sinkhorn = values.get(f"{base}.use_sinkhorn_wasserstein")
        features = values.get("data.base.n_features")
        if sliced is True and sinkhorn is True:
            self.issue(
                f"{base}.use_sinkhorn_wasserstein",
                "wasserstein_precedence",
                "Both flags are enabled. Existing behavior gives max-sliced Wasserstein precedence over Sinkhorn.",
                "warning",
            )
        if sliced is True and features != 1:
            self.range(
                f"{base}.n_projections",
                lambda x: x >= 1,
                "Use at least one projection for max-sliced Wasserstein.",
            )
        elif sliced is not True or features == 1:
            self.inactive(
                f"{base}.n_projections",
                "Projections apply only to multivariate max-sliced Wasserstein.",
            )
        if sliced is not True and sinkhorn is True and features != 1:
            self.range(
                f"{base}.sinkhorn_wasserstein_reg",
                lambda x: x > 0,
                "Use positive regularization for Sinkhorn transport.",
            )
        else:
            self.inactive(
                f"{base}.sinkhorn_wasserstein_reg",
                "Regularization applies only to multivariate Sinkhorn transport.",
            )
        self.range(
            f"{base}.gamma",
            lambda x: x > 0,
            "Zero gamma produces a constant RBF kernel.",
            severity="warning",
        )
        path = f"{base}.early_stopping"
        stopping = values.get(path)
        if not stopping:
            self.inactive(
                path,
                "An empty map or null uses the existing disabled early-stop fallback.",
            )
            return
        self.walk(stopping, _DRIFT_STOP_FIELDS, path)
        minimum = f"{path}.min_permutations"
        threshold = f"{path}.confidence_threshold"
        if values.get(f"{path}.enabled") is True:
            budget = values.get(f"{base}.n_permutations")
            if minimum in values and budget is not None and values[minimum] > budget:
                self.issue(
                    minimum,
                    "unreachable_budget",
                    f"Early stopping cannot activate within the {budget}-permutation budget.",
                    "warning",
                )
            self.range(
                minimum,
                lambda x: x >= 1,
                "A minimum below 1 permits stopping after the first permutation.",
                severity="warning",
            )
            self.range(
                threshold,
                lambda x: 0 < x < 1,
                "Use a threshold in (0, 1). Nonpositive thresholds prevent stopping, and thresholds above 1 always trigger once eligible.",
                severity="warning",
            )
        elif values.get(f"{path}.enabled") is False:
            self.inactive(
                minimum,
                "The minimum applies only when permutation early stopping is enabled.",
            )
            self.inactive(
                threshold,
                "The threshold applies only when permutation early stopping is enabled.",
            )


def validate_configuration(configuration: object) -> ValidationResult:
    """Check a resolved YAML mapping without mutation, I/O or execution.

    Paths use dotted keys, zero-based list indices and `$` for the root.
    Warnings and informational findings do not make the result invalid.
    """
    validator = _Validator()
    validator.walk(configuration, _SCHEMA)
    validator.graph_and_device()
    validator.preprocessing()
    validator.synthetic()
    validator.partitions_and_seeds()
    validator.training_and_calibration()
    validator.drift()
    return ValidationResult(tuple(validator.issues))
