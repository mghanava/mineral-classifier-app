"""Load and apply the distributed synthetic experiment presets."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from src.configuration import ValidationResult, validate_configuration
from src.configuration_yaml import load_configuration_yaml
from src.utilities.reproducibility import DEFAULT_SEED

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_PRESET_IDS = ("quick_demo", "standard", "research")
_PRESET_KEYS = {"id", "version", "display_name", "description", "overrides"}


@dataclass(frozen=True)
class ExperimentPreset:
    """Describe a versioned preset and its scientific overrides."""

    id: str
    version: int
    display_name: str
    description: str
    overrides: Mapping[str, object]


@dataclass(frozen=True)
class AppliedPreset:
    """Carry an independent scientific configuration and its static findings."""

    configuration: dict[str, object]
    validation: ValidationResult


def list_presets() -> tuple[str, ...]:
    """Return the distributed preset identifiers in display order."""
    return _PRESET_IDS


def load_preset(preset_id: str) -> ExperimentPreset:
    """Load a distributed preset or raise ValueError for an invalid definition."""
    if preset_id not in _PRESET_IDS:
        raise ValueError(
            f"Unknown preset {preset_id!r}. Choose {', '.join(_PRESET_IDS)}."
        )
    path = _CONFIG_DIR / "presets" / f"{preset_id}.yaml"
    document = load_configuration_yaml(path)
    if document.keys() != _PRESET_KEYS:
        missing = sorted(_PRESET_KEYS - document.keys())
        unexpected = sorted(map(repr, document.keys() - _PRESET_KEYS))
        raise ValueError(
            f"{path.name!r} has invalid fields. Missing {missing}, unexpected {unexpected}."
        )
    if document["id"] != preset_id:
        raise ValueError(f"{path.name!r} must declare id {preset_id!r}.")
    if type(document["version"]) is not int or document["version"] < 1:
        raise ValueError(f"{path.name!r} version must be a positive integer.")
    for field in ("display_name", "description"):
        value = document[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path.name!r} {field} must be a nonblank string.")
    if not isinstance(document["overrides"], dict):
        raise ValueError(f"{path.name!r} overrides must be a YAML mapping.")
    return ExperimentPreset(**document)


# These are omission fallbacks shared by the existing stages, independent of
# the canonical baseline. Sections and fields without a fallback remain required.
_LEGACY_FALLBACKS = {
    "": {"device": "auto"},
    "data": {"scaler_params": None},
    "data.base": {
        "depth": -500,
        "n_samples": 1000,
        "spacing": 10,
        "threshold_binary": 0.3,
        "min_samples_per_class": 50,
        "k_nearest": None,
        "connection_radius": None,
        "distance_percentile": None,
        "test_size": None,
        "calib_size": None,
        "n_hotspots_max": 5,
        "hotspot_drift": 0.1,
        "mineral_noise_level": 0.1,
        "exp_decay_factor": 0.01,
        "feature_noise_level": 0.5,
        "prototype_evolution_rate": 0.3,
        "class_separation": 2.0,
        "class_influence": 0.3,
        "spatial_weight": 0.5,
        "interaction_strength": 0.1,
        "correlation_strength": 0.2,
        "feature_weight_range": [0.5, 2.0],
        "feature_offset_std": 0.3,
        "seed": DEFAULT_SEED,
    },
    "data.pred": {
        "n_samples": 200,
        "min_samples_per_class": None,
        "seed": DEFAULT_SEED,
    },
    "models.gat": {
        "hidden_channels": 64,
        "n_layers": 3,
        "n_heads": 5,
        "dropout": 0.3,
        "batch_norm": True,
        "negative_slope": 0.2,
        "bias": True,
        "residual": False,
    },
    "models.gcn": {
        "hidden_channels": 64,
        "n_layers": 3,
        "dropout": 0.3,
        "batch_norm": True,
        "improved": False,
    },
    "train": {"seed": DEFAULT_SEED},
    "evaluate": {"seed": DEFAULT_SEED},
    "analyze_drift": {"seed": DEFAULT_SEED, "early_stopping": {}},
}


def _clone(value):
    if isinstance(value, Mapping):
        return {key: _clone(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_clone(child) for child in value)
    return deepcopy(value)


def _overlay(target: dict, overrides: Mapping, *, section: str = "") -> None:
    if (
        section == "data"
        and "scaler_type" in overrides
        and (
            "scaler_type" not in target
            or overrides["scaler_type"] != target["scaler_type"]
        )
        and "scaler_params" not in overrides
    ):
        target["scaler_params"] = {}
    for key, value in overrides.items():
        existing = target.get(key)
        if (
            isinstance(value, Mapping)
            and value
            and (section, key) != ("data", "scaler_params")
        ):
            child = f"{section}.{key}" if section else key
            if not isinstance(existing, dict):
                existing = {}
                target[key] = existing
            _overlay(existing, value, section=child)
        else:
            target[key] = _clone(value)


def _apply_overrides(configuration: dict, overrides: Mapping | None) -> None:
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("Configuration overrides must be a mapping.")
        _overlay(configuration, overrides)


def resolve_new_configuration(
    preset: ExperimentPreset, overrides: Mapping[str, object] | None = None
) -> AppliedPreset:
    """Resolve canonical defaults, preset overrides and explicit overrides in order."""
    if not isinstance(preset.overrides, Mapping):
        raise ValueError("Preset overrides must be a mapping.")
    configuration = load_configuration_yaml(_CONFIG_DIR / "defaults.yaml")
    _overlay(configuration, preset.overrides)
    _apply_overrides(configuration, overrides)
    return AppliedPreset(configuration, validate_configuration(configuration))


def resolve_imported_configuration(
    imported: Mapping[str, object], overrides: Mapping[str, object] | None = None
) -> AppliedPreset:
    """Resolve documented runtime omissions in an existing experiment, then overlay edits."""
    if not isinstance(imported, Mapping):
        raise ValueError("Imported configuration must be a mapping.")
    configuration = _clone(imported)
    for path, defaults in _LEGACY_FALLBACKS.items():
        section = configuration
        for key in path.split(".") if path else ():
            section = section.get(key) if isinstance(section, dict) else None
        if isinstance(section, dict):
            for key, value in defaults.items():
                if key not in section:
                    section[key] = _clone(value)
    _apply_overrides(configuration, overrides)
    validation = validate_configuration(configuration)
    stopping = configuration.get("analyze_drift")
    if isinstance(stopping, dict):
        stopping = stopping.get("early_stopping")
    if isinstance(stopping, dict) and stopping.get("enabled") is False:
        # A nonempty disabled block skips both thresholds at runtime. There is
        # no fallback for missing thresholds in this form of the block.
        validation = ValidationResult(
            tuple(
                replace(
                    issue,
                    code="unresolved_compatibility",
                    message="Supply this value or use an empty early_stopping mapping for the disabled fallback.",
                )
                if issue.code == "missing_key"
                and issue.path.startswith("analyze_drift.early_stopping.")
                else issue
                for issue in validation.issues
            )
        )
    return AppliedPreset(configuration, validation)


def apply_preset(preset: ExperimentPreset) -> AppliedPreset:
    """Resolve a preset against fresh canonical defaults."""
    return resolve_new_configuration(preset)
