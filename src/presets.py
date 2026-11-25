"""Load and apply the distributed synthetic experiment presets."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.configuration import ValidationResult, validate_configuration

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


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            raise ValueError(f"Duplicate mapping key at {node.start_mark}.")
        return mapping


def _read_mapping(path: Path) -> dict:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot load {path.name!r}. {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path.name!r} must contain a YAML mapping.")
    return document


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
    document = _read_mapping(path)
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


def _overlay(target: dict, overrides: Mapping) -> None:
    for key, value in overrides.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping) and value:
            _overlay(existing, value)
        elif isinstance(value, Mapping):
            target[key] = deepcopy(dict(value))
        else:
            target[key] = deepcopy(value)


def apply_preset(preset: ExperimentPreset) -> AppliedPreset:
    """Apply a preset to fresh canonical defaults and return static validation."""
    if not isinstance(preset.overrides, Mapping):
        raise ValueError("Preset overrides must be a mapping.")
    configuration = _read_mapping(_CONFIG_DIR / "defaults.yaml")
    _overlay(configuration, preset.overrides)
    return AppliedPreset(configuration, validate_configuration(configuration))
