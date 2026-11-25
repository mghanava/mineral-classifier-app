"""Verify the distributed presets and their resolved scientific settings."""

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from src import presets
from src.configuration import validate_configuration
from src.presets import apply_preset, list_presets, load_preset

_DEFAULTS = Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"
# Sorted compact JSON of the parsed M1.7 Experiment A params.yaml retained by the run.
_EXPERIMENT_A_SHA256 = (
    "ad05360b7ce0ac9846e8008df644aa4d7629d34d4710a0a748b41b8ada3d664d"
)


def test_list_presets():
    """Expose the stable distributed identifiers in display order."""
    assert list_presets() == ("quick_demo", "standard", "research")


@pytest.mark.parametrize(
    ("preset_id", "name"),
    [("quick_demo", "Quick demo"), ("standard", "Standard"), ("research", "Research")],
)
def test_distributed_metadata_and_validation(preset_id, name):
    """Keep valid versioned metadata separate from the scientific mapping."""
    preset = load_preset(preset_id)
    assert (preset.id, preset.version, preset.display_name) == (preset_id, 1, name)
    assert preset.description.strip()
    applied = apply_preset(preset)
    assert applied.validation == validate_configuration(applied.configuration)
    assert applied.validation.is_valid
    assert set(applied.configuration) == set(yaml.safe_load(_DEFAULTS.read_bytes()))
    assert set(applied.configuration["models"]) == {"gcn", "gat"}


def test_research_matches_canonical_defaults():
    """Research inherits the entire canonical configuration."""
    preset = load_preset("research")
    assert preset.overrides == {}
    applied = apply_preset(preset)
    assert applied.configuration == yaml.safe_load(_DEFAULTS.read_bytes())


def test_quick_demo_matches_actual_experiment_a():
    """Match the entire verified A configuration and keep its budget warnings."""
    applied = apply_preset(load_preset("quick_demo"))
    encoded = json.dumps(applied.configuration, sort_keys=True, separators=(",", ":"))
    assert sha256(encoded.encode()).hexdigest() == _EXPERIMENT_A_SHA256
    assert {
        (issue.path, issue.code)
        for issue in applied.validation.issues
        if issue.severity == "warning"
    } == {
        ("train.patience_learning_rate_scheduler", "unreachable_budget"),
        ("evaluate.patience_learning_rate_scheduler", "unreachable_budget"),
        ("evaluate.patience_early_stopping", "unreachable_budget"),
        ("analyze_drift.early_stopping.min_permutations", "unreachable_budget"),
    }


def test_standard_budgets_and_inheritance():
    """Apply only approved Standard budgets to the canonical configuration."""
    expected = yaml.safe_load(_DEFAULTS.read_bytes())
    expected.update(default_model="gcn", device="auto", cycles=2)
    expected["data"]["base"].update(n_samples=700, n_splits=3)
    expected["data"]["pred"]["n_samples"] = 150
    expected["train"].update(n_epochs=100, patience_early_stopping=20)
    expected["evaluate"]["n_epochs"] = 100
    expected["analyze_drift"]["n_permutations"] = 199
    assert apply_preset(load_preset("standard")).configuration == expected


def test_independent_mappings_and_null_preservation():
    """Preserve null and keep returned nested values independent of their sources."""
    class_names = ["A", "B", "C", "D"]
    base_overrides = {"seed": None, "class_names": class_names}
    preset = replace(
        load_preset("research"),
        overrides=MappingProxyType(
            {"data": {"base": MappingProxyType(base_overrides)}}
        ),
    )
    baseline_bytes = _DEFAULTS.read_bytes()
    first = apply_preset(preset)
    second = apply_preset(preset)
    assert first.validation.is_valid
    assert first.configuration["data"]["base"]["seed"] is None
    first.configuration["data"]["base"]["class_names"].append("E")
    first.configuration["data"]["base"]["feature_weight_range"].append(0)
    assert second.configuration["data"]["base"]["class_names"] == ["A", "B", "C", "D"]
    assert second.configuration["data"]["base"]["feature_weight_range"] == [0.7, 1.7]
    assert base_overrides == {"seed": None, "class_names": ["A", "B", "C", "D"]}
    assert _DEFAULTS.read_bytes() == baseline_bytes


def test_independence_from_active_params(tmp_path, monkeypatch):
    """Resolve equal configurations across changes to the active working file."""
    monkeypatch.chdir(tmp_path)
    active = tmp_path / "params.yaml"
    before = apply_preset(load_preset("quick_demo"))
    active.write_text("default_model: gat\ncycles: 0\n")
    after = apply_preset(load_preset("quick_demo"))
    assert after == before
    assert active.read_text() == "default_model: gat\ncycles: 0\n"


@pytest.mark.parametrize("preset_id", ["unknown", "../defaults", "quick_demo.yaml"])
def test_unknown_presets(preset_id):
    """Reject identifiers outside the distributed catalog."""
    with pytest.raises(ValueError, match="Unknown preset"):
        load_preset(preset_id)


@pytest.fixture
def local_definition(tmp_path, monkeypatch):
    """Provide a temporary copy of the Research definition for malformed input tests."""
    document = load_preset("research")
    monkeypatch.setattr(presets, "_CONFIG_DIR", tmp_path)
    directory = tmp_path / "presets"
    directory.mkdir()
    path = directory / "research.yaml"
    fields = {
        "id": document.id,
        "version": document.version,
        "display_name": document.display_name,
        "description": document.description,
        "overrides": {},
    }
    return path, fields


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "quick_demo"}, "must declare id"),
        ({"version": 0}, "positive integer"),
        ({"version": True}, "positive integer"),
        ({"version": "1"}, "positive integer"),
        ({"display_name": ""}, "display_name must be"),
        ({"description": "  "}, "description must be"),
        ({"displayName": "Research"}, "unexpected"),
        ({"overrides": None}, "overrides must be"),
        ({"overrides": ["cycles"]}, "overrides must be"),
    ],
)
def test_invalid_preset_fields(local_definition, changes, message):
    """Reject malformed metadata and non-mapping overrides at load time."""
    path, fields = local_definition
    path.write_text(yaml.safe_dump(fields | changes))
    with pytest.raises(ValueError, match=message):
        load_preset("research")


def test_missing_metadata(local_definition):
    """Report metadata omitted from the definition."""
    path, fields = local_definition
    del fields["description"]
    path.write_text(yaml.safe_dump(fields))
    with pytest.raises(ValueError, match=r"Missing.*description"):
        load_preset("research")


@pytest.mark.parametrize(
    "contents", ["[broken", "- research", "id: research\nid: research"]
)
def test_invalid_yaml_documents(local_definition, contents):
    """Reject malformed, non-mapping, and duplicate-key YAML definitions."""
    path, _ = local_definition
    path.write_text(contents)
    with pytest.raises(ValueError, match=r"research\.yaml"):
        load_preset("research")


def test_malformed_scientific_overrides_use_the_shared_validator():
    """Return structured findings for unknown paths and invalid scientific values."""
    preset = replace(
        load_preset("research"),
        overrides={"data": {"base": {"n_splits": 1, "n_fold": 3}}, "train": []},
    )
    before = deepcopy(preset.overrides)
    applied = apply_preset(preset)
    assert not applied.validation.is_valid
    assert {
        (issue.path, issue.code)
        for issue in applied.validation.issues
        if issue.severity == "error"
    } == {
        ("data.base.n_splits", "out_of_range"),
        ("data.base.n_fold", "unknown_key"),
        ("train", "invalid_type"),
    }
    assert preset.overrides == before
