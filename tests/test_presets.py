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


def test_resolve_new_order_and_independent_nested_values():
    """Layer explicit edits over a preset and preserve unedited defaults."""
    overrides = {
        "cycles": 0,
        "data": {"base": {"class_names": [], "feature_weight_range": [1, 2]}},
        "models": {"gcn": {"batch_norm": False}},
    }
    original = deepcopy(overrides)
    resolved = presets.resolve_new_configuration(load_preset("quick_demo"), overrides)
    assert resolved.validation.is_valid
    settings = resolved.configuration
    assert settings["cycles"] == 0
    assert settings["train"]["n_epochs"] == 15
    assert settings["data"]["base"]["seed"] == 110
    assert settings["data"]["base"]["class_names"] == []
    assert settings["data"]["base"]["feature_weight_range"] == [1, 2]
    assert settings["models"]["gcn"]["batch_norm"] is False
    assert settings["models"]["gcn"]["n_layers"] == 3
    assert settings["models"]["gat"]["residual"] is True
    settings["data"]["base"]["feature_weight_range"].append(3)
    assert overrides == original


@pytest.mark.parametrize(
    ("edits", "options"),
    [
        ({"scaler_type": "StandardScaler"}, {}),
        (
            {"scaler_type": "MinMaxScaler", "scaler_params": {"clip": False}},
            {"clip": False},
        ),
        ({"scaler_type": None, "scaler_params": None}, None),
        ({"scaler_params": {}}, {}),
        ({"scaler_params": {"with_centering": False}}, {"with_centering": False}),
        ({"scaler_type": "RobustScaler"}, {"quantile_range": [25.0, 75.0]}),
    ],
)
def test_scaler_options_belong_to_the_method(edits, options):
    """Use identical scaler overlay rules for a preset and an imported base."""
    preset = load_preset("research")
    imported = apply_preset(preset).configuration
    for result in (
        presets.resolve_new_configuration(preset, {"data": edits}),
        presets.resolve_imported_configuration(imported, {"data": edits}),
    ):
        assert result.validation.is_valid
        assert result.configuration["data"]["scaler_params"] == options


def test_preset_scaler_override_is_validated_for_selected_method():
    """Reset options on both layers and reject incompatible explicit options."""
    preset = replace(
        load_preset("research"),
        overrides={
            "data": {
                "scaler_type": "StandardScaler",
                "scaler_params": {"with_std": False},
            }
        },
    )
    resolved = presets.resolve_new_configuration(
        preset, {"data": {"scaler_type": "MinMaxScaler"}}
    )
    assert resolved.validation.is_valid
    assert resolved.configuration["data"]["scaler_params"] == {}
    invalid = presets.resolve_new_configuration(
        preset,
        {
            "data": {
                "scaler_type": "MinMaxScaler",
                "scaler_params": {"with_std": False},
            }
        },
    )
    assert any(
        issue.path == "data.scaler_params.with_std" and issue.severity == "error"
        for issue in invalid.validation.issues
    )


def test_import_completes_runtime_omissions_without_research_inheritance():
    """Apply only runtime fallbacks before explicit imported-experiment edits."""
    imported = apply_preset(load_preset("standard")).configuration
    base = imported["data"]["base"]
    for key in (
        "seed",
        "test_size",
        "k_nearest",
        "mineral_noise_level",
        "interaction_strength",
    ):
        del base[key]
    base.update(class_names=[], calib_size=10)
    del imported["data"]["pred"]["n_samples"]
    del imported["data"]["scaler_params"]
    del imported["models"]["gat"]["residual"]
    del imported["models"]["gcn"]["improved"]
    del imported["analyze_drift"]["early_stopping"]
    imported["analyze_drift"].update(
        use_max_sliced_wasserstein=True, use_sinkhorn_wasserstein=True
    )
    imported["train"]["seed"] = None
    before = deepcopy(imported)
    resolved = presets.resolve_imported_configuration(imported, {"cycles": 0})
    assert resolved.validation.is_valid
    result = resolved.configuration
    assert result["cycles"] == 0
    assert result["train"]["n_epochs"] == 100
    assert result["train"]["seed"] is None
    assert result["data"]["base"] == base | {
        "seed": 42,
        "test_size": None,
        "k_nearest": None,
        "mineral_noise_level": 0.1,
        "interaction_strength": 0.1,
    }
    assert result["data"]["pred"]["n_samples"] == 200
    assert result["data"]["scaler_params"] is None
    assert result["analyze_drift"]["early_stopping"] == {}
    assert result["analyze_drift"]["use_max_sliced_wasserstein"] is True
    assert result["analyze_drift"]["use_sinkhorn_wasserstein"] is True
    assert any(
        issue.code == "wasserstein_precedence" for issue in resolved.validation.issues
    )
    assert result["models"]["gat"]["residual"] is False
    assert result["models"]["gcn"]["improved"] is False
    assert imported == before


def test_import_preserves_complete_values_and_explicit_null():
    """Keep explicit null distinct from omissions and honor overriding a fallback."""
    config = apply_preset(load_preset("research")).configuration
    config["data"]["base"].update(seed=None, test_size=100, calib_size=None)
    assert presets.resolve_imported_configuration(config).configuration == config
    del config["data"]["base"]["seed"]
    resolved = presets.resolve_imported_configuration(
        config, {"data": {"base": {"seed": None}}}
    )
    assert resolved.validation.is_valid
    assert resolved.configuration["data"]["base"]["seed"] is None


def test_import_reports_required_unknown_and_unresolved_options():
    """Leave unsupported fields visible and report omissions with no reliable fallback."""
    config = apply_preset(load_preset("research")).configuration
    del config["data"]["base"]["n_features"]
    del config["models"]["gcn"]["normalize"]
    config["data"]["make_edge_weight"] = True
    config["data"]["base"]["k_nearest"] = 5
    config["analyze_drift"]["early_stopping"] = {"enabled": False}
    result = presets.resolve_imported_configuration(config)
    assert result.configuration["data"]["make_edge_weight"] is True
    assert {
        (issue.path, issue.code)
        for issue in result.validation.issues
        if issue.severity == "error"
    } == {
        ("data.base.n_features", "missing_key"),
        ("models.gcn.normalize", "missing_key"),
        ("data.make_edge_weight", "unknown_key"),
        ("data.base", "connectivity_conflict"),
        (
            "analyze_drift.early_stopping.confidence_threshold",
            "unresolved_compatibility",
        ),
        ("analyze_drift.early_stopping.min_permutations", "unresolved_compatibility"),
    }
