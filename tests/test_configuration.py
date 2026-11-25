"""Exercise static configuration validation against the released YAML contract."""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from src.configuration import ValidationIssue, validate_configuration
from src.presets import resolve_imported_configuration

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULTS = _ROOT / "config/defaults.yaml"
_RELEASE_SHA256 = "37926c283c0badd87d5c6ce59403a75aa60bd3fdc8c2935616c1bdebde4cc86d"


@pytest.fixture
def configuration():
    """Return a fresh mapping from the release snapshot."""
    return yaml.safe_load(_DEFAULTS.read_bytes())


def _parent(mapping, path):
    *parents, key = path.split(".")
    for section in parents:
        mapping = mapping[section]
    return mapping, key


def _set(mapping, path, value):
    parent, key = _parent(mapping, path)
    parent[key] = value


def _leaves(mapping, prefix=""):
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _leaves(value, f"{path}.")
        else:
            yield path


def _findings(result):
    return {(issue.severity, issue.path, issue.code) for issue in result.issues}


def test_defaults_match_tagged_snapshot():
    """Pin the exact bytes of v0.2.0.dev1:params.yaml, including comments."""
    assert sha256(_DEFAULTS.read_bytes()).hexdigest() == _RELEASE_SHA256


def test_all_88_release_leaves_are_recognized_and_checked(configuration):
    """Check the inventory even for settings inactive in the release configuration."""
    paths = list(_leaves(configuration))
    assert len(paths) == 88
    for path in paths:
        invalid = deepcopy(configuration)
        _set(invalid, path, {"invalid": True})
        result = validate_configuration(invalid)
        assert ("error", path, "invalid_type") in _findings(result), path
        parent, key = _parent(invalid, path)
        del parent[key]
        if path == "data.scaler_params.quantile_range":
            assert validate_configuration(invalid).is_valid
        else:
            result = validate_configuration(invalid)
            assert ("error", path, "missing_key") in _findings(result), path


def test_current_complete_configuration():
    """Accept the current active YAML without relying on it for defaults."""
    result = validate_configuration(
        yaml.safe_load((_ROOT / "params.yaml").read_bytes())
    )
    assert result.is_valid, result.issues


def test_exact_quick_demo(configuration):
    """Accept the M1.7 Experiment A settings with their original inactive budgets."""
    for path, value in {
        "device": "cpu",
        "cycles": 2,
        "data.base.n_samples": 400,
        "data.pred.n_samples": 100,
        "train.n_epochs": 15,
        "train.patience_early_stopping": 5,
        "evaluate.n_epochs": 20,
        "analyze_drift.n_permutations": 19,
    }.items():
        _set(configuration, path, value)
    result = validate_configuration(configuration)
    assert result.is_valid
    warnings = {
        (issue.path, issue.code)
        for issue in result.issues
        if issue.severity == "warning"
    }
    assert warnings == {
        ("train.patience_learning_rate_scheduler", "unreachable_budget"),
        ("evaluate.patience_learning_rate_scheduler", "unreachable_budget"),
        ("evaluate.patience_early_stopping", "unreachable_budget"),
        ("analyze_drift.early_stopping.min_permutations", "unreachable_budget"),
    }


@pytest.mark.parametrize(
    ("path", "value", "location", "code"),
    [
        ("cycles", True, "cycles", "invalid_type"),
        ("cycles", -1, "cycles", "out_of_range"),
        ("default_model", "GCN", "default_model", "invalid_choice"),
        ("device", "cuda:-1", "device", "invalid_device"),
        ("device", "cuda:1.5", "device", "invalid_device"),
        ("device", "mps", "device", "invalid_device"),
        ("data", [], "data", "invalid_type"),
        ("data.scaler_type", "standard", "data.scaler_type", "invalid_scaler"),
        ("data.scaler_type", [], "data.scaler_type", "invalid_type"),
        ("data.scaler_params", [], "data.scaler_params", "invalid_type"),
        (
            "data.scaler_params.quantile_range",
            [30],
            "data.scaler_params.quantile_range",
            "invalid_length",
        ),
        (
            "data.scaler_params.quantile_range",
            [True, 75],
            "data.scaler_params.quantile_range[0]",
            "invalid_type",
        ),
        (
            "data.scaler_params.quantile_range",
            [75, 25],
            "data.scaler_params.quantile_range",
            "out_of_range",
        ),
        (
            "data.edge_affinity.method",
            "linear",
            "data.edge_affinity.method",
            "invalid_choice",
        ),
        ("data.edge_affinity.scale", False, "data.edge_affinity.scale", "invalid_type"),
        (
            "data.edge_affinity.scale",
            None,
            "data.edge_affinity.scale",
            "affinity_scale",
        ),
        ("data.edge_affinity.scale", 0, "data.edge_affinity.scale", "affinity_scale"),
        (
            "data.base.connection_radius",
            0,
            "data.base.connection_radius",
            "out_of_range",
        ),
        (
            "data.base.distance_percentile",
            101,
            "data.base.distance_percentile",
            "out_of_range",
        ),
        ("data.base.k_nearest", 0, "data.base.k_nearest", "out_of_range"),
        ("data.base.n_samples", 0, "data.base.n_samples", "out_of_range"),
        ("data.base.n_classes", 1, "data.base.n_classes", "out_of_range"),
        ("data.base.n_features", 0, "data.base.n_features", "out_of_range"),
        ("data.base.n_splits", 1, "data.base.n_splits", "out_of_range"),
        ("data.base.class_names", "Gold", "data.base.class_names", "invalid_type"),
        ("data.base.class_names", ["Gold"], "data.base.class_names", "invalid_length"),
        (
            "data.base.class_names",
            ["Gold", "Gold", "C", "D"],
            "data.base.class_names[1]",
            "duplicate_class_name",
        ),
        (
            "data.base.class_names",
            ["Gold", " ", "C", "D"],
            "data.base.class_names[1]",
            "invalid_class_name",
        ),
        (
            "data.base.min_samples_per_class",
            -1,
            "data.base.min_samples_per_class",
            "out_of_range",
        ),
        (
            "data.pred.min_samples_per_class",
            False,
            "data.pred.min_samples_per_class",
            "invalid_type",
        ),
        (
            "data.pred.min_samples_per_class",
            -1,
            "data.pred.min_samples_per_class",
            "out_of_range",
        ),
        (
            "data.base.feature_weight_range",
            [3, 1],
            "data.base.feature_weight_range",
            "out_of_range",
        ),
        ("data.base.test_size", 1.0, "data.base.test_size", "out_of_range"),
        ("data.base.calib_size", 0, "data.base.calib_size", "out_of_range"),
        ("data.base.test_size", True, "data.base.test_size", "invalid_type"),
        ("data.base.seed", 2**32, "data.base.seed", "invalid_seed"),
        ("data.pred.seed", -1, "data.pred.seed", "invalid_seed"),
        ("models.gcn.normalize", False, "models.gcn.normalize", "gcn_normalization"),
        ("models.gcn.n_layers", 0, "models.gcn.n_layers", "out_of_range"),
        ("models.gat.n_layers", 1, "models.gat.n_layers", "out_of_range"),
        ("models.gat.n_heads", 0, "models.gat.n_heads", "out_of_range"),
        ("models.gat.dropout", 1.01, "models.gat.dropout", "out_of_range"),
        ("models.gat.bias", 1, "models.gat.bias", "invalid_type"),
        ("train.lr", "0.1", "train.lr", "invalid_type"),
        ("train.lr", -0.1, "train.lr", "out_of_range"),
        ("train.seed", True, "train.seed", "invalid_type"),
        (
            "train.factor_learning_rate_scheduler",
            1,
            "train.factor_learning_rate_scheduler",
            "out_of_range",
        ),
        ("evaluate.n_bins", 0, "evaluate.n_bins", "out_of_range"),
        (
            "evaluate.calibration_method",
            "sigmoid",
            "evaluate.calibration_method",
            "invalid_choice",
        ),
        (
            "analyze_drift.n_permutations",
            0,
            "analyze_drift.n_permutations",
            "out_of_range",
        ),
        ("analyze_drift.gamma", -1, "analyze_drift.gamma", "out_of_range"),
        (
            "analyze_drift.early_stopping",
            {"enabled": True},
            "analyze_drift.early_stopping.min_permutations",
            "missing_key",
        ),
    ],
)
def test_invalid_settings(configuration, path, value, location, code):
    """Reject malformed inputs and locate each error at the relevant YAML path."""
    _set(configuration, path, value)
    result = validate_configuration(configuration)
    assert not result.is_valid
    assert ("error", location, code) in _findings(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_nonfinite_numbers(configuration, value):
    """Reject unrepresentable reals in scalar and sequence positions."""
    configuration["evaluate"]["eps"] = value
    configuration["data"]["base"]["feature_weight_range"] = [value, 1]
    result = validate_configuration(configuration)
    assert ("error", "evaluate.eps", "nonfinite_number") in _findings(result)
    assert (
        "error",
        "data.base.feature_weight_range[0]",
        "nonfinite_number",
    ) in _findings(result)


def test_multiple_issues_and_null_vs_missing(configuration):
    """Keep distinct issue codes for omission, null and unknown or misplaced fields."""
    del configuration["data"]["pred"]["seed"]
    del configuration["models"]["gat"]
    configuration["train"]["seed"] = None
    configuration["device"] = None
    configuration["models"]["gcn"]["add_self_loops"] = False
    configuration["train"]["n_bins"] = 10
    configuration["data"]["base"]["test.size"] = 0.2
    result = validate_configuration(configuration)
    assert {
        ("error", "data.pred.seed", "missing_key"),
        ("error", "models.gat", "missing_key"),
        ("info", "train.seed", "unseeded_execution"),
        ("error", "device", "null_not_allowed"),
        ("error", "models.gcn.add_self_loops", "unknown_key"),
        ("error", "train.n_bins", "unknown_key"),
        ("error", "data.base['test.size']", "unknown_key"),
    } <= _findings(result)


@pytest.mark.parametrize(
    "connectivity", [(None, None, None), (4, 25, None), (4, None, 5)]
)
def test_connectivity_is_exclusive(configuration, connectivity):
    """Reject both absent and competing graph neighborhood rules."""
    configuration["data"]["base"].update(
        zip(
            ("k_nearest", "connection_radius", "distance_percentile"),
            connectivity,
            strict=True,
        )
    )
    assert ("error", "data.base", "connectivity_conflict") in _findings(
        validate_configuration(configuration)
    )


@pytest.mark.parametrize(
    "key,value",
    [("k_nearest", 4), ("connection_radius", 0.25), ("distance_percentile", 0)],
)
def test_each_connectivity_method_and_unit_affinity(configuration, key, value):
    """Accept each method alone and require null scale for unit weights."""
    configuration["data"]["base"]["distance_percentile"] = None
    configuration["data"]["base"][key] = value
    configuration["data"]["edge_affinity"] = {"method": "unit", "scale": None}
    assert validate_configuration(configuration).is_valid
    configuration["data"]["edge_affinity"]["scale"] = 1
    assert ("error", "data.edge_affinity.scale", "affinity_scale") in _findings(
        validate_configuration(configuration)
    )


@pytest.mark.parametrize(
    ("scaler", "options"),
    [
        (None, None),
        (None, {}),
        ("StandardScaler", None),
        ("MinMaxScaler", {}),
        (
            "RobustScaler",
            {
                "with_centering": False,
                "with_scaling": False,
                "unit_variance": False,
                "copy": True,
                "quantile_range": [0, 100],
            },
        ),
        ("StandardScaler", {"with_mean": True, "with_std": True, "copy": False}),
        ("MinMaxScaler", {"feature_range": [-1, 1], "clip": True, "copy": False}),
    ],
)
def test_supported_scaler_options(configuration, scaler, options):
    """Accept existing scaler-specific kwargs and both no-options representations."""
    configuration["data"].update(scaler_type=scaler, scaler_params=options)
    assert validate_configuration(configuration).is_valid


@pytest.mark.parametrize(
    ("scaler", "options", "option", "code"),
    [
        (None, {"copy": True}, "copy", "scaler_option"),
        (
            "StandardScaler",
            {"quantile_range": [25, 75]},
            "quantile_range",
            "scaler_option",
        ),
        ("RobustScaler", {"feature_range": [0, 1]}, "feature_range", "scaler_option"),
        ("MinMaxScaler", {"with_mean": True}, "with_mean", "scaler_option"),
        ("MinMaxScaler", {"feature_range": [1, 1]}, "feature_range", "out_of_range"),
        ("MinMaxScaler", {"clip": 1}, "clip", "invalid_type"),
        ("RobustScaler", {"unit_variance": None}, "unit_variance", "null_not_allowed"),
    ],
)
def test_scaler_option_validation(configuration, scaler, options, option, code):
    """Reject incompatible options and malformed kwargs without fitting a scaler."""
    configuration["data"].update(scaler_type=scaler, scaler_params=options)
    assert ("error", f"data.scaler_params.{option}", code) in _findings(
        validate_configuration(configuration)
    )


@pytest.mark.parametrize(
    ("holdout", "calibration"),
    [(0.25, 0.5), (200, 50), (None, None), (0.3, 1), (5, None)],
)
def test_supported_partition_sizes(configuration, holdout, calibration):
    """Preserve sklearn fraction, count, and explicit-null semantics."""
    configuration["data"]["base"].update(test_size=holdout, calib_size=calibration)
    assert validate_configuration(configuration).is_valid


def test_incompatible_integer_partition_sizes(configuration):
    """Catch a calibration count that would exhaust a known holdout."""
    configuration["data"]["base"].update(test_size=10, calib_size=10)
    assert ("error", "data.base.calib_size", "partition_conflict") in _findings(
        validate_configuration(configuration)
    )


@pytest.mark.parametrize("class_names", [[], None, ["a", "b", "c", "d"]])
def test_top_up_and_class_name_compatibility(configuration, class_names):
    """Allow targets below the total class floor and preserve generated names."""
    configuration["data"]["base"].update(
        n_samples=91, min_samples_per_class=50, class_names=class_names
    )
    configuration["data"]["pred"].update(n_samples=90, min_samples_per_class=50)
    result = validate_configuration(configuration)
    assert result.is_valid
    assert ("warning", "data.base.n_samples", "population_feasibility") in _findings(
        result
    )


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "cuda:999"])
def test_seed_scopes_and_device_syntax(configuration, monkeypatch, device):
    """Validate large NumPy seeds and explicit null without checking CUDA visibility."""

    def probe():
        pytest.fail("Static validation must not probe CUDA availability.")

    monkeypatch.setattr("torch.cuda.is_available", probe)
    configuration.update(device=device, cycles=0)
    configuration["data"]["pred"]["seed"] = 2**64
    configuration["data"]["base"]["seed"] = 2**32 - 1
    configuration["analyze_drift"]["seed"] = None
    result = validate_configuration(configuration)
    assert result.is_valid
    assert ("info", "analyze_drift.seed", "unseeded_execution") in _findings(result)


@pytest.mark.parametrize(
    ("method", "option", "value", "severity", "code"),
    [
        ("temperature", "initial_temperature", 1, "error", "invalid_type"),
        ("temperature", "initial_temperature", 0.0, "error", "out_of_range"),
        ("temperature", "initial_temperature", -1.0, "warning", "unusual_value"),
        ("platt", "eps", 0, "error", "out_of_range"),
        ("beta", "eps", 0.5, "warning", "unusual_value"),
        ("dirichlet", "initial_temperature", 0, "info", "inactive_setting"),
        ("dirichlet", "reg_mu", -1, "warning", "unusual_value"),
        ("isotonic", "eps", 0, "info", "inactive_setting"),
        ("isotonic", "factor_learning_rate_scheduler", 2, "info", "inactive_setting"),
        ("isotonic", "lr", "0.1", "error", "invalid_type"),
    ],
)
def test_calibration_applicability(
    configuration, method, option, value, severity, code
):
    """Check active constraints while preserving required inactive values."""
    configuration["evaluate"].update(calibration_method=method, **{option: value})
    result = validate_configuration(configuration)
    assert result.is_valid is (severity != "error")
    assert (severity, f"evaluate.{option}", code) in _findings(result)


@pytest.mark.parametrize(
    "stopping",
    [None, {}, {"enabled": False, "confidence_threshold": 0, "min_permutations": 0}],
)
def test_drift_precedence_and_disabled_stopping(configuration, stopping):
    """Preserve the flags and all supported disabled early-stop representations."""
    configuration["analyze_drift"].update(
        use_max_sliced_wasserstein=True,
        use_sinkhorn_wasserstein=True,
        sinkhorn_wasserstein_reg=0,
        early_stopping=stopping,
    )
    result = validate_configuration(configuration)
    assert result.is_valid
    assert (
        "warning",
        "analyze_drift.use_sinkhorn_wasserstein",
        "wasserstein_precedence",
    ) in _findings(result)
    assert (
        "info",
        "analyze_drift.sinkhorn_wasserstein_reg",
        "inactive_setting",
    ) in _findings(result)
    configuration["analyze_drift"]["n_projections"] = 0
    assert not validate_configuration(configuration).is_valid
    configuration["analyze_drift"]["use_max_sliced_wasserstein"] = False
    assert (
        "error",
        "analyze_drift.sinkhorn_wasserstein_reg",
        "out_of_range",
    ) in _findings(validate_configuration(configuration))


def test_proposed_restrictions_remain_nonblocking(configuration):
    """Keep established expert settings that are degenerate but executable."""
    configuration["data"]["base"].update(depth=0, spacing=-1, exp_decay_factor=-0.5)
    configuration["data"]["scaler_params"]["quantile_range"] = [25, 25]
    configuration["models"]["gcn"]["dropout"] = 1.0
    configuration["train"].update(
        lr=0, factor_learning_rate_scheduler=0, patience_early_stopping=0
    )
    configuration["evaluate"]["n_epochs"] = 0
    configuration["analyze_drift"]["gamma"] = 0
    result = validate_configuration(configuration)
    assert result.is_valid
    assert sum(issue.code == "unusual_value" for issue in result.issues) == 10


def test_structured_output_and_input_immutability(configuration):
    """Return ordered immutable issues and accept read-only mappings."""
    configuration.update(cycles=True, surplus=0)
    before = deepcopy(configuration)

    def freeze(mapping):
        return MappingProxyType(
            {
                key: freeze(value) if isinstance(value, Mapping) else value
                for key, value in mapping.items()
            }
        )

    result = validate_configuration(freeze(configuration))
    assert result.issues[:2] == (
        ValidationIssue("error", "cycles", "invalid_type", "Provide an integer."),
        ValidationIssue(
            "error",
            "surplus",
            "unknown_key",
            "Remove this key or move it to its documented section.",
        ),
    )
    assert validate_configuration(configuration) == result
    assert configuration == before


@pytest.mark.parametrize(
    ("quantiles", "scaling", "valid"),
    [
        ([0, 100], True, False),
        ([25, 25], True, False),
        ([25, 75], True, True),
        ([0, 100], False, True),
    ],
)
def test_robust_unit_variance_range(configuration, quantiles, scaling, valid):
    """Check the adjusted scaler coefficients without fitting on a distribution."""
    configuration["data"]["scaler_params"].update(
        quantile_range=quantiles, unit_variance=True, with_scaling=scaling
    )
    result = validate_configuration(configuration)
    assert result.is_valid is valid
    if not valid:
        assert (
            "error",
            "data.scaler_params.quantile_range",
            "invalid_scaler_range",
        ) in _findings(result)


@pytest.mark.parametrize(
    "path",
    (
        "data.base.seed",
        "data.pred.seed",
        "train.seed",
        "evaluate.seed",
        "analyze_drift.seed",
    ),
)
def test_missing_seed_requires_resolution_but_explicit_null_is_valid(
    configuration, path
):
    """Keep strict validation independent from legacy fallback resolution."""
    parent, key = _parent(configuration, path)
    del parent[key]
    assert any(
        (issue.path, issue.code) == (path, "missing_key")
        for issue in validate_configuration(configuration).issues
    )
    imported = resolve_imported_configuration(configuration)
    assert imported.validation.is_valid
    resolved, key = _parent(imported.configuration, path)
    assert resolved[key] == 42
    parent[key] = None
    assert resolve_imported_configuration(configuration).configuration == configuration
