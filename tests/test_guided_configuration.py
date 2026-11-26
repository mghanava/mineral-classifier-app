"""Exercise the guided editor through its normal Streamlit entry point."""

from pathlib import Path
from shutil import copyfile

import pytest
from streamlit.testing.v1 import AppTest

from scripts import configure as configure_cli
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml
from src.presets import apply_preset, load_preset, resolve_new_configuration

APP = Path(__file__).resolve().parents[1] / "app.py"
CONFIGURE = "dashboard/views/configure.py"


@pytest.fixture
def guided(tmp_path, monkeypatch):
    """Open Basic in a disposable workspace with the production YAML as a fixture."""
    copyfile(APP.parent / "params.yaml", tmp_path / "params.yaml")
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(str(APP)).switch_page(CONFIGURE).run()


def _widget(elements, label):
    return next(widget for widget in elements if widget.label == label)


def _level(app, level):
    app.radio[0].set_value(level).run()
    assert not app.exception


def _select(app, label, value):
    _widget(app.selectbox, label).select(value).run()
    assert not app.exception


def _group(app, group):
    _level(app, "Advanced")
    _select(app, "Advanced group", group)


def _click(app, label):
    _widget(app.button, label).click().run()
    assert not app.exception


def _draft(app):
    return parse_configuration_yaml(app.session_state["mineral_dashboard"].draft)


def _fields(app):
    return {
        widget.key.split(":", 2)[-1]
        for widget in [
            *app.number_input,
            *app.text_input,
            *app.selectbox,
            *app.checkbox,
        ]
        if widget.key and widget.key.startswith("_guided:")
    }


def test_basic_fields_preset_loading_and_cli_equivalence(guided, tmp_path, capsys):
    """Edit Basic without a YAML widget and match the existing CLI preset resolver."""
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    assert not guided.text_area
    assert _fields(guided) == {
        "default_model",
        "device",
        "cycles",
        "data.base.n_samples",
        "data.pred.n_samples",
    }
    _select(guided, "Experiment preset", "standard")
    assert _draft(guided)["cycles"] == 3
    assert any("101.66 s" in item.value for item in guided.caption)
    _click(guided, "Load preset into draft")
    assert _widget(guided.number_input, "Learning cycles").value == 2
    assert _widget(guided.number_input, "Requested bootstrap observations").value == 700
    _select(guided, "Model", "GAT")
    _select(guided, "Computing device", "CUDA")
    _widget(guided.text_input, "CUDA device index").set_value("2").run()
    _widget(guided.number_input, "Learning cycles").set_value(0).run()
    assert any("Bootstrap only" in item.value for item in guided.caption)
    _select(guided, "Experiment preset", "quick_demo")
    assert _draft(guided)["device"] == "cuda:2"
    assert any("74.31 s" in item.value for item in guided.caption)

    overrides = {"default_model": "gat", "device": "cuda:2", "cycles": 0}
    expected = resolve_new_configuration(
        load_preset("standard"), overrides
    ).configuration
    override_file = tmp_path / "overrides.yaml"
    override_file.write_text(dump_configuration_yaml(overrides))
    assert (
        configure_cli.main(["resolve", "standard", "--overrides", str(override_file)])
        == 0
    )
    cli = parse_configuration_yaml(capsys.readouterr().out)
    assert cli == expected == _draft(guided)

    _click(guided, "Validate & Preview")
    assert any("0 errors" in item.value for item in guided.markdown)
    assert active.read_bytes() == original
    assert not (tmp_path / "dvc.yaml").exists()
    _click(guided, "Apply configuration")
    assert parse_configuration_yaml(active.read_bytes()) == expected
    assert not (tmp_path / "dvc.yaml").exists()
    assert guided.get("page_link")


@pytest.mark.parametrize(
    ("group", "paths"),
    [
        (
            "Synthetic environment and class schema",
            {
                "data.base.radius",
                "data.base.depth",
                "data.base.spacing",
                "data.base.n_features",
                "data.base.n_classes",
                "data.base.class_names",
                "data.base.min_samples_per_class",
            },
        ),
        (
            "Partitions and validation",
            {"data.base.n_splits", "data.base.test_size", "data.base.calib_size"},
        ),
        (
            "Spatial graph and preprocessing",
            {
                "connectivity",
                "data.base.distance_percentile",
                "add_self_loops",
                "data.edge_affinity.method",
                "data.edge_affinity.scale",
                "data.scaler_type",
            },
        ),
        ("Model training", {"train.n_epochs"}),
        (
            "Probability calibration",
            {"evaluate.calibration_method", "evaluate.n_bins", "evaluate.n_epochs"},
        ),
        (
            "Drift analysis",
            {
                "wasserstein",
                "analyze_drift.n_permutations",
                "analyze_drift.early_stopping.enabled",
            },
        ),
        (
            "Reproducibility",
            {
                "data.base.seed",
                "data.pred.seed",
                "train.seed",
                "evaluate.seed",
                "analyze_drift.seed",
            },
        ),
    ],
)
def test_advanced_group_field_coverage(guided, group, paths):
    """Render exactly the selected inventory group, with conditional branches below."""
    _group(guided, group)
    assert _fields(guided) == paths


def test_graph_and_scaler_transitions_preserve_unedited_options(guided):
    """Map a single graph rule onto three fields and use resolver-owned scaler transitions."""
    _group(guided, "Spatial graph and preprocessing")
    before = _draft(guided)
    _select(guided, "Connectivity rule", "K-nearest neighbors")
    _widget(guided.number_input, "Neighbors per node").set_value(17).run()
    assert _draft(guided)["data"]["base"]["k_nearest"] == 17
    _select(guided, "Connectivity rule", "Fixed radius")
    _widget(guided.text_input, "Connection radius").set_value("125.5").run()
    _select(guided, "Edge affinity", "Unit weights")
    assert "data.edge_affinity.scale" not in _fields(guided)
    _select(guided, "Feature scaler", "StandardScaler")
    current = _draft(guided)
    assert current["data"]["base"] == {
        **before["data"]["base"],
        "k_nearest": None,
        "connection_radius": 125.5,
        "distance_percentile": None,
    }
    assert current["data"]["edge_affinity"] == {"method": "unit", "scale": None}
    assert current["data"]["scaler_params"] == {}
    assert current["models"] == before["models"]
    _select(guided, "Edge affinity", "Inverse quadratic")
    _widget(guided.text_input, "Affinity scale").set_value("75").run()
    assert _draft(guided)["data"]["edge_affinity"]["scale"] == 75
    _select(guided, "Feature scaler", "Identity (no scaling)")
    assert _draft(guided)["data"]["scaler_type"] is None


def test_partition_types_null_seeds_and_empty_class_names(guided):
    """Round-trip integer counts, fractions, nulls, and empty sequences."""
    _group(guided, "Partitions and validation")
    _widget(guided.text_input, "Holdout size of the full base").set_value("25").run()
    _widget(guided.text_input, "Calibration size of the holdout").set_value("0.2").run()
    base = _draft(guided)["data"]["base"]
    assert type(base["test_size"]) is int and base["test_size"] == 25
    assert type(base["calib_size"]) is float and base["calib_size"] == 0.2
    _widget(guided.text_input, "Holdout size of the full base").set_value("").run()
    _select(guided, "Advanced group", "Synthetic environment and class schema")
    _widget(guided.text_input, "Class names").set_value("[]").run()
    _select(guided, "Advanced group", "Reproducibility")
    _widget(guided.text_input, "Bootstrap and partition seed").set_value("").run()
    _widget(guided.text_input, "Incoming batch seed").set_value(
        "9007199254740997"
    ).run()
    _level(guided, "Expert")
    document = parse_configuration_yaml(guided.text_area[0].value)
    assert document["data"]["base"]["test_size"] is None
    assert document["data"]["base"]["seed"] is None
    assert document["data"]["base"]["class_names"] == []
    assert document["data"]["pred"]["seed"] == 9007199254740997
    _level(guided, "Advanced")
    assert _widget(guided.selectbox, "Advanced group").value == "Reproducibility"
    assert _widget(guided.text_input, "Bootstrap and partition seed").value == ""


def test_expert_to_guided_preserves_inactive_calibration_and_drift_fields(guided):
    """Retain Expert values through conditional selectors and legacy dual flags."""
    config = apply_preset(load_preset("quick_demo")).configuration
    config["analyze_drift"]["use_max_sliced_wasserstein"] = True
    config["analyze_drift"]["use_sinkhorn_wasserstein"] = True
    config["analyze_drift"]["early_stopping"] = None
    config["data"]["scaler_params"]["with_centering"] = False
    _level(guided, "Expert")
    guided.text_area[0].set_value(dump_configuration_yaml(config)).run()
    _group(guided, "Probability calibration")
    assert _widget(guided.number_input, "Calibration fit epoch budget").value == 20
    _select(guided, "Calibration method", "Isotonic")
    assert "evaluate.n_epochs" not in _fields(guided)
    _select(guided, "Advanced group", "Reproducibility")
    assert "evaluate.seed" not in _fields(guided)
    assert _draft(guided)["evaluate"] == {
        **config["evaluate"],
        "calibration_method": "isotonic",
    }
    _select(guided, "Advanced group", "Drift analysis")
    assert any("Max-sliced takes precedence" in item.value for item in guided.warning)
    _widget(guided.number_input, "Permutation budget").set_value(35).run()
    drift = _draft(guided)["analyze_drift"]
    assert drift["use_max_sliced_wasserstein"] and drift["use_sinkhorn_wasserstein"]
    assert drift["early_stopping"] is None
    _widget(guided.checkbox, "Enable permutation early stopping").check().run()
    _select(guided, "Wasserstein method", "Sinkhorn")
    _level(guided, "Expert")
    final = parse_configuration_yaml(guided.text_area[0].value)
    assert final["data"]["scaler_params"] == config["data"]["scaler_params"]
    assert final["models"] == config["models"]
    assert final["analyze_drift"]["use_max_sliced_wasserstein"] is False
    assert final["analyze_drift"]["use_sinkhorn_wasserstein"] is True
    assert final["analyze_drift"]["early_stopping"]["enabled"] is True
    assert (
        final["analyze_drift"]["n_projections"]
        == config["analyze_drift"]["n_projections"]
    )


@pytest.mark.parametrize("invalid", ["cycles: [\n", "cycles: -1\n"])
def test_expert_invalid_draft_survives_mode_and_page_changes(guided, tmp_path, invalid):
    """Keep the exact accepted widget buffer while disabling guided edits."""
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    _click(guided, "Validate & Preview")
    _level(guided, "Expert")
    guided.text_area[0].set_value(invalid)
    _level(
        guided, "Basic"
    )  # submit the editor and level change in the same widget event
    assert not _fields(guided)
    assert guided.error
    assert not any(button.label == "Apply configuration" for button in guided.button)
    guided.switch_page("dashboard/views/run.py").run()
    guided.switch_page(CONFIGURE).run()
    _level(guided, "Expert")
    assert guided.text_area[0].value == invalid
    _select(guided, "Experiment preset", "standard")
    assert guided.text_area[0].value == invalid
    _click(guided, "Load preset into draft")
    assert _draft(guided)["data"]["base"]["n_samples"] == 700
    assert active.read_bytes() == original


def test_guided_edit_invalidates_preview_and_exposes_blocking_finding(guided, tmp_path):
    """Keep guided invalid values actionable and protect the active snapshot."""
    active = tmp_path / "params.yaml"
    original = active.read_bytes()
    _click(guided, "Validate & Preview")
    _widget(guided.number_input, "Learning cycles").set_value(-1).run()
    assert all(button.label != "Apply configuration" for button in guided.button)
    _click(guided, "Validate & Preview")
    assert any("cycles" in item.value for item in guided.error)
    assert _widget(guided.button, "Apply configuration").disabled
    assert all(not expander.proto.expanded for expander in guided.expander)
    _widget(guided.number_input, "Learning cycles").set_value(1).run()
    _click(guided, "Validate & Preview")
    assert not _widget(guided.button, "Apply configuration").disabled
    assert active.read_bytes() == original
