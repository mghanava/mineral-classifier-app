"""Check shared spatial presentation, scoped selection and download invalidation."""

import json
import os
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_spatial_predictions import spatial_fixture as spatial_fixture
from test_spatial_predictions import workspace as workspace

from dashboard import spatial_explorer
from dashboard.artifacts import Workspace
from dashboard.spatial_data import PopulationKind, read_population
from dashboard.spatial_explorer import prediction_download
from dashboard.spatial_figure import (
    CLASS_COLORS,
    CLASS_SYMBOLS,
    ColorField,
    ColorMode,
    color_field,
    observation_label,
    spatial_figure,
)
from dashboard.spatial_predictions import ProbabilityFamily, verify_prediction_join

APP = Path(__file__).resolve().parents[1] / "app.py"


def _widget(app, kind, label):
    return next(widget for widget in getattr(app, kind) if widget.label == label)


def _select(app, label, value):
    return _widget(app, "selectbox", label).select(value).run()


def _caption(app, text):
    return any(text in item.value for item in app.caption)


def _start(workspace, monkeypatch):
    monkeypatch.chdir(workspace.root)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.switch_page("dashboard/views/results.py").run()
    _widget(app, "segmented_control", "Results mode").select("Cycle explorer").run()
    assert not app.exception
    return app


def _spec(app):
    return json.loads(app.get("plotly_chart")[0].proto.spec)


def test_plot_geometry_categories_and_visible_links(workspace):
    """Keep positions and category indices while filtering and drawing stored links."""
    population = read_population(workspace, PopulationKind.INCOMING, cycle=1)
    subset = population.view(class_ids={2})
    field = color_field(population, ColorMode.OBSERVED)
    args = {
        "dark": True,
        "links": subset.edge_rows(),
        "selected": None,
        "revision": "test",
    }
    figure = spatial_figure(subset, field, three_d=False, **args)
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.scaleratio == 1
    category = figure.data[-1]
    assert (category.marker.color, category.marker.symbol) == (
        CLASS_COLORS[2],
        CLASS_SYMBOLS[2],
    )
    assert [item[0] for item in category.customdata] == [
        population.sample_ids[row] for row in subset.row_indices
    ]
    selected = population.sample_ids[subset.row_indices[0]]
    args["selected"] = selected
    figure = spatial_figure(subset, field, three_d=True, **args)
    assert figure.layout.scene.aspectmode == "data"
    assert figure.layout.scene.zaxis.autorange is True
    assert figure.layout.scene.zaxis.nticks == 3
    assert 0 < figure.layout.scene.domain.x[0] < figure.layout.scene.domain.x[1] < 1
    assert 0 < figure.layout.scene.domain.y[0] < figure.layout.scene.domain.y[1] < 1
    np.testing.assert_array_equal(
        figure.data[-2].z, population.coordinates[subset.row_indices, 2]
    )
    assert figure.data[-1].name == "Selected observation"
    if len(subset.edge_rows()):
        assert figure.data[0].mode == "lines"
        np.testing.assert_array_equal(
            figure.data[0].z[::3], population.coordinates[subset.edge_rows()[:, 0], 2]
        )


@pytest.mark.parametrize("family", list(ProbabilityFamily))
def test_spatial_field_domains_and_family(workspace, family):
    """Use exact fields and global display domains for either probability family."""
    joined = verify_prediction_join(workspace, 1).joined
    assert joined
    population = joined.population
    for mode, expected in (
        (ColorMode.PREDICTED, joined.scores(family).predicted_class_ids()),
        (ColorMode.MAXIMUM, joined.scores(family).maximum_probability()),
        (ColorMode.PROBABILITY, joined.scores(family).selected_probability(1)),
        (ColorMode.ENTROPY, joined.scores(family).entropy_nats),
        (ColorMode.CORRECTNESS, joined.correctness(family)),
        (ColorMode.FEATURE, population.raw_features[:, 1]),
    ):
        field = color_field(population, mode, joined=joined, family=family, column=1)
        np.testing.assert_array_equal(field.values, expected)
        if mode in (ColorMode.MAXIMUM, ColorMode.PROBABILITY):
            assert (field.bounds, field.scale) == ((0, 1), "Cividis")
            assert field.tick_format == ".1%"
            assert field.display_value(0.5271) == "52.7%"
        if mode == ColorMode.ENTROPY:
            assert (field.bounds, field.scale) == (
                (0, np.log(len(population.class_names))),
                "Viridis",
            )
    appended = read_population(workspace, PopulationKind.POST_APPEND, cycle=1)
    membership = color_field(appended, ColorMode.MEMBERSHIP)
    assert membership.categories == ("Input base", "Incoming batch")
    assert sum(membership.values) == 16


@pytest.mark.parametrize(
    "mode", [ColorMode.MAXIMUM, ColorMode.PROBABILITY, ColorMode.ENTROPY]
)
def test_population_color_range_is_invariant_under_filters(workspace, mode):
    """Take relative bounds from all stored scores for the active family."""
    joined = verify_prediction_join(workspace, 1).joined
    assert joined
    population = joined.population
    for family in ProbabilityFamily:
        field = color_field(
            population,
            mode,
            joined=joined,
            family=family,
            column=1,
            population_range=True,
        )
        scores = joined.scores(family)
        values = {
            ColorMode.MAXIMUM: scores.maximum_probability(),
            ColorMode.PROBABILITY: scores.selected_probability(1),
            ColorMode.ENTROPY: scores.entropy_nats,
        }[mode]
        expected = (float(values.min()), float(values.max()))
        assert field.bounds == expected
        view = population.view(class_ids={0})
        figure = spatial_figure(
            view,
            field,
            three_d=False,
            dark=False,
            links=view.edge_rows()[:0],
            selected=None,
            revision="relative",
        )
        assert (figure.data[0].marker.cmin, figure.data[0].marker.cmax) == expected
        assert figure.layout.xaxis.range == pytest.approx(
            spatial_figure(
                population.view(),
                field,
                three_d=False,
                dark=False,
                links=view.edge_rows()[:0],
                selected=None,
                revision="relative",
            ).layout.xaxis.range
        )


def test_constant_population_range_has_one_marker_color(workspace):
    """Use a single marker color when the relative range collapses."""
    population = read_population(workspace, PopulationKind.INCOMING, cycle=1)
    field = ColorField(
        "Max probability",
        np.full(population.count, 0.25),
        scale="Cividis",
        bounds=(0.25, 0.25),
    )
    figure = spatial_figure(
        population.view(),
        field,
        three_d=False,
        dark=True,
        links=population.view().edge_rows()[:0],
        selected=None,
        revision="constant",
    )
    assert isinstance(figure.data[0].marker.color, str)
    assert not figure.data[0].marker.showscale


def test_population_summary_and_short_label_identity(workspace, monkeypatch):
    """Expose zero counts while ordinals and exact IDs keep their separate roles."""
    app = _start(workspace, monkeypatch)
    assert any("Population summary" in item.value for item in app.markdown)
    assert _caption(app, "96 total")
    _widget(app, "multiselect", "Observed classes").set_value([]).run()
    tables = [
        item.proto.body for item in app.get("html") if "scope='row'" in item.proto.body
    ]
    assert len(tables) == 1
    assert "Shown</th>" in tables[0]
    assert tables[0].count("<td>0</td>") == 4
    _select(app, "Result section", "Spatial predictions")
    population = app.session_state["_spatial_session"].population
    ids = population.sample_ids
    picker = _widget(app, "selectbox", "Observation ID")
    assert picker.options == [
        observation_label(row, len(ids)) for row in range(len(ids))
    ]
    picker.select(ids[1]).run()
    assert app.session_state["_spatial_session"].selected == ids[1]


def test_zero_predicted_counts_and_exact_full_id_lookup(workspace, monkeypatch):
    """Retain zero class slots and reject a short label as an identity key."""
    population = read_population(workspace, PopulationKind.INCOMING, cycle=1)
    view = population.view(class_ids=set())
    total, shown = spatial_explorer._population_totals(
        np.zeros(population.count, dtype=int), 4, view
    )
    assert list(total) == [population.count, 0, 0, 0]
    assert list(shown) == [0, 0, 0, 0]

    state = spatial_explorer._SpatialSession()
    session = {"search": observation_label(0, population.count)}
    monkeypatch.setattr(spatial_explorer.st, "session_state", session)
    spatial_explorer._find_stable_id(state, "picker", "search", population.sample_ids)
    assert state.selected is None
    session["search"] = f" {population.sample_ids[-1]} "
    spatial_explorer._find_stable_id(state, "picker", "search", population.sample_ids)
    assert session["picker"] == state.selected == population.sample_ids[-1]


def test_shared_map_filters_selection_and_invalidation(workspace, monkeypatch):
    """Keep ID selection across view edits and never reuse stale prediction data."""
    calls = []
    verify = spatial_explorer.verify_prediction_join
    read = Workspace.read

    def record_join(workspace, cycle):
        calls.append(cycle)
        return verify(workspace, cycle)

    def unexpected_read(*args):
        pytest.fail("The spatial map must not read hidden diagnostic files.")

    monkeypatch.setattr(spatial_explorer, "verify_prediction_join", record_join)
    monkeypatch.setattr(Workspace, "read", unexpected_read)
    app = _start(workspace, monkeypatch)
    _select(app, "Result section", "Spatial predictions")
    assert _caption(app, "Verified prediction alignment")
    population = app.session_state["_spatial_session"].population
    picked = population.sample_ids[0]
    _select(app, "Observation ID", picked)
    _select(app, "Color by", ColorMode.ENTROPY)
    _widget(app, "segmented_control", "Color scale").select("Population range").run()
    assert _caption(app, "Colors compare values within this population only.")
    _widget(app, "segmented_control", "Probability family").select(
        ProbabilityFamily.UNCALIBRATED
    ).run()
    assert _widget(app, "selectbox", "Observation ID").value == picked
    assert _spec(app)["data"][-1]["name"] == "Selected observation"
    z = population.coordinates[:, 2]
    _widget(app, "slider", "Vertical range (Z)").set_value(
        (float(z.min()), float(np.median(z)))
    ).run()
    assert _caption(app, "8 visible / 16")
    _widget(app, "multiselect", "Observed classes").set_value([]).run()
    assert not app.get("plotly_chart")
    assert _caption(app, "0 visible / 16")
    assert any("outside the current display filters" in item.value for item in app.info)
    _widget(app, "toggle", "Observation table").set_value(True).run()
    assert app.dataframe[0].value.empty
    _widget(app, "toggle", "Prediction downloads").set_value(True).run()
    assert len(app.get("download_button")) == 3
    assert calls == [1]
    csv = workspace.results / "prediction/cycle_1/predictions.csv"
    csv.write_bytes(csv.read_bytes() + b"\n")
    app.run()
    assert calls == [1, 1]
    assert not app.get("download_button")
    assert any("Prediction alignment unavailable" in item.value for item in app.warning)
    monkeypatch.setattr(Workspace, "read", read)
    _widget(app, "segmented_control", "Spatial view").select(
        "Original diagnostics"
    ).run()
    assert (
        "Incoming-batch predictions table"
        in _widget(app, "selectbox", "Diagnostic").options
    )


def test_data_routing_bootstrap_cv_membership_and_historical(workspace, monkeypatch):
    """Use cycle lineage for populations and retain an explicit historical route."""
    app = _start(workspace, monkeypatch)
    _select(app, "Learning cycle", 0)
    assert _caption(app, "Bootstrap · snapshot 0")
    _select(app, "Learning cycle", 2)
    _select(app, "Population", "CV validation")
    _select(app, "CV fold", 1)
    assert app.session_state["_spatial_session"].population.fold == 1
    _select(app, "Population", "Post-append base · snapshot 2")
    _select(app, "Color by", ColorMode.MEMBERSHIP)
    assert _caption(app, "128 visible / 128")
    assert [trace["name"] for trace in _spec(app)["data"]] == [
        "Input base",
        "Incoming batch",
    ]
    _select(
        app,
        "Observation ID",
        app.session_state["_spatial_session"].population.sample_ids[-1],
    )
    _select(app, "Population", "Development membership")
    assert _widget(app, "selectbox", "Observation ID").value is None
    assert _widget(app, "checkbox", "Show graph edges").disabled
    historical = workspace.results / "prediction/cycle_3/predictions.csv"
    historical.parent.mkdir()
    historical.write_text("label\n1\n")
    app.run()
    _select(app, "Learning cycle", 3)
    _select(app, "Result section", "Spatial predictions")
    assert any("outside the applied scope" in item.value for item in app.warning)
    _widget(app, "segmented_control", "Spatial view").select(
        "Original diagnostics"
    ).run()
    assert _caption(app, "Standalone prediction CSV")
    workspace.params.write_text(
        workspace.params.read_text().replace("cycles: 2", "cycles: 0")
    )
    _select(app, "Learning cycle", 0)
    assert _caption(app, "Bootstrap · snapshot 0")


def test_edges_limit_leaves_points_and_original_statistics(workspace, monkeypatch):
    """Keep every point visible and withhold all links above the display cap."""
    monkeypatch.setattr(spatial_explorer, "LINK_LIMIT", 1)
    app = _start(workspace, monkeypatch)
    _select(app, "Population", "Incoming batch 1")
    _widget(app, "checkbox", "Show graph edges").check().run()
    assert any("1-link display limit" in item.value for item in app.info)
    assert _caption(app, "16 visible / 16")
    assert all(trace["mode"] == "markers" for trace in _spec(app)["data"])
    _widget(app, "toggle", "Graph statistics").set_value(True).run()
    assert _caption(app, "Original graph · 16 nodes")


def test_deferred_downloads_require_current_snapshot(workspace):
    """Round-trip IDs and coordinates and refuse elapsed changes at download time."""
    joined = verify_prediction_join(workspace, 1).joined
    assert joined
    full = prediction_download(workspace, joined)
    table = pd.read_csv(BytesIO(full), float_precision="round_trip")
    assert tuple(table["sample_id"]) == joined.population.sample_ids
    np.testing.assert_array_equal(table[["x", "y", "z"]], joined.population.coordinates)
    view = joined.population.view(class_ids={0})
    filtered = pd.read_csv(BytesIO(prediction_download(workspace, joined, view=view)))
    assert filtered["sample_id"].tolist() == [
        joined.population.sample_ids[row] for row in view.row_indices
    ]
    csv = workspace.results / "prediction/cycle_1/predictions.csv"
    assert prediction_download(workspace, joined, original=True) == csv.read_bytes()
    csv.write_bytes(csv.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="evidence changed"):
        prediction_download(workspace, joined, view=view)


@pytest.mark.integration
@pytest.mark.parametrize("cycle", [1, 2])
def test_retained_spatial_maps(cycle, tmp_path, monkeypatch):
    """Open both genuine prediction populations through the actual Results page."""
    configured = os.environ.get("MINERAL_QUICK_DEMO_WORKSPACE")
    if not configured:
        pytest.skip("Set MINERAL_QUICK_DEMO_WORKSPACE to the retained Quick demo.")
    root = Path(configured).expanduser()
    for name in ("params.yaml", "dvc.yaml", "dvc.lock", ".dvcignore"):
        if (root / name).exists():
            shutil.copy2(root / name, tmp_path / name)
    for name in ("src", "results"):
        shutil.copytree(root / name, tmp_path / name)
    (tmp_path / ".dvc").mkdir()
    shutil.copy2(root / ".dvc/config", tmp_path / ".dvc/config")
    app = _start(Workspace(tmp_path), monkeypatch)
    _select(app, "Learning cycle", cycle)
    _select(app, "Result section", "Spatial predictions")
    assert _caption(app, "100 visible / 100")
    assert _caption(app, "Verified prediction alignment")
    _select(app, "Color by", ColorMode.MAXIMUM)
    assert _spec(app)["data"][0]["marker"]["cmin"] == 0
    assert _spec(app)["data"][0]["marker"]["cmax"] == 1
    assert not app.exception
