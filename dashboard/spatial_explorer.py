"""Shared Streamlit map, observation inspector and verified downloads."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.artifacts import Workspace
from dashboard.dvc_evidence import WorkspaceStamp
from dashboard.spatial_data import (
    Connectivity,
    PopulationKind,
    SpatialPopulation,
    SpatialView,
    population_paths,
    read_population,
)
from dashboard.spatial_figure import (
    CLASS_COLORS,
    LINK_LIMIT,
    MEMBERSHIP_NAMES,
    POINT_LIMIT,
    SCORE_MODES,
    ColorMode,
    color_field,
    observation_label,
    probability_figure,
    spatial_figure,
)
from dashboard.spatial_predictions import (
    JoinedPredictions,
    PredictionJoin,
    ProbabilityFamily,
    verify_prediction_join,
)
from src.configuration_files import configuration_operation

_CV = (PopulationKind.CV_TRAIN, PopulationKind.CV_VALIDATION)


@dataclass(frozen=True)
class SpatialScope:
    """Identify the selected display population and its original diagnostic directory."""

    kind: PopulationKind
    cycle: int
    label: str
    fold: int | None = None

    @property
    def directory(self) -> Path:
        """Locate the original figures for the selected provenance scope."""
        if self.kind == PopulationKind.INCOMING:
            return Path(f"data/prediction/cycle_{self.cycle}")
        snapshot = (
            self.cycle
            if self.kind
            in (
                PopulationKind.SNAPSHOT,
                PopulationKind.POST_APPEND,
                PopulationKind.FINAL_APPEND,
            )
            else self.cycle - 1
        )
        return Path(f"data/base/cycle_{snapshot}")


def choose_population(
    cycle: int,
    *,
    declared: int | None,
    folds: int,
    predictions: bool,
) -> SpatialScope:
    """Route the shared cycle to one raw population or incoming prediction batch."""
    if predictions:
        return SpatialScope(PopulationKind.INCOMING, cycle, f"Incoming batch {cycle}")
    if cycle == 0:
        return SpatialScope(PopulationKind.SNAPSHOT, 0, "Bootstrap · snapshot 0")
    choices = {
        f"Input base · snapshot {cycle - 1}": PopulationKind.INPUT_BASE,
        "Development membership": PopulationKind.DEVELOPMENT,
        "Calibration partition": PopulationKind.CALIBRATION,
        "Final-test partition": PopulationKind.FINAL_TEST,
        "CV training": PopulationKind.CV_TRAIN,
        "CV validation": PopulationKind.CV_VALIDATION,
        f"Incoming batch {cycle}": PopulationKind.INCOMING,
        f"Post-append base · snapshot {cycle}": (
            PopulationKind.FINAL_APPEND
            if cycle == declared
            else PopulationKind.POST_APPEND
        ),
    }
    label = st.selectbox(
        "Population", choices, key=f"spatial-population:{cycle}", width=315
    )
    kind = choices[label]
    fold = None
    if kind in _CV:
        fold = st.selectbox(
            "CV fold", range(folds), format_func=lambda index: f"Fold {index + 1}"
        )
    return SpatialScope(kind, cycle, label, fold)


@dataclass
class _SpatialSession:
    scope: tuple | None = None
    stamp: WorkspaceStamp | None = None
    population: SpatialPopulation | None = None
    join: PredictionJoin | None = None
    error: str | None = None
    selected: str | None = None


def _snapshot(
    workspace: Workspace,
    scope: SpatialScope,
    *,
    predictions: bool,
    refresh: bool,
) -> _SpatialSession:
    root = workspace.root.resolve()
    key = (root, scope, predictions)
    state = st.session_state.get("_spatial_session")
    if state is None or state.scope != key:
        state = st.session_state["_spatial_session"] = _SpatialSession(scope=key)
    paths = population_paths(scope.kind, scope.cycle)
    with configuration_operation(root / "params.yaml"):
        stamp = WorkspaceStamp.capture(root, paths)
        join_changed = bool(
            state.join
            and state.join.joined
            and not state.join.joined.still_observed(workspace)
        )
    if state.stamp != stamp or refresh or join_changed:
        state.population = state.join = state.error = None
        state.stamp = stamp
    if predictions and state.join is None:
        with st.spinner("Verifying prediction alignment…"):
            state.join = verify_prediction_join(workspace, scope.cycle)
    if state.join is not None and state.join.joined is not None:
        state.population = state.join.joined.population
    elif state.population is None and state.error is None:
        try:
            state.population = read_population(
                workspace, scope.kind, cycle=scope.cycle, fold=scope.fold
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            EOFError,
            UnpicklingError,
        ) as error:
            state.error = str(error)
    return state


def _open(label: str, key: str):
    return st.expander(label, key=key, on_change="rerun", type="compact")


def _settings(
    population: SpatialPopulation,
    joined: JoinedPredictions | None,
    key: str,
) -> tuple[bool, ColorMode, ProbabilityFamily, int, SpatialView, bool, bool]:
    controls = st.container(
        key="spatial-controls",
        horizontal=True,
        gap="small",
        vertical_alignment="bottom",
    )
    with controls:
        perspective = st.segmented_control(
            "Perspective",
            ["XY plan", "3D"],
            default="XY plan",
            required=True,
            key=f"{key}:perspective",
        )
        modes = [ColorMode.OBSERVED, ColorMode.FEATURE]
        if len(set(population.membership)) > 1:
            modes.append(ColorMode.MEMBERSHIP)
        if joined:
            modes = [
                ColorMode.PREDICTED,
                ColorMode.OBSERVED,
                ColorMode.MAXIMUM,
                ColorMode.PROBABILITY,
                ColorMode.ENTROPY,
                ColorMode.CORRECTNESS,
                ColorMode.FEATURE,
            ]
        mode = st.selectbox("Color by", modes, key=f"{key}:color", width=240)
        family = ProbabilityFamily.CALIBRATED
        if joined:
            family = st.segmented_control(
                "Probability family",
                list(ProbabilityFamily),
                format_func=str.capitalize,
                default=ProbabilityFamily.CALIBRATED,
                required=True,
                key=f"{key}:family",
            )
        column = 0
        if mode == ColorMode.FEATURE:
            column = st.selectbox(
                "Raw feature",
                range(len(population.feature_names)),
                format_func=population.feature_names.__getitem__,
                key=f"{key}:feature",
            )
        elif mode == ColorMode.PROBABILITY:
            column = st.selectbox(
                "Probability class",
                range(len(population.class_names)),
                format_func=population.class_names.__getitem__,
                key=f"{key}:class",
            )
        edges = st.checkbox(
            "Show graph edges", disabled=population.topology is None, key=f"{key}:edges"
        )
        relative = False
        if mode in SCORE_MODES:
            scale = st.segmented_control(
                "Color scale",
                ["Absolute", "Population range"],
                default="Absolute",
                required=True,
                key=f"{key}:scale",
            )
            relative = scale == "Population range"
        filters = st.popover("Display filters", key=f"{key}:filters")
    # Mounted even while closed so active filters survive closing the panel.
    with filters:
        z = population.coordinates[:, 2]
        lower, upper = float(z.min()), float(z.max())
        depth = None
        if lower < upper:
            bounds = st.slider(
                "Vertical range (Z)",
                lower,
                upper,
                (lower, upper),
                step=(upper - lower) / 200,
                format="%.3f",
                key=f"{key}:depth",
            )
            if bounds != (lower, upper):
                depth = bounds
        else:
            st.caption(f"All observations have Z = {lower:.5g}.")
        classes = tuple(range(len(population.class_names)))
        if joined:
            st.caption(
                "Class filters use observed labels, including when the map is colored by predictions."
            )
        selected = st.multiselect(
            "Observed classes",
            classes,
            default=classes,
            format_func=population.class_names.__getitem__,
            key=f"{key}:class-filter",
        )
    view = population.view(
        depth=depth, class_ids=selected if set(selected) != set(classes) else None
    )
    return perspective == "3D", mode, family, column, view, edges, relative


def _plot_selection(
    state: _SpatialSession, plot_key: str, picker_key: str, visible: tuple[str, ...]
) -> None:
    event = st.session_state[plot_key]
    for point in reversed(event["selection"]["points"]):
        custom = point.get("customdata")
        if isinstance(custom, (tuple, list)) and custom and custom[0] in visible:
            state.selected = custom[0]
            st.session_state[picker_key] = state.selected
            break


def _remember_id(state: _SpatialSession, picker_key: str) -> None:
    state.selected = st.session_state[picker_key]


def _population_totals(values, size: int, view: SpatialView):
    """Count every category slot for the full and displayed populations."""
    return (
        np.bincount(values, minlength=size),
        np.bincount(values[view.row_indices], minlength=size),
    )


def _population_summary(
    view: SpatialView,
    joined: JoinedPredictions | None,
    family: ProbabilityFamily,
    mode: ColorMode,
    key: str,
) -> None:
    population = view.population
    choices = ["Observed classes"]
    if joined and mode in (*SCORE_MODES, ColorMode.PREDICTED, ColorMode.CORRECTNESS):
        choices.append(
            "Correctness" if mode == ColorMode.CORRECTNESS else "Predicted classes"
        )
    summary = choices[0]
    if len(choices) > 1:
        summary = st.segmented_control(
            "Population breakdown",
            choices,
            default=choices[-1],
            required=True,
            label_visibility="collapsed",
            key=f"{key}:summary:{mode}",
        )
    names, values, colors = (
        population.class_names,
        population.observed_class_ids,
        CLASS_COLORS,
    )
    if summary == "Predicted classes":
        values = joined.scores(family).predicted_class_ids()
    elif summary == "Correctness":
        names, values = ("Incorrect", "Correct"), joined.correctness(family).astype(int)
        colors = ("#c45457", "#208a78")
    first, displayed = _population_totals(values, len(names), view)
    shown_header = "<th scope='col'>Shown</th>" if view.filtered else ""
    rows = []
    for index, (name, total, shown) in enumerate(
        zip(names, first, displayed, strict=True)
    ):
        portion = total / population.count
        color = colors[index % len(colors)]
        shown_cell = f"<td>{shown:,}</td>" if view.filtered else ""
        rows.append(
            f"<tr><th scope='row'>{escape(name)}"
            f"<span class='spatial-pop-bar' aria-hidden='true'><i style='width:{portion:.2%};background:{color}'></i></span></th>"
            f"<td>{total:,}</td><td>{portion:.1%}</td>{shown_cell}</tr>"
        )
    label = (
        summary
        if summary == "Observed classes"
        else f"{family.capitalize()} {summary.lower()}"
    )
    st.html(
        f"<table class='spatial-population-table'><caption>{label}</caption>"
        f"<thead><tr><th scope='col'>Class</th><th scope='col'>Total</th>"
        f"<th scope='col'>Share</th>{shown_header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _find_stable_id(
    state: _SpatialSession, picker: str, search: str, ids: tuple[str, ...]
) -> None:
    submitted = st.session_state[search].strip()
    if submitted in ids:
        state.selected = submitted
        st.session_state[picker] = submitted


def _inspector(
    state: _SpatialSession,
    view: SpatialView,
    joined: JoinedPredictions | None,
    family: ProbabilityFamily,
    mode: ColorMode,
    key: str,
    dark: bool,
) -> None:
    population = view.population
    id_rows = {sample_id: row for row, sample_id in enumerate(population.sample_ids)}
    st.markdown(
        "#### Population summary"
        if state.selected is None
        else "#### Observation inspector"
    )
    if state.selected is None:
        st.caption(f"**{population.count:,} total** · **{view.count:,} displayed**")
        _population_summary(view, joined, family, mode, key)
    st.selectbox(
        "Observation ID",
        population.sample_ids,
        index=None,
        format_func=lambda sample_id: observation_label(
            id_rows[sample_id], population.count
        ),
        placeholder="Choose an observation",
        key=f"{key}:observation",
        on_change=_remember_id,
        args=(state, f"{key}:observation"),
    )
    search = _open("Stable ID lookup and details", f"{key}:sample-details")
    if search.open:
        with search:
            search_key = f"{key}:stable-id-search"
            submitted = st.text_input(
                "Find by full stable ID",
                key=search_key,
                placeholder="Paste an exact sample ID",
                on_change=_find_stable_id,
                args=(state, f"{key}:observation", search_key, population.sample_ids),
            ).strip()
            if submitted and submitted not in id_rows:
                st.caption("That stable ID is outside this population.")
            if state.selected is not None:
                row = id_rows[state.selected]
                x, y, z = population.coordinates[row]
                role = population.membership[row]
                st.code(state.selected, language=None, wrap_lines=True)
                st.caption(f"X {x:,.3f} · Y {y:,.3f} · Z {z:,.3f}")
                st.caption(
                    f"Membership · {MEMBERSHIP_NAMES.get(role, role.replace('_', ' ').capitalize())}"
                )
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Feature": population.feature_names,
                            "Value": population.raw_features[row],
                        }
                    ),
                    hide_index=True,
                    height=160,
                )
    if state.selected is None:
        st.caption("Pick an observation above or click a point in XY.")
        return
    row = id_rows[state.selected]
    if row not in view.row_indices:
        st.info("This observation is outside the current display filters.")
        return
    label = population.observed_class_names[row]
    with st.container(key="spatial-observation-detail", gap="xxsmall"):
        st.markdown(f"Observed · **{label}**")
        if joined:
            scores = joined.scores(family)
            prediction = joined.predicted_class_names(family)[row]
            matched = joined.correctness(family)[row]
            badge = ":green-badge[Correct]" if matched else ":red-badge[Incorrect]"
            st.markdown(f"Predicted · **{prediction}** · {badge}")
            st.caption(
                f"**{scores.maximum_probability()[row]:.1%}** max probability · "
                f"**{scores.entropy_nats[row]:.3f} nats** entropy · {family.capitalize()}"
            )
            st.plotly_chart(
                probability_figure(
                    population.class_names, scores.probabilities[row], dark=dark
                ),
                theme=None,
                key=f"{key}:probability-bars",
                config={"displayModeBar": False},
            )


def _observation_table(
    view: SpatialView, joined: JoinedPredictions | None, family: ProbabilityFamily
) -> pd.DataFrame:
    population = view.population
    data = {
        "Sample ID": population.sample_ids,
        **dict(zip(("X", "Y", "Z"), population.coordinates.T, strict=True)),
        "Observed class": population.observed_class_names,
        "Observed class ID": population.observed_class_ids,
        "Membership": [
            MEMBERSHIP_NAMES.get(role, role.replace("_", " "))
            for role in population.membership
        ],
    }
    if joined:
        scores = joined.scores(family)
        data.update(
            {
                "Predicted class": joined.predicted_class_names(family),
                "Predicted class ID": scores.predicted_class_ids(),
                "Maximum probability": scores.maximum_probability(),
                "Predictive entropy (nats)": scores.entropy_nats,
                "Correct": joined.correctness(family),
            }
        )
    return pd.DataFrame(data).iloc[view.row_indices]


def prediction_download(
    workspace: Workspace,
    joined: JoinedPredictions,
    *,
    original: bool = False,
    view: SpatialView | None = None,
) -> bytes:
    """Check current correspondence again at the time a download is requested."""
    with configuration_operation(workspace.params):
        if not joined.still_observed(workspace):
            raise ValueError(
                "Workspace evidence changed. Verify the prediction join again."
            )
        if original:
            path = Path(f"prediction/cycle_{joined.cycle}/predictions.csv")
            payload = workspace.read_bytes(path)
            if sha256(payload).hexdigest() != joined.producer_csv_sha256:
                raise ValueError(
                    "The producer CSV changed. Verify the prediction join again."
                )
        else:
            payload = joined.export(view).csv_bytes
        if not joined.still_observed(workspace):
            raise ValueError("Workspace evidence changed during download preparation.")
        return payload


def _secondary(
    workspace: Workspace,
    scope: SpatialScope,
    view: SpatialView,
    joined: JoinedPredictions | None,
    family: ProbabilityFamily,
    key: str,
) -> None:
    row = st.container(horizontal=True, gap="small")
    with row:
        table = st.toggle("Observation table", key=f"{key}:table")
        downloads = (
            st.toggle("Prediction downloads", key=f"{key}:downloads")
            if joined
            else False
        )
        topology = view.population.topology
        statistics = (
            st.toggle("Graph statistics", key=f"{key}:statistics")
            if topology
            else False
        )
    if table:
        if joined:
            st.caption(f"Visible observations · {family.capitalize()} scores")
        st.dataframe(
            _observation_table(view, joined, family), hide_index=True, height=290
        )
    if statistics:
        stats = topology.statistics
        st.caption(
            f"Original graph · {stats.node_count:,} nodes · {stats.undirected_edge_count:,} physical links · "
            f"{stats.self_loop_count:,} self-loops · {stats.isolated_node_count:,} isolated nodes"
        )
        st.json(asdict(stats), expanded=False)
    if downloads and joined:
        with st.container(horizontal=True, gap="small"):
            st.download_button(
                "Original producer CSV",
                lambda: prediction_download(workspace, joined, original=True),
                "predictions.csv",
                "text/csv",
                on_click="ignore",
            )
            st.download_button(
                f"Full verified CSV · {view.population.count:,} rows",
                lambda: prediction_download(workspace, joined),
                f"cycle-{scope.cycle}-verified-full.csv",
                "text/csv",
                on_click="ignore",
            )
            if view.filtered:
                st.download_button(
                    f"Filtered verified CSV · {view.count:,} rows",
                    lambda: prediction_download(workspace, joined, view=view),
                    f"cycle-{scope.cycle}-verified-filtered.csv",
                    "text/csv",
                    on_click="ignore",
                )
            else:
                st.caption("Apply a display filter to download just the visible rows.")


def render_spatial(
    workspace: Workspace,
    scope: SpatialScope,
    *,
    predictions: bool,
    historical: bool,
    refresh: bool,
) -> None:
    """Render one shared map with snapshot-bound scores and accessible inspection."""
    if historical:
        st.warning(
            "This cycle is outside the applied scope. Open Original diagnostics to inspect the historical files."
        )
        return
    retry = st.session_state.get("_spatial_session")
    if (
        predictions
        and retry
        and retry.scope == (workspace.root.resolve(), scope, predictions)
        and retry.join
        and retry.join.joined is None
    ):
        refresh = st.button("Retry prediction verification") or refresh
    try:
        state = _snapshot(workspace, scope, predictions=predictions, refresh=refresh)
    except (OSError, RuntimeError, ValueError) as error:
        st.warning(str(error))
        return
    joined = state.join.joined if state.join else None
    if joined is not None and not joined.still_observed(workspace):
        state.join = state.population = None
        st.warning(
            "Workspace evidence changed. Refresh before opening the prediction map again."
        )
        return
    if state.join and joined is None:
        st.warning(
            f"Prediction alignment unavailable · {state.join.evidence.correspondence}. "
            "The original CSV is available under Original diagnostics when present."
        )
        details = _open("Alignment evidence", "spatial-join-reasons")
        if details.open:
            with details:
                for reason in state.join.evidence.reasons:
                    st.text(reason)
    if state.population is None:
        st.warning(
            "No spatial observations available for this selection. The output may be absent or partial."
        )
        if state.error:
            st.caption(state.error)
        return
    population = state.population
    if state.selected not in population.sample_ids:
        state.selected = None
    st.html("""
        <style>
        .st-key-spatial-canvas [data-testid="stColumn"] { min-width: 0; }
        .st-key-spatial-canvas .st-key-spatial-inspector {
            border-left: 1px solid color-mix(in srgb, currentColor 18%, transparent);
            padding-left: 1rem;
        }
        .spatial-population-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
        .spatial-population-table caption { text-align: left; font-weight: 650; margin-bottom: .4rem; }
        .spatial-population-table th, .spatial-population-table td {
            padding: .3rem .4rem; border-bottom: 1px solid color-mix(in srgb, currentColor 14%, transparent);
        }
        .spatial-population-table th:first-child { text-align: left; padding-left: 0; }
        .spatial-population-table th:not(:first-child), .spatial-population-table td {
            text-align: right; font-variant-numeric: tabular-nums;
        }
        .spatial-pop-bar {
            display:block; height: 3px; margin-top: .2rem;
            background:color-mix(in srgb, currentColor 12%, transparent);
        }
        .spatial-pop-bar i { display:block; height: 100%; }
        @media (max-width: 700px) {
            .st-key-spatial-canvas [data-testid="stHorizontalBlock"] { flex-direction: column; }
            .st-key-spatial-canvas [data-testid="stColumn"] { width: 100% !important; flex: 1 1 auto !important; }
            .st-key-spatial-canvas .st-key-spatial-inspector {
                padding: .75rem 0 0; border-left: 0;
                border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent);
            }
        }
        </style>
    """)
    key = f"spatial:{scope.kind}:{scope.cycle}:{scope.fold}:{population.dataset_sha256[:12]}"
    control_key = f"{key}:verified" if joined else f"{key}:raw"
    three_d, mode, family, column, view, edges, relative = _settings(
        population, joined, control_key
    )
    topology = population.topology
    connectivity = (
        "Descriptive full-base connectivity"
        if topology and topology.kind == Connectivity.DESCRIPTIVE
        else "Independent partition graph"
        if topology
        else "Membership only"
    )
    identity = (
        f"**{scope.label}** · :green-badge[Verified prediction alignment] · "
        if joined
        else f"**{scope.label}** · "
        if scope.cycle == 0
        else ""
    )
    st.caption(
        f"{identity}**{view.count:,} visible / {population.count:,} observations**"
        f"  \n{connectivity} · Simulated local Cartesian coordinates"
    )
    field = color_field(
        population,
        mode,
        joined=joined,
        family=family,
        column=column,
        population_range=relative,
    )
    if mode == ColorMode.FEATURE and field.bounds:
        low, high = field.bounds
        st.caption(
            f"{field.title} · Full-population bounds {low:.5g} to {high:.5g}"
            + (" · Constant feature" if low == high else "")
        )
    if mode in SCORE_MODES:
        low, high = field.bounds
        scale = "Population range" if relative else "Absolute scale"
        limitation = (
            " · Colors compare values within this population only." if relative else ""
        )
        constant = " · Constant value" if low == high else ""
        st.caption(
            f"**{scale}** · {field.title} {low:.6g} to {high:.6g}{constant}{limitation}"
        )
    dark = st.context.theme.type == "dark"
    links = np.empty((0, 2), dtype=np.int64)
    if edges and topology and view.count <= POINT_LIMIT:
        links = view.edge_rows()
        if len(links) > LINK_LIMIT:
            st.info(
                f"{len(links):,} visible links exceed the {LINK_LIMIT:,}-link display limit. Narrow the filters to show every visible link."
            )
            links = links[:0]
        else:
            st.caption(
                f"{len(links):,} visible physical links · {topology.statistics.undirected_edge_count:,} in the original graph"
            )
    with st.container(key="spatial-canvas"):
        map_column, inspector_column = st.columns([1.6, 1], gap="medium")
        picker_key = f"{key}:observation"
        if picker_key not in st.session_state:
            st.session_state[picker_key] = state.selected
        with map_column:
            if not view.count:
                st.info("No observations match the display filters.")
            elif view.count > POINT_LIMIT:
                st.info(
                    f"The map supports up to {POINT_LIMIT:,} visible observations. Narrow the filters to display the whole selection."
                )
            else:
                plotted = spatial_figure(
                    view,
                    field,
                    three_d=three_d,
                    dark=dark,
                    links=links,
                    selected=state.selected,
                    revision=f"{key}:{three_d}",
                    predicted=joined.predicted_class_names(family) if joined else None,
                )
                plot_key = f"{key}:map:{three_d}"
                visible = tuple(population.sample_ids[row] for row in view.row_indices)
                st.plotly_chart(
                    plotted,
                    theme=None,
                    key=plot_key,
                    selection_mode="points",
                    on_select="ignore"
                    if three_d
                    else lambda: _plot_selection(state, plot_key, picker_key, visible),
                    config={
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                )
        with inspector_column, st.container(key="spatial-inspector"):
            _inspector(state, view, joined, family, mode, key, dark)
    _secondary(workspace, scope, view, joined, family, key)
