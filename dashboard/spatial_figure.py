"""Plot only stored observations, scores and links in the shared spatial explorer."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from dashboard.spatial_data import SpatialPopulation, SpatialView
from dashboard.spatial_predictions import JoinedPredictions, ProbabilityFamily

POINT_LIMIT = 10_000
LINK_LIMIT = 8_000
CLASS_COLORS = ("#2378c4", "#e38a26", "#28936a", "#9364c7", "#c25370", "#188997")
CLASS_SYMBOLS = ("circle", "diamond", "square", "cross", "x", "circle-open")
MEMBERSHIP_NAMES = {"input_base": "Input base", "incoming_batch": "Incoming batch"}


class ColorMode(StrEnum):
    """Supported observation-level color encodings."""

    OBSERVED = "Observed class"
    FEATURE = "Raw feature"
    MEMBERSHIP = "Dataset membership"
    PREDICTED = "Predicted class"
    MAXIMUM = "Maximum probability"
    PROBABILITY = "Selected-class probability"
    ENTROPY = "Predictive entropy"
    CORRECTNESS = "Classification correctness"


SCORE_MODES = (ColorMode.MAXIMUM, ColorMode.PROBABILITY, ColorMode.ENTROPY)


@dataclass(frozen=True)
class ColorField:
    """Full-population values and their categorical or continuous encoding."""

    title: str
    values: NDArray
    categories: tuple[str, ...] = ()
    colors: tuple[str, ...] = CLASS_COLORS
    symbols: tuple[str, ...] = CLASS_SYMBOLS
    scale: str = "Viridis"
    bounds: tuple[float, float] | None = None


def color_field(
    population: SpatialPopulation,
    mode: ColorMode,
    *,
    joined: JoinedPredictions | None = None,
    family: ProbabilityFamily = ProbabilityFamily.CALIBRATED,
    column: int = 0,
    population_range: bool = False,
) -> ColorField:
    """Resolve a color field from the population or one verified probability family."""
    if mode == ColorMode.OBSERVED:
        return ColorField(mode, population.observed_class_ids, population.class_names)
    if mode == ColorMode.MEMBERSHIP:
        categories = tuple(dict.fromkeys(population.membership))
        return ColorField(
            mode,
            np.array([categories.index(item) for item in population.membership]),
            tuple(MEMBERSHIP_NAMES.get(item, item) for item in categories),
        )
    if mode == ColorMode.FEATURE:
        values = population.raw_features[:, column]
        return ColorField(
            population.feature_names[column],
            values,
            scale="Blues",
            bounds=(float(values.min()), float(values.max())),
        )
    if joined is None or joined.population is not population:
        raise ValueError("This color field requires the verified prediction join.")
    scores = joined.scores(family)
    if mode == ColorMode.PREDICTED:
        return ColorField(mode, scores.predicted_class_ids(), population.class_names)
    if mode == ColorMode.CORRECTNESS:
        return ColorField(
            mode,
            joined.correctness(family).astype(int),
            ("Incorrect", "Correct"),
            ("#c45457", "#208a78"),
            ("x", "circle"),
        )
    if mode == ColorMode.ENTROPY:
        title, values, scale = "Entropy (nats)", scores.entropy_nats, "Viridis"
        bounds = (0, float(np.log(len(population.class_names))))
    elif mode == ColorMode.MAXIMUM:
        title, values, scale, bounds = (
            "Max probability",
            scores.maximum_probability(),
            "Cividis",
            (0, 1),
        )
    else:
        title, values = (
            f"P({population.class_names[column]})",
            scores.selected_probability(column),
        )
        scale, bounds = "Cividis", (0, 1)
    if population_range:
        bounds = (float(values.min()), float(values.max()))
    return ColorField(title, values, scale=scale, bounds=bounds)


def observation_label(row: int, count: int) -> str:
    """Format a one-based population ordinal for display only."""
    return f"Observation {row + 1:0{len(str(count))}d}"


def _plan_ranges(coords: NDArray) -> NDArray:
    low, high = coords[:, :2].min(axis=0), coords[:, :2].max(axis=0)
    center = (low + high) / 2
    half = float((high - low).max()) * 0.55 or 0.5
    return np.column_stack((center - half, center + half))


def _trace(coords: NDArray, three_d: bool, **kwargs):
    if three_d:
        return go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], **kwargs)
    return go.Scatter(x=coords[:, 0], y=coords[:, 1], **kwargs)


def spatial_figure(
    view: SpatialView,
    field: ColorField,
    *,
    three_d: bool,
    dark: bool,
    links: NDArray,
    selected: str | None,
    revision: str,
    predicted: tuple[str, ...] | None = None,
) -> go.Figure:
    """Construct an equal-scale XY map or a physically proportioned 3D scene."""
    population = view.population
    rows = view.row_indices
    coords = population.coordinates
    names = population.observed_class_names
    if names is None:
        names = ("Unavailable",) * population.count
    ink = "#e1eaf3" if dark else "#233a51"
    grid = "#344354" if dark else "#d6e3ec"
    panel = "#141e2a" if dark else "#f1f7fb"
    outline = "#afc4d7" if dark else "#314b62"
    fig = go.Figure()
    if len(links):
        lines = np.full((len(links), 3, 3), np.nan)
        lines[:, :2] = coords[links]
        fig.add_trace(
            _trace(
                lines.reshape(-1, 3),
                three_d,
                mode="lines",
                line={
                    "width": 1.5 if three_d else 1,
                    "color": "rgba(151, 183, 208, 0.55)"
                    if dark
                    else "rgba(57, 86, 110, 0.47)",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def markers(subset, marker, name, *, legend=False):
        custom = [
            (
                population.sample_ids[row],
                coords[row, 2],
                names[row],
                predicted[row] if predicted is not None else "",
                field.categories[int(field.values[row])]
                if field.categories
                else f"{field.values[row]:.5g}",
                observation_label(row, population.count),
            )
            for row in subset
        ]
        hover = (
            "<b>%{customdata[5]}</b><br>X %{x:.3f} · Y %{y:.3f} · Z %{customdata[1]:.3f}<br>"
            "Observed: %{customdata[2]}<br>"
            + ("Predicted: %{customdata[3]}<br>" if predicted is not None else "")
            + f"{field.title}: %{{customdata[4]}}<extra></extra>"
        )
        fig.add_trace(
            _trace(
                coords[subset],
                three_d,
                mode="markers",
                name=name,
                showlegend=legend,
                marker={
                    "size": (3.8 if marker.get("symbol") == "x" else 5)
                    if three_d
                    else 10,
                    "line": {"color": outline, "width": 0.85},
                    **marker,
                },
                customdata=custom,
                hovertemplate=hover,
                **({} if three_d else {"unselected": {"marker": {"opacity": 1}}}),
            )
        )

    if field.categories:
        for index, category in enumerate(field.categories):
            subset = rows[field.values[rows] == index]
            if len(subset):
                markers(
                    subset,
                    {
                        "color": field.colors[index % len(field.colors)],
                        "symbol": field.symbols[index % len(field.symbols)],
                    },
                    category,
                    legend=True,
                )
    else:
        low, high = field.bounds
        marker = (
            {"color": "#2378c4"}
            if low == high
            else {
                "color": field.values[rows],
                "colorscale": field.scale,
                "cmin": low,
                "cmax": high,
                "showscale": True,
                "colorbar": {
                    "title": {"text": field.title, "side": "right"},
                    "thickness": 12,
                    "len": 0.76,
                    "outlinewidth": 0,
                    "tickformat": ".3~g",
                },
            }
        )
        markers(rows, marker, field.title)
    if selected in population.sample_ids:
        index = population.sample_ids.index(selected)
        if index in rows:
            fig.add_trace(
                _trace(
                    coords[[index]],
                    three_d,
                    mode="markers",
                    name="Selected observation",
                    marker={
                        "size": 10 if three_d else 18,
                        "symbol": "circle-open",
                        "color": ink,
                        "line": {"width": 2.5, "color": ink},
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    axis = {"gridcolor": grid, "zeroline": False, "showgrid": True, "automargin": True}
    fig.update_layout(
        template="none",
        height=550,
        uirevision=revision,
        selectionrevision=revision,
        clickmode="event" if three_d else "event+select",
        font={"color": ink, "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=panel,
        margin={"l": 12, "r": 8, "t": 38, "b": 12},
        dragmode="orbit" if three_d else "pan",
        legend={
            "orientation": "h",
            "y": 1.015,
            "x": 0,
            "xanchor": "left",
            "yanchor": "bottom",
            "traceorder": "normal",
            "itemclick": False,
            "itemdoubleclick": False,
        },
    )
    if three_d:
        axis.pop("automargin")
        fig.update_layout(
            scene={
                "aspectmode": "data",
                "bgcolor": panel,
                "xaxis": {**axis, "title": "X", "showbackground": False},
                "yaxis": {**axis, "title": "Y", "showbackground": False},
                "zaxis": {
                    **axis,
                    "title": "Z",
                    "showbackground": False,
                    "autorange": True,
                },
                "camera": {
                    "eye": {"x": 1.5, "y": -1.9, "z": 1.65},
                    "projection": {"type": "orthographic"},
                },
            }
        )
    else:
        x_range, y_range = _plan_ranges(coords)
        fig.update_xaxes(title="X", range=x_range, constrain="domain", **axis)
        fig.update_yaxes(
            title="Y",
            range=y_range,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            **axis,
        )
    return fig


def probability_figure(
    names: tuple[str, ...], values: NDArray, *, dark: bool
) -> go.Figure:
    """Show the selected observation's probability distribution on an absolute scale."""
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[
                CLASS_COLORS[index % len(CLASS_COLORS)] for index in range(len(names))
            ],
            text=[f"{value:.1%}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=max(135, 29 * len(names) + 35),
        margin={"l": 0, "r": 36, "b": 25, "t": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 1], "tickformat": ".0%", "nticks": 3, "fixedrange": True},
        yaxis={"autorange": "reversed", "fixedrange": True, "automargin": True},
    )
    return figure
