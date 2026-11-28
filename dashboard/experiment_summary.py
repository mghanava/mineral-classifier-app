"""Project verified experiment evidence into population-aware overview data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from dashboard.provenance import Availability, Correspondence, Evidence, Verification

BLUE = "#0072B2"
ORANGE = "#BC591E"
PURPLE = "#8274AD"

if TYPE_CHECKING:
    from dashboard.scientific_summary import BasePopulation, CycleSummary


class Metric(StrEnum):
    """Selectable scientific metrics with their published population semantics."""

    ACCURACY = "Accuracy"
    MACRO_F05 = "Macro F0.5"
    MCC = "Matthews correlation coefficient"
    ECE = "Expected calibration error"
    MCE = "Maximum calibration error"

    @property
    def calibration_error(self) -> bool:
        """Identify errors published for the final-test calibration comparison."""
        return self in (Metric.ECE, Metric.MCE)


def is_matching(evidence: Evidence) -> bool:
    """Require both complete artifacts and matching correspondence."""
    return (
        evidence.availability == Availability.AVAILABLE
        and evidence.correspondence == Correspondence.MATCHING
    )


@dataclass(frozen=True)
class ExperimentSummary:
    """Keep applied positions, verified summaries and per-scope evidence together."""

    snapshots: dict[int, BasePopulation]
    learning: dict[int, CycleSummary]
    evidence: dict[int, Evidence]

    @classmethod
    def from_verification(cls, checked: Verification | None) -> ExperimentSummary:
        """Include only declared scopes and gate each snapshot by its own producer."""
        snapshots = {}
        learning = {}
        evidence = {}
        if checked is not None and checked.applied is not None:
            stages = dict(checked.stages)
            bootstrap = stages.get("bootstrap")
            if bootstrap is not None:
                evidence[0] = bootstrap
                if is_matching(bootstrap) and checked.bootstrap is not None:
                    snapshots[0] = checked.bootstrap
            for cycle in checked.cycles:
                if cycle.cycle not in checked.declared_cycles:
                    continue
                evidence[cycle.cycle] = cycle.evidence
                if is_matching(cycle.evidence) and cycle.matching_summary is not None:
                    learning[cycle.cycle] = cycle.matching_summary
                append = stages.get(f"prepare_next_cycle_data_{cycle.cycle}")
                snapshot = cycle.local_summary.post_append
                if append is not None and is_matching(append) and snapshot is not None:
                    snapshots[cycle.cycle] = snapshot
        return cls(snapshots, learning, evidence)

    @property
    def latest(self) -> int | None:
        """Choose the highest complete matching learning cycle."""
        return max(self.learning, default=None)


@dataclass(frozen=True)
class Series:
    """A named measurement and the sizes of its evaluation populations."""

    name: str
    values: list[float | int | None]
    counts: list[int | None]


def metric_series(
    summary: ExperimentSummary, metric: Metric, cycles: range
) -> tuple[Series, Series]:
    """Align the two population or calibration series on the declared cycle axis."""
    if metric.calibration_error:
        first = Series("Final test · after calibration", [], [])
        second = Series("Final test · before calibration", [], [])
    else:
        first = Series("Final test · calibrated", [], [])
        second = Series("Prediction batch · calibrated", [], [])
    for cycle in cycles:
        item = summary.learning.get(cycle)
        values = item.evaluation if item else None
        test_count = item.input_base.final_test if item and item.input_base else None
        if metric.calibration_error:
            field = "ece" if metric == Metric.ECE else "mce"
            first.values.append(
                getattr(values.after_calibration, field) if values else None
            )
            second.values.append(
                getattr(values.before_calibration, field) if values else None
            )
            second.counts.append(test_count)
        else:
            field = {
                Metric.ACCURACY: "accuracy",
                Metric.MACRO_F05: "macro_f05",
                Metric.MCC: "mcc",
            }[metric]
            first.values.append(
                getattr(values.after_calibration.classification, field)
                if values
                else None
            )
            second.values.append(
                getattr(item.prediction, field) if item and item.prediction else None
            )
            second.counts.append(item.incoming_batch if item else None)
        first.counts.append(test_count)
    return first, second


def _layout(figure: go.Figure, y_title: str, *, showlegend: bool) -> None:
    figure.update_layout(
        height=312,
        margin={"l": 12, "r": 14, "t": 12, "b": 8},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.02,
            "yanchor": "bottom",
            "font": {"size": 13},
            "traceorder": "normal",
        },
        showlegend=showlegend,
    )
    figure.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    figure.update_yaxes(title=y_title, showgrid=True, automargin=True)


def trajectory_figure(
    metric: Metric, cycles: range, series: tuple[Series, Series]
) -> go.Figure:
    """Draw bounded metric trajectories with explicit gaps and integer positions."""
    percent = metric == Metric.ACCURACY or metric.calibration_error
    sparse = len(cycles) <= 3
    figure = go.Figure()
    for item, color, dash, symbol in zip(
        series,
        (BLUE, ORANGE),
        ("solid", "dash"),
        ("circle", "diamond-open"),
        strict=True,
    ):
        figure.add_scatter(
            name=item.name,
            x=list(cycles),
            y=item.values,
            text=[
                (f"{value:.4%}" if percent else str(value))
                + (f" · n = {count:,}" if count is not None else "")
                if value is not None
                else ""
                for value, count in zip(item.values, item.counts, strict=True)
            ],
            hovertemplate=f"Cycle %{{x}}<br>%{{text}}<extra>{item.name}</extra>",
            mode="lines+markers",
            connectgaps=False,
            cliponaxis=False,
            line={"color": color, "width": 2.5, "dash": dash},
            marker={
                "symbol": symbol,
                "size": 11 if sparse else 7,
                "line": {"width": 2},
            },
        )
    _layout(
        figure,
        {
            Metric.ACCURACY: "Weighted accuracy (%)",
            Metric.MACRO_F05: "Macro F0.5",
            Metric.MCC: "MCC",
            Metric.ECE: "ECE (%)",
            Metric.MCE: "MCE (%)",
        }[metric],
        showlegend=True,
    )
    padding = 0.34
    figure.update_xaxes(
        range=[cycles.start - padding, cycles.stop - 1 + padding],
        title=None if sparse else "Learning cycle",
        tickmode="array" if sparse else "linear",
        tickvals=list(cycles) if sparse else None,
        ticktext=[f"Cycle {cycle}" for cycle in cycles] if sparse else None,
        tick0=cycles.start,
        dtick=max(1, (len(cycles) + 6) // 7),
    )
    upper = 1.0
    if metric.calibration_error:
        peak = max(
            (value for item in series for value in item.values if value is not None),
            default=0,
        )
        upper = min(1.0, max(0.1, peak * 1.25))
    figure.update_yaxes(
        range=[-1 if metric == Metric.MCC else 0, upper],
        tickformat=".0%" if percent else None,
    )
    return figure


def snapshot_label(index: int, cycles: int) -> str:
    """Distinguish bootstrap, intermediate and final appended base snapshots."""
    if index == 0:
        return "Bootstrap"
    if index == cycles:
        return f"Final append · cycle {index}"
    return f"After cycle {index}"


def growth_figure(
    summary: ExperimentSummary, cycles: int, *, composition: bool = False
) -> go.Figure:
    """Show verified base totals or partition counts on the same snapshot axis."""
    positions = list(range(cycles + 1))
    populations = [summary.snapshots.get(index) for index in positions]
    labels = [snapshot_label(index, cycles) for index in positions]
    figure = go.Figure()
    if composition:
        for field, name, color, pattern in (
            ("development", "Development", BLUE, ""),
            ("calibration", "Calibration", ORANGE, "/"),
            ("final_test", "Final test", PURPLE, "."),
        ):
            figure.add_bar(
                name=name,
                x=positions,
                y=[
                    getattr(base, field) if base is not None else None
                    for base in populations
                ],
                customdata=labels,
                marker={
                    "color": color,
                    "pattern": {
                        "shape": pattern,
                        "fillmode": "overlay",
                        "fgopacity": 0.42,
                        "size": 7,
                        "solidity": 0.28,
                    },
                },
                hovertemplate="%{customdata}<br>%{y:,.0f} observations<extra>%{fullData.name}</extra>",
            )
        figure.update_layout(barmode="stack", bargap=0.48 if cycles <= 3 else 0.24)
    else:
        figure.add_scatter(
            name="Base observations",
            x=positions,
            y=[base.total if base is not None else None for base in populations],
            text=labels,
            hovertemplate="%{text}<br>%{y:,.0f} observations<extra></extra>",
            mode="lines+markers",
            connectgaps=False,
            line={"color": BLUE, "width": 2.5},
            marker={"symbol": "circle", "size": 10 if cycles <= 3 else 6},
        )
    _layout(figure, "Observations", showlegend=composition)
    sparse = cycles <= 3
    step = max(1, (cycles + 4) // 5)
    ticks = positions if sparse else [0, *range(step, cycles - step // 2, step), cycles]
    tick_labels = [
        "Bootstrap"
        if index == 0
        else "Final append"
        if index == cycles
        else f"After cycle {index}"
        if sparse
        else str(index)
        for index in ticks
    ]
    padding = 0.52 if sparse else max(0.52, cycles * 0.10)
    figure.update_xaxes(
        range=[-padding, cycles + padding],
        tickmode="array",
        tickvals=ticks,
        ticktext=tick_labels,
        title=None if sparse else "Base snapshot by cycle",
    )
    figure.update_yaxes(rangemode="tozero", tickformat=",.0f", minexponent=0)
    return figure
