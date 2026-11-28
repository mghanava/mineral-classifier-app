"""Render the verified scientific overview without opening diagnostic files."""

from collections.abc import Callable

import streamlit as st

from dashboard.experiment_summary import (
    ExperimentSummary,
    Metric,
    growth_figure,
    is_matching,
    metric_series,
    snapshot_label,
    trajectory_figure,
)
from dashboard.state import CONFIGURE_PAGE, RUN_PAGE

POPULATION_CAVEAT = "Final-test sets and incoming batches change across cycles."


def cycle_label(cycle: int) -> str:
    """Name bootstrap independently from the learning cycles."""
    return f"Cycle {cycle}" if cycle else "Bootstrap · 0"


def _size(count: int | None) -> str:
    return f"n = {count:,}" if count is not None else "Cohort size unavailable"


def _score(value: float | int | None, form: str) -> str:
    return format(value, form) if value is not None else "Unavailable"


def ece_change(after: float, before: float) -> str:
    """Describe the same-cohort change without rounding away its direction."""
    points = (after - before) * 100
    if points == 0:
        return "Unchanged (0.00 pp)"
    direction = "Increased" if points > 0 else "Decreased"
    magnitude = "<0.01" if abs(points) < 0.01 else f"{abs(points):.2f}"
    return f"{direction} by {magnitude} pp"


def _headlines(summary: ExperimentSummary, cycle: int) -> None:
    test, batch = metric_series(summary, Metric.ACCURACY, range(cycle, cycle + 1))
    after, before = metric_series(summary, Metric.ECE, range(cycle, cycle + 1))
    after_value, before_value = after.values[0], before.values[0]
    with st.container(key="results-headlines", gap="xxsmall"):
        st.markdown(f"**Cycle {cycle}** · Latest complete matching learning cycle")
        test_col, calibration_col, batch_col = st.columns(3, gap="medium")
        with test_col, st.container(gap="xxsmall"):
            st.metric(
                "Final-test accuracy",
                _score(test.values[0], ".2%"),
                help="Calibrated class-weighted accuracy on the held-out final-test population.",
            )
            st.caption(f"Calibrated class-weighted · {_size(test.counts[0])}")
        with calibration_col, st.container(gap="xxsmall"):
            st.metric(
                "Final-test ECE after calibration",
                _score(after_value, ".2%"),
                help="Expected calibration error (ECE) on the same held-out observations before and after calibration. Changes in ECE are reported in percentage points (pp).",
            )
            if after_value is not None and before_value is not None:
                st.caption(
                    f"Same final-test cohort · {_size(after.counts[0])}  \n"
                    f"Before {before_value:.2%} · {ece_change(after_value, before_value)}"
                )
            else:
                st.caption(f"Final test · {_size(after.counts[0])}")
        with batch_col, st.container(gap="xxsmall"):
            st.metric(
                "Prediction-batch accuracy",
                _score(batch.values[0], ".2%"),
                help="Calibrated class-weighted accuracy against simulator labels in the incoming prediction batch.",
            )
            st.caption(f"Calibrated class-weighted · {_size(batch.counts[0])}")


def _population_summary(summary: ExperimentSummary, declared: int) -> str:
    latest_index = max(summary.snapshots)
    latest = summary.snapshots[latest_index]
    bootstrap = summary.snapshots.get(0)
    if latest_index == 0:
        return f"**{latest.total:,} observations** · Bootstrap"
    latest_label = snapshot_label(latest_index, declared).lower()
    if bootstrap is None:
        return f"**{latest.total:,} observations** · Latest verified ({latest_label})"
    return f"**{bootstrap.total:,} → {latest.total:,} observations** · Bootstrap to {latest_label}"


def _charts(summary: ExperimentSummary, declared: int) -> Metric:
    trajectory, growth = (
        st.columns([3, 2], gap="medium")
        if summary.learning and summary.snapshots
        else (st.container(), st.container())
    )
    metric = Metric.ACCURACY
    config = {"displayModeBar": False, "scrollZoom": False}
    if summary.learning:
        with trajectory, st.container(gap="xsmall"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.subheader("Learning trajectory")
                metric = st.selectbox(
                    "Metric",
                    list(Metric),
                    format_func=str,
                    width=278,
                    label_visibility="collapsed",
                )
            st.caption(POPULATION_CAVEAT)
            cycles = range(1, declared + 1)
            series = metric_series(summary, metric, cycles)
            st.plotly_chart(
                trajectory_figure(metric, cycles, series),
                key="learning-trajectory",
                config=config,
            )
    if summary.snapshots:
        with growth, st.container(gap="xsmall"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.subheader("Dataset evolution")
                view = st.segmented_control(
                    "Dataset view",
                    ["Totals", "Partitions"],
                    default="Totals",
                    required=True,
                    label_visibility="collapsed",
                )
            st.markdown(_population_summary(summary, declared))
            st.plotly_chart(
                growth_figure(summary, declared, composition=view == "Partitions"),
                key="dataset-growth",
                config=config,
            )
            missing = declared + 1 - len(summary.snapshots)
            if missing:
                st.caption(
                    f"{missing} {'snapshot lacks' if missing == 1 else 'snapshots lack'} verified counts. "
                    "Gaps mark unavailable evidence."
                )
            st.caption(
                "Appends add simulator-labeled observations. The final append is a data snapshot for future use."
            )
    return metric


def _cycle_table(
    summary: ExperimentSummary,
    declared: int,
    metric: Metric,
    observed: set[int],
    selected: int | None,
    inspect: Callable[[int], None],
) -> None:
    if metric.calibration_error:
        first_label, second_label = "Test · after", "Test · before"
    else:
        first_label, second_label = "Final test", "Prediction batch"
    options = range(declared + 1)
    has_counts = bool(summary.snapshots or summary.learning)
    first, second = metric_series(summary, metric, options)
    rows = []
    for cycle, left, right in zip(options, first.values, second.values, strict=True):
        source = summary.snapshots.get(max(0, cycle - 1))
        appended = summary.snapshots.get(cycle) if cycle else None
        item = summary.learning.get(cycle)
        if item and item.input_base is not None:
            source = item.input_base
        evidence = summary.evidence.get(cycle)
        status = (
            f"{evidence.availability} · {evidence.correspondence}"
            if evidence
            else "Unverified"
            if cycle in observed
            else "No local results"
        )
        rows.append(
            {
                "Cycle": cycle_label(cycle),
                "Evidence": status,
                "Input base": source.total if source else None,
                "Development": source.development if source else None,
                "Calibration": source.calibration if source else None,
                "Final-test n": source.final_test if source else None,
                "Incoming n": item.incoming_batch if item else None,
                "Post-append": appended.total if appended else None,
                first_label: left,
                second_label: right,
            }
        )

    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Cycle summary", width="content")
        cycle = st.selectbox(
            "Cycle to inspect",
            options,
            index=selected if selected in options else summary.latest or 0,
            format_func=cycle_label,
            width=166,
            label_visibility="collapsed",
        )
        st.button("Inspect cycle", on_click=inspect, args=(cycle,))
        show_partitions = has_counts and st.checkbox("Show partition counts")
    columns = ["Cycle", "Evidence"]
    if has_counts:
        columns.extend(("Input base", "Incoming n", "Post-append"))
    if show_partitions:
        columns[3:3] = ["Development", "Calibration", "Final-test n"]
    if summary.learning:
        columns.extend((first_label, second_label))
        context = "Final test" if metric.calibration_error else "Calibrated scores"
        st.caption(f"Table metric · {metric} · {context}")
    number = st.column_config.NumberColumn
    metric_format = (
        "percent" if metric == Metric.ACCURACY or metric.calibration_error else "%.4f"
    )
    st.dataframe(
        rows,
        column_order=columns,
        column_config={
            "Cycle": st.column_config.TextColumn("Cycle", width=120, pinned=True),
            "Input base": number("Input base", format="localized"),
            "Development": number(format="localized"),
            "Calibration": number(format="localized"),
            "Final-test n": number("Final-test n", format="localized"),
            "Incoming n": number("Incoming n", format="localized"),
            "Post-append": number("Post-append", format="localized"),
            first_label: number(format=metric_format),
            second_label: number(format=metric_format),
        },
        height=min(262, 35 + 32 * len(rows)),
        row_height=32,
        width="stretch" if has_counts else 650,
        hide_index=True,
        placeholder="",
    )
    if has_counts:
        st.caption(
            "Bootstrap counts describe the initial base. Blank cells indicate unavailable or inapplicable values."
        )


def render_overview(
    summary: ExperimentSummary,
    declared: int | None,
    observed: set[int],
    aggregate: bool,
    selected: int | None,
    inspect: Callable[[int], None],
) -> None:
    """Show only supported charts and keep scoped diagnostics one action away."""
    if not observed and not aggregate:
        st.info("No display artifacts found in this workspace.")
        if declared is None:
            st.page_link(CONFIGURE_PAGE, label="Go to Configure")
        else:
            st.page_link(RUN_PAGE, label="Go to Run")
        return
    if declared is None:
        st.caption(
            "Apply a valid configuration in Configure to establish the result scope."
        )
        st.page_link(CONFIGURE_PAGE, label="Go to Configure")
        return

    latest = summary.latest
    if latest is not None:
        _headlines(summary, latest)
    elif declared == 0 and 0 in summary.snapshots:
        st.caption(
            "Bootstrap is verified. This configuration declares zero learning cycles."
        )
    elif not summary.snapshots:
        message = (
            "No current results match the applied configuration."
            if summary.evidence
            else "Refresh result evidence to check local outputs."
        )
        st.caption(
            f"{message} Cycle explorer can open available diagnostics with their evidence labels."
        )
    else:
        st.caption(
            "No complete matching learning cycle yet. The verified base snapshots are available below."
        )

    metric = _charts(summary, declared)
    historical = observed.difference(range(declared + 1))
    if historical:
        st.caption(
            f"Cycle explorer includes {len(historical)} historical "
            f"{'cycle' if len(historical) == 1 else 'cycles'} outside the applied scope."
        )
    if any(not is_matching(item) for item in summary.evidence.values()):
        st.page_link(RUN_PAGE, label="Review execution in Run")
    if (
        summary.learning
        or summary.snapshots
        or observed.intersection(range(declared + 1))
    ):
        _cycle_table(summary, declared, metric, observed, selected, inspect)
