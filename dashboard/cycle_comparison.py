"""Compare two learning cycles under their verified producer evidence."""

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.drift_summary import METHOD_LABELS, MODEL_REPRESENTATION, DriftSummary
from dashboard.experiment_summary import BLUE, ORANGE, is_matching
from dashboard.model_results import WEIGHTING, scope_summary
from dashboard.provenance import Correspondence, Verification
from dashboard.scientific_summary import (
    BasePopulation,
    CalibrationMetadata,
    ClassificationMetrics,
    EvaluationMetrics,
)

UNAVAILABLE = "Unavailable"
POPULATIONS = (
    "Final test · before calibration",
    "Final test · after calibration",
    "Incoming batch · calibrated",
)
CLASSIFICATION = {"Accuracy": "accuracy", "Macro F0.5": "macro_f05", "MCC": "mcc"}
ERRORS = {"ECE": "ece", "MCE": "mce"}
COHORT_NOTE = (
    "Final-test populations change as the base grows and is split again. Incoming batches "
    "contain different observations. Cross-cycle differences do not establish learning improvement."
)


class ComparisonView(StrEnum):
    """The scientific quantities available for a paired display."""

    CLASSIFICATION = "Classification metrics"
    CALIBRATION = "Calibration errors"
    GROWTH = "Dataset growth"


@dataclass(frozen=True)
class ComparisonCycle:
    """Keep one selected cycle's proven measurements and schema."""

    cycle: int
    status: str
    base: BasePopulation | None = None
    incoming: int | None = None
    appended: int | None = None
    evaluation: EvaluationMetrics | None = None
    prediction: ClassificationMetrics | None = None
    calibration: CalibrationMetadata | None = None
    drift: DriftSummary | None = None
    features: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()

    @classmethod
    def from_evidence(
        cls, cycle: int, declared: int | None, checked: Verification | None
    ) -> "ComparisonCycle":
        """Keep historical or conflicting data out of the paired measurements."""
        if declared is not None and cycle > declared:
            return cls(cycle, "Historical · outside applied scope")
        row = (
            next((item for item in checked.cycles if item.cycle == cycle), None)
            if checked
            else None
        )
        if checked is None or row is None or cycle not in checked.declared_cycles:
            return cls(cycle, Correspondence.UNKNOWN)

        stages = dict(checked.stages)
        base_stage = stages.get(
            "bootstrap" if cycle == 1 else f"prepare_next_cycle_data_{cycle - 1}"
        )
        base = row.local_summary if base_stage and is_matching(base_stage) else None
        incoming, _ = scope_summary(checked, cycle, "generate_pred_data")
        append_stage = stages.get(f"prepare_next_cycle_data_{cycle}")
        appended = (
            row.local_summary.post_append
            if append_stage and is_matching(append_stage)
            else None
        )
        evaluation, _ = scope_summary(checked, cycle, "evaluate")
        prediction, _ = scope_summary(checked, cycle, "predict")
        drift, _ = scope_summary(checked, cycle, "analyze_drift")
        return cls(
            cycle,
            f"{row.evidence.availability} · {row.evidence.correspondence}",
            base.input_base if base else None,
            incoming.incoming_batch if incoming else None,
            appended.total if appended else None,
            evaluation.evaluation if evaluation else None,
            prediction.prediction if prediction else None,
            evaluation.calibration if evaluation else None,
            drift.drift if drift else None,
            base.feature_names if base else (),
            base.class_names if base else (),
        )

    def counts(self) -> tuple[int | None, ...]:
        """Return the realized cohorts in display order."""
        base = self.base
        return (
            base.total if base else None,
            base.development if base else None,
            base.calibration if base else None,
            base.final_test if base else None,
            self.incoming,
            self.appended,
        )

    def classification(self, population: str, field: str) -> float | None:
        """Read the named publication and calibration state."""
        if population == POPULATIONS[2]:
            scores = self.prediction
        elif self.evaluation is None:
            scores = None
        elif population == POPULATIONS[0]:
            scores = self.evaluation.before_calibration.classification
        else:
            scores = self.evaluation.after_calibration.classification
        return getattr(scores, field) if scores else None

    def calibration_errors(self, field: str) -> tuple[float | None, float | None]:
        """Read both probability families on this cycle's final-test population."""
        if self.evaluation is None:
            return None, None
        return (
            getattr(self.evaluation.before_calibration, field),
            getattr(self.evaluation.after_calibration, field),
        )


def chart_incompatibility(
    first: ComparisonCycle,
    second: ComparisonCycle,
    view: ComparisonView,
    series: tuple[tuple[float | int | None, ...], ...],
) -> str | None:
    """Require known matching schemas, identical definitions and complete selected values."""
    if first.cycle == second.cycle:
        return "Choose two different learning cycles."
    if (
        not first.classes
        or not first.features
        or not second.classes
        or not second.features
    ):
        return "A paired chart requires verified evidence and schemas for both cycles."
    if first.classes != second.classes or first.features != second.features:
        return "The ordered feature schema or class mapping differs between the cycles."
    if any(value is None for values in series for value in values):
        return "The selected measurements require verified values for both cycles."
    if view == ComparisonView.CALIBRATION:
        a, b = first.calibration, second.calibration
        if not a or not b or a.reporting_bins is None or b.reporting_bins is None:
            return "Calibration error charts require verified reporting bins for both cycles."
        if a.reporting_bins != b.reporting_bins:
            return "Calibration error reporting bin counts differ between the cycles."
    return None


def drift_context_changes(first: DriftSummary, second: DriftSummary) -> list[str]:
    """Identify differences in the recorded contexts of the two drift tests."""
    return [
        label
        for label, a, b in (
            (
                "Representation",
                (first.representation, first.feature_names),
                (second.representation, second.feature_names),
            ),
            (
                "Fitted preprocessing",
                (first.scaler, first.preprocessing_sha256),
                (second.scaler, second.preprocessing_sha256),
            ),
            ("Effective RBF gamma", first.effective_gamma, second.effective_gamma),
            (
                "Wasserstein method or settings",
                first.wasserstein_context,
                second.wasserstein_context,
            ),
            (
                "Permutation budget or stopping policy",
                (first.budget, first.early_stopping),
                (second.budget, second.early_stopping),
            ),
            (
                "Permutation seed provenance",
                (first.configured_seed, first.effective_seed),
                (second.configured_seed, second.effective_seed),
            ),
        )
        if a != b
    ]


def _shown(value: str | int | float | None, *, decimal: bool = False) -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}" if decimal else f"{value:.2%}"
    return value


def _paired_table(
    cycles: tuple[ComparisonCycle, ComparisonCycle],
    view: ComparisonView,
    population: str,
) -> pd.DataFrame:
    first, second = cycles
    rows = [("Evidence", first.status, second.status)]
    if view == ComparisonView.GROWTH:
        labels = (
            "Input base",
            "Development",
            "Calibration partition",
            "Final test",
            "Incoming batch",
            "Appended base",
        )
        rows.extend(
            (label, _shown(a), _shown(b))
            for label, a, b in zip(labels, first.counts(), second.counts(), strict=True)
        )
    elif view == ComparisonView.CLASSIFICATION:
        final = population != POPULATIONS[2]
        rows.append(
            (
                "Final-test observations" if final else "Incoming observations",
                *(_shown(item.counts()[3 if final else 4]) for item in cycles),
            )
        )
        for label, field in CLASSIFICATION.items():
            rows.append(
                (
                    label,
                    *(
                        _shown(
                            item.classification(population, field),
                            decimal=field == "mcc",
                        )
                        for item in cycles
                    ),
                )
            )
    else:
        rows.append(
            ("Final-test observations", *(_shown(item.counts()[3]) for item in cycles))
        )
        for label, field in ERRORS.items():
            for index, state in enumerate(("Before calibration", "After calibration")):
                rows.append(
                    (
                        f"{label} · {state}",
                        *(
                            _shown(item.calibration_errors(field)[index])
                            for item in cycles
                        ),
                    )
                )
        rows.append(
            (
                "Reporting bins",
                *(
                    _shown(
                        item.calibration.reporting_bins if item.calibration else None
                    )
                    for item in cycles
                ),
            )
        )
    return pd.DataFrame(
        rows,
        columns=["Measure", f"Cycle {first.cycle} · A", f"Cycle {second.cycle} · B"],
    )


def _drift_comparison(cycles: tuple[ComparisonCycle, ComparisonCycle]) -> None:
    first, second = cycles
    result = st.expander(
        "Drift findings and method context",
        key="compare-drift",
        on_change="rerun",
        type="compact",
    )
    if not result.open:
        return
    with result:
        st.caption(
            "Per-method findings at p < 0.05. Each column describes its own cycle's reference and incoming populations."
        )
        if first.drift and second.drift:
            changes = drift_context_changes(first.drift, second.drift)
            if changes:
                st.caption("Recorded differences · " + " · ".join(changes))
        st.caption(
            "Open Distribution drift for each cycle's statistics and settings. The findings below are descriptive."
        )
        columns = st.columns(2)
        for item, column in zip(cycles, columns, strict=True):
            with column:
                st.markdown(f"**Cycle {item.cycle} · {item.status}**")
                drift = item.drift
                if drift is None:
                    st.write(UNAVAILABLE)
                    continue
                st.caption(
                    f"Base snapshot {item.cycle - 1} (n={drift.reference_count:,}) · Incoming batch {item.cycle} (n={drift.incoming_count:,})"
                )
                st.markdown(
                    "  \n".join(
                        f"**{METHOD_LABELS[test.method]}** · {test.finding} (p = {test.p_value:.6g})"
                        for test in drift.tests
                    )
                )
                st.caption(
                    f"{MODEL_REPRESENTATION} · {drift.scaler or 'Identity'} · "
                    f"fit `{drift.preprocessing_sha256[:12]}` · gamma {drift.effective_gamma:.8g}  \n"
                    f"{drift.wasserstein_context} · {drift.budget:,} permutations budgeted per test · "
                    f"effective seed {_shown(drift.effective_seed)}  \n"
                    f"Early stopping · {drift.early_stopping.describe(drift.budget)}"
                )


def render_comparison(cycles: tuple[ComparisonCycle, ComparisonCycle]) -> None:
    """Show the paired summary for one topic and one compatible chart."""
    st.markdown(f"**Changing evaluation populations** · {COHORT_NOTE}")
    view = st.segmented_control(
        "Comparison view",
        list(ComparisonView),
        default=ComparisonView.CLASSIFICATION,
        required=True,
        key="compare-view",
        label_visibility="collapsed",
    )
    first, second = cycles
    population = POPULATIONS[1]
    with st.container(key="compare-content"):
        table, chart = st.columns([1.25, 1], gap="large")
        with table:
            if view == ComparisonView.CLASSIFICATION:
                population = st.selectbox(
                    "Classification population and calibration state",
                    POPULATIONS,
                    index=1,
                    key="compare-population",
                )
            else:
                st.markdown(
                    "#### Realized cohorts"
                    if view == ComparisonView.GROWTH
                    else "#### Final-test calibration errors"
                )
            st.dataframe(_paired_table(cycles, view, population), hide_index=True)
            if view == ComparisonView.CLASSIFICATION:
                st.caption(WEIGHTING)
            elif view == ComparisonView.CALIBRATION:
                st.caption(
                    "Lower is better. Before and after use the same final-test observations within each cycle."
                )
            else:
                st.caption(
                    "Development, calibration and final test partition each input base."
                )
        with chart:
            if view == ComparisonView.CLASSIFICATION:
                label = st.selectbox(
                    "Classification metric", CLASSIFICATION, key="compare-metric"
                )
                field = CLASSIFICATION[label]
                series = (
                    (
                        first.classification(population, field),
                        second.classification(population, field),
                    ),
                )
                names, decimal = (label,), field == "mcc"
            elif view == ComparisonView.CALIBRATION:
                label = st.selectbox("Calibration error", ERRORS, key="compare-error")
                series = tuple(
                    zip(
                        first.calibration_errors(ERRORS[label]),
                        second.calibration_errors(ERRORS[label]),
                        strict=True,
                    )
                )
                names, decimal = ("Before calibration", "After calibration"), False
            else:
                label = "Observations"
                st.markdown("#### Input and appended base")
                series = (
                    (first.counts()[0], second.counts()[0]),
                    (first.appended, second.appended),
                )
                names, decimal = ("Input base", "Appended base"), True
            issue = chart_incompatibility(first, second, view, series)
            if issue:
                st.info(issue)
            else:
                figure = go.Figure()
                labels = [f"Cycle {first.cycle} · A", f"Cycle {second.cycle} · B"]
                colors = (
                    (ORANGE, BLUE)
                    if view == ComparisonView.CALIBRATION
                    else (BLUE, ORANGE)
                )
                label_format = (
                    ",.0f"
                    if view == ComparisonView.GROWTH
                    else ".3f"
                    if decimal
                    else ".2%"
                )
                for name, values, color in zip(
                    names, series, colors[: len(series)], strict=True
                ):
                    figure.add_bar(
                        x=labels,
                        y=values,
                        name=name,
                        marker_color=color,
                        text=[format(value, label_format) for value in values],
                        textposition="outside",
                        cliponaxis=False,
                    )
                figure.update_layout(
                    height=290,
                    barmode="group",
                    showlegend=len(series) > 1,
                    margin={"l": 8, "r": 8, "t": 26, "b": 8},
                    legend={"orientation": "h", "y": 1.02, "yanchor": "bottom"},
                )
                figure.update_yaxes(
                    title=label,
                    rangemode="tozero",
                    automargin=True,
                    tickformat=",.0f"
                    if view == ComparisonView.GROWTH
                    else ".2f"
                    if decimal
                    else ".0%",
                    range=[-1, 1]
                    if decimal and view == ComparisonView.CLASSIFICATION
                    else None,
                )
                st.plotly_chart(figure, config={"displayModeBar": False})
        _drift_comparison(cycles)
