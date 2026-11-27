"""Present classification, calibration and CV/refit results from checked artifacts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from numpy.typing import NDArray

from dashboard.artifacts import Workspace
from dashboard.components import show_artifact
from dashboard.experiment_summary import is_matching
from dashboard.provenance import Availability, Evidence, Verification
from dashboard.scientific_summary import ClassificationMetrics, CycleSummary
from dashboard.spatial_predictions import (
    JoinedPredictions,
    PredictionJoin,
    ProbabilityFamily,
    verify_prediction_join,
)
from src.configuration_files import configuration_operation

WEIGHTING = (
    "Published scores use class-balanced sample weights. Each observed class "
    "contributes the same total weight within its cohort."
)


def scope_summary(
    checked: Verification | None,
    cycle: int,
    stage: str,
) -> tuple[CycleSummary | None, Evidence | None]:
    """Select a matching producer independently of unrelated cycle outputs."""
    if checked is None or cycle not in checked.declared_cycles:
        return None, None
    evidence = dict(checked.stages).get(f"{stage}_cycle_{cycle}")
    result = next((row for row in checked.cycles if row.cycle == cycle), None)
    return (
        result.local_summary if result and evidence and is_matching(evidence) else None,
        evidence,
    )


def _current_evidence(
    workspace: Workspace, checked: Verification | None
) -> Verification | None:
    if checked is None:
        return None
    try:
        with configuration_operation(workspace.params):
            return checked if checked.still_observed(workspace) else None
    except (OSError, ValueError):
        return None


def _context(
    cycle: int, scope: str, count: int | None, evidence: Evidence | None
) -> None:
    label = f"{count:,} observations" if count is not None else "Cohort size unverified"
    status = str(evidence.correspondence) if evidence else "Unverified correspondence"
    st.caption(f"Cycle {cycle} · {scope} · {label} · {status}")


def _formatted(value: float | None, *, percent: bool = True) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.2%}" if percent else f"{value:.3f}"


def classification_table(
    before: ClassificationMetrics, after: ClassificationMetrics
) -> pd.DataFrame:
    """Format two published evaluations of the same test observations."""
    return pd.DataFrame(
        {
            "Metric": ["Accuracy", "Macro F0.5", "MCC"],
            "Before calibration": [
                _formatted(before.accuracy),
                _formatted(before.macro_f05),
                _formatted(before.mcc, percent=False),
            ],
            "After calibration": [
                _formatted(after.accuracy),
                _formatted(after.macro_f05),
                _formatted(after.mcc, percent=False),
            ],
        }
    )


def calibration_change(before: float | None, after: float | None) -> str:
    """Report direction and percentage-point difference."""
    if before is None or after is None:
        return "Unavailable"
    difference = (after - before) * 100
    if difference == 0:
        return "Unchanged · 0.00 pp"
    magnitude = "<0.01" if abs(difference) < 0.005 else f"{abs(difference):.2f}"
    return f"{'Increased' if difference > 0 else 'Decreased'} · {magnitude} pp"


@dataclass(frozen=True)
class BatchDiagnostics:
    """Unweighted incoming-batch counts ordered by the persisted class mapping."""

    classes: tuple[str, ...]
    counts: NDArray[np.int64]

    @classmethod
    def from_join(
        cls, joined: JoinedPredictions, family: ProbabilityFamily
    ) -> "BatchDiagnostics":
        """Count observed rows and predicted columns after the verified join."""
        names = joined.population.class_names
        observed = joined.population.observed_class_ids
        if observed is None:
            raise ValueError("Observed labels are unavailable for the incoming batch.")
        predicted = joined.scores(family).predicted_class_ids()
        size = len(names)
        counts = np.bincount(
            observed * size + predicted, minlength=size * size
        ).reshape(size, size)
        return cls(names, counts)

    def per_class(self) -> pd.DataFrame:
        """Show undefined precision or recall when the relevant denominator is zero."""
        support = self.counts.sum(axis=1)
        predicted = self.counts.sum(axis=0)
        true_positive = self.counts.diagonal()

        def rates(denominator):
            return [
                f"{tp / total:.2%}" if total else "Undefined"
                for tp, total in zip(true_positive, denominator, strict=True)
            ]

        return pd.DataFrame(
            {
                "Class": self.classes,
                "Observed support": support,
                "Predicted": predicted,
                "Precision": rates(predicted),
                "Recall": rates(support),
            }
        )

    def figure(self, *, dark: bool) -> go.Figure:
        """Plot counts with observed rows and predicted columns, including zero cells."""
        figure = go.Figure(
            go.Heatmap(
                x=self.classes,
                y=self.classes,
                z=self.counts,
                text=self.counts,
                texttemplate="%{text}",
                colorscale="Cividis",
                zmin=0,
                colorbar={"title": "Count", "thickness": 12},
                hovertemplate="Observed %{y}<br>Predicted %{x}<br>%{z} observations<extra></extra>",
            )
        )
        figure.update_layout(
            height=435,
            template="plotly_dark" if dark else "plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            margin={"l": 4, "r": 12, "t": 14, "b": 4},
        )
        figure.update_xaxes(
            title="Predicted class",
            type="category",
            categoryorder="array",
            categoryarray=self.classes,
            automargin=True,
        )
        figure.update_yaxes(
            title="Observed class",
            type="category",
            categoryorder="array",
            categoryarray=self.classes,
            autorange="reversed",
            automargin=True,
        )
        return figure


def _batch_join(workspace: Workspace, cycle: int, refresh: bool) -> PredictionJoin:
    key = (workspace.root.resolve(), cycle)
    old_key, result = st.session_state.get("_classification_join", (None, None))
    try:
        with configuration_operation(workspace.params):
            changed = bool(
                result and result.joined and not result.joined.still_observed(workspace)
            )
    except (ValueError, OSError) as error:
        st.info(str(error))
        return PredictionJoin(
            Evidence(availability=Availability.PARTIAL, reasons=(str(error),))
        )
    if old_key != key or refresh or changed:
        with st.spinner("Verifying incoming predictions…"):
            result = verify_prediction_join(workspace, cycle)
            st.session_state["_classification_join"] = (key, result)
    return result


def _figure(workspace: Workspace, path: Path, width: int) -> None:
    if workspace.available(path):
        show_artifact(workspace, path, image_width=width)
    else:
        st.info(f"Original diagnostic unavailable · {path.name}.")


def render_classification(
    workspace: Workspace,
    cycle: int,
    checked: Verification | None,
    *,
    historical: bool,
    refresh: bool,
) -> None:
    """Display published classification scores and verified incoming count diagnostics."""
    population = st.segmented_control(
        "Classification population",
        ["Final test", "Incoming prediction batch"],
        default="Final test",
        required=True,
        key="classification-population",
        label_visibility="collapsed",
    )
    checked = _current_evidence(workspace, checked)
    final = population == "Final test"
    summary, evidence = scope_summary(
        checked, cycle, "evaluate" if final else "predict"
    )
    count = (
        summary.input_base.final_test
        if final and summary and summary.input_base
        else summary.incoming_batch
        if summary and not final
        else None
    )
    st.markdown("### Classification")
    _context(cycle, population, count, evidence)
    if final:
        figure_column, notes = st.columns([1.35, 1], gap="large")
        with notes:
            st.markdown("#### Published classification scores")
            if summary and summary.evaluation:
                test = summary.evaluation
                st.dataframe(
                    classification_table(
                        test.before_calibration.classification,
                        test.after_calibration.classification,
                    ),
                    hide_index=True,
                )
                st.caption(
                    WEIGHTING + " Both columns use the same final-test observations."
                )
            else:
                st.info(
                    "Refresh result evidence to check published scores. Original diagnostics remain available."
                )
            st.caption(
                "The saved confusion matrix uses calibrated final-test predictions."
            )
        with figure_column:
            st.markdown("#### Calibrated confusion matrix")
            _figure(
                workspace, Path(f"evaluation/cycle_{cycle}/confusion_matrix.png"), 600
            )
        return
    if summary and summary.prediction:
        st.markdown("#### Published calibrated scores")
        metrics = summary.prediction
        for column, (name, value) in zip(
            st.columns(3),
            (
                ("Accuracy", _formatted(metrics.accuracy)),
                ("Macro F0.5", _formatted(metrics.macro_f05)),
                ("MCC", _formatted(metrics.mcc, percent=False)),
            ),
            strict=True,
        ):
            column.metric(name, value)
        st.caption(WEIGHTING)
    else:
        st.caption("Refresh result evidence to check the published calibrated scores.")
    if historical:
        st.info(
            "Historical batch · open Original diagnostics to inspect saved figures."
        )
        return
    result = _batch_join(workspace, cycle, refresh)
    if not result.joined:
        st.info(
            "Verified batch rows unavailable. Original diagnostics remain accessible."
        )
        for reason in result.evidence.reasons:
            st.caption(reason)
        if st.button("Retry batch verification"):
            st.session_state.pop("_classification_join", None)
            st.rerun()
        return
    if not result.joined.still_observed(workspace):
        st.info(
            "Workspace evidence changed. Reopen the view to verify batch rows again."
        )
        return
    st.caption(
        f"Verified batch alignment · {result.joined.population.count:,} observations · Unweighted derived counts and rates"
    )
    plot, details = st.columns([1.12, 1], gap="large")
    with details:
        st.markdown("#### Per-class summary")
        family = st.segmented_control(
            "Batch probability family",
            list(ProbabilityFamily),
            default=ProbabilityFamily.CALIBRATED,
            required=True,
            format_func=str.capitalize,
            key="classification-family",
        )
        batch = BatchDiagnostics.from_join(result.joined, family)
        st.dataframe(batch.per_class(), hide_index=True, height=230)
        st.caption(
            "Precision uses predicted counts. Recall uses observed support. A zero denominator yields an undefined rate."
        )
    with plot:
        st.markdown("#### Confusion matrix")
        st.plotly_chart(
            batch.figure(dark=st.context.theme.type == "dark"),
            theme=None,
            key=f"batch-counts:{cycle}",
        )


def optimizer_applicability(method: str, verbose: bool | None) -> str | None:
    """Describe known cases where optimizer diagnostics are inapplicable."""
    if method == "isotonic":
        return "Optimizer diagnostics are inapplicable to isotonic calibration."
    if verbose is False:
        return (
            "Optimizer diagnostics are inapplicable with evaluation verbosity disabled."
        )
    return None


def render_calibration(
    workspace: Workspace, cycle: int, checked: Verification | None
) -> None:
    """Compare calibration errors on the same checked final-test cohort."""
    summary, evidence = scope_summary(
        _current_evidence(workspace, checked), cycle, "evaluate"
    )
    base = summary.input_base if summary else None
    calibration = summary.calibration if summary else None
    st.markdown("### Probability calibration")
    _context(cycle, "Final test", base.final_test if base else None, evidence)
    if summary and summary.evaluation:
        before, after = (
            summary.evaluation.before_calibration,
            summary.evaluation.after_calibration,
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Calibration error ↓": [
                        "Expected calibration error (ECE)",
                        "Maximum calibration error (MCE)",
                    ],
                    "Before": [_formatted(before.ece), _formatted(before.mce)],
                    "After": [_formatted(after.ece), _formatted(after.mce)],
                    "Change": [
                        calibration_change(before.ece, after.ece),
                        calibration_change(before.mce, after.mce),
                    ],
                }
            ),
            hide_index=True,
        )
        st.caption(
            "Lower is better · Same final-test observations · Class-balanced sample weights"
        )
    else:
        st.info("Refresh result evidence to check the paired calibration scores.")
    image, notes = st.columns([1.65, 1], gap="large")
    directory = Path(f"evaluation/cycle_{cycle}")
    diagnostic, title = directory / "reliability_diagram.png", "Reliability diagram"
    with notes:
        st.markdown("#### Reading the diagram")
        if calibration and base:
            st.markdown(
                f"**{calibration.method.replace('_', ' ').capitalize()}** · "
                f"{base.calibration:,} in the calibration partition · {base.final_test:,} final test · "
                f"{calibration.reporting_bins or 'Unrecorded'} reporting bins"
            )
            st.caption(
                "Calibration uses the reserved calibration partition. Parametric methods use an internal fit and monitor split."
            )
        st.markdown(
            "Below the diagonal, confidence exceeds accuracy (overconfidence). "
            "Above it, accuracy exceeds confidence (underconfidence). "
            "The secondary bars sum sample weights within each bin. ECE and MCE depend on the binning."
        )
        st.caption(
            "Legacy figures may invert the annotations or label the weighted bars “Sample Count”."
        )
        applicability = (
            optimizer_applicability(calibration.method, calibration.verbose)
            if calibration
            else None
        )
        if applicability:
            st.caption(applicability)
        else:
            options = [
                path
                for path in workspace.files(directory)
                if path.name == "optimum_params.txt"
                or path.name.endswith("_loss_and_gradient.png")
            ]
            if options and st.toggle(
                "Calibration optimizer details", key=f"optimizer:{cycle}"
            ):
                diagnostic = st.selectbox(
                    "Optimizer diagnostic",
                    options,
                    format_func=lambda path: (
                        "Fitted parameters"
                        if path.suffix == ".txt"
                        else "Loss and gradient"
                    ),
                    key=f"optimizer-kind:{cycle}",
                )
                title = (
                    "Fitted parameters"
                    if diagnostic.suffix == ".txt"
                    else "Optimizer loss and gradient"
                )
            elif not options and calibration:
                st.caption("No saved optimizer diagnostic is available.")
    with image:
        st.markdown(f"#### {title}")
        _figure(
            workspace,
            diagnostic,
            550 if diagnostic.name == "reliability_diagram.png" else 730,
        )


def render_training(
    workspace: Workspace, cycle: int, checked: Verification | None
) -> None:
    """Show the verified CV record and fresh full-development deployment schedule."""
    summary, evidence = scope_summary(
        _current_evidence(workspace, checked), cycle, "train"
    )
    model = summary.model if summary else None
    training = model.training if model else None
    st.markdown("### Training and deployed refit")
    _context(cycle, "Development", model.development_count if model else None, evidence)
    if model:
        scaler = model.scaler or "Identity transform"
        st.markdown(
            f"**{model.model.upper()}** · Device {model.effective_device or 'unrecorded'} · Preprocessing {scaler}"
        )
    if training:
        st.markdown(
            f"**{len(training.folds)} CV folds** · {training.epoch_budget} epochs budget per fold · "
            f"**{training.refit_epochs} deployed refit epochs**"
        )
    else:
        st.info(
            "Verified fold and refit metadata unavailable. Refresh result evidence or browse the saved diagnostics."
        )
    directory = Path(f"trained/cycle_{cycle}")
    options = [
        path
        for path in workspace.files(directory)
        if path.name.startswith("training_plots_dataset_") and path.suffix == ".png"
    ]
    image, notes = st.columns([1.7, 1], gap="large")
    with notes:
        st.markdown("#### Cross-validation and refit")
        st.caption(
            "The deployed model starts fresh on the full development population and follows the CV-derived schedule."
        )
        if training:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Fold": index,
                            "Train": fold.train_count,
                            "Validation": fold.validation_count,
                            "Best epoch": fold.best_epoch,
                            "Completed": fold.completed_epochs,
                        }
                        for index, fold in enumerate(training.folds, 1)
                    ]
                ),
                hide_index=True,
                height=190,
            )
            st.caption(
                f"Refit schedule replays fold {training.source_fold + 1} rates, truncating or holding the last rate to match the refit duration."
            )
            if st.toggle("Refit schedule and seeds", key=f"refit-schedule:{cycle}"):
                st.caption(
                    f"Configured seed {training.configured_seed if training.configured_seed is not None else 'unrecorded'} · "
                    f"Refit seed {training.refit_seed if training.refit_seed is not None else 'unrecorded'}"
                )
                st.caption(
                    "CV seeds · "
                    + " · ".join(
                        f"Fold {i}: {f.seed if f.seed is not None else 'unrecorded'}"
                        for i, f in enumerate(training.folds, 1)
                    )
                )
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Epoch": range(1, training.refit_epochs + 1),
                            "Learning rate": training.learning_rates,
                        }
                    ),
                    hide_index=True,
                    height=175,
                )
    with image:
        st.markdown("#### CV fold diagnostic")
        if options:
            path = st.selectbox(
                "Training fold",
                options,
                format_func=lambda item: f"Fold {item.stem.rsplit('_', 1)[-1]}",
                key=f"training-fold:{cycle}",
            )
            _figure(workspace, path, 800)
            st.caption("Legacy F1 legends refer to Macro F0.5.")
        else:
            st.info("No stored training fold figures are available.")
