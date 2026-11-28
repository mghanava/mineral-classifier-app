"""Present one existing result diagnostic at a time."""

from collections.abc import Iterable
from pathlib import Path

import streamlit as st

from dashboard.artifacts import AGGREGATE, Workspace
from dashboard.components import UNVERIFIED, read_active, show_artifact
from dashboard.experiment_summary import ExperimentSummary, is_matching
from dashboard.overview import cycle_label, render_overview
from dashboard.provenance import (
    AppliedConfiguration,
    Availability,
    Correspondence,
    Verification,
    verify_workspace,
)
from dashboard.state import (
    COMPARE_CYCLES,
    CYCLE_EXPLORER,
    EXPERIMENT_OVERVIEW,
    dashboard_state,
)

DATA = "Data and spatial graphs"
MODEL = "Model performance and training"
CALIBRATION = "Probability calibration"
PREDICTION = "Spatial predictions"
DRIFT = "Distribution drift"
PERFORMANCE = "Cross-cycle performance"

LABELS = {
    "base_data_graph": "Full base graph",
    "test_graph": "Final-test graph",
    "calib_graph": "Calibration graph",
    "prediction_data_graph": "Incoming batch graph · simulator labels",
    "tsne_comparison": "Feature separation · t-SNE",
    "reliability_diagram": "Final-test reliability diagram",
    "confusion_matrix": "Final-test confusion matrix",
    "confussion_matrix_": "Incoming-batch confusion matrix",
    "predictions": "Incoming-batch predictions table",
    "histograms": "Incoming-batch score distributions",
    "optimum_params": "Fitted calibration parameters",
    "calibration_binding": "Calibration binding JSON",
    "graph_binding": "Training graph binding JSON",
    "preprocessing": "Fitted preprocessing JSON",
    "drift_results": "Permutation test report",
    "drift_metadata": "Drift settings and bindings JSON",
    "distance_distributions": "Distance distributions",
    "marginal_distributions": "Marginal feature distributions",
    "mutual_information": "Mutual information",
    "pca_kpca_grid": "PCA and Kernel PCA projections",
    "cycles_performance": "Original cross-cycle performance figure",
}


def _label(path: Path) -> str:
    if path.stem == "metrics":
        population = "Final-test" if path.parts[0] == "evaluation" else "Incoming-batch"
        return f"{population} metrics JSON"
    if path.parts[0] == "prediction" and path.stem == "confusion_matrix":
        return "Incoming-batch confusion matrix"
    for prefix, description in (
        ("train_graph_", "Training graph · fold"),
        ("val_graph_", "Validation graph · fold"),
        ("training_plots_dataset_", "Training diagnostics · fold"),
    ):
        number = path.stem.removeprefix(prefix)
        if number != path.stem and number.isdecimal():
            return f"{description} {number}"
    if path.stem.startswith("calibration_") and path.stem.endswith(
        "_loss_and_gradient"
    ):
        method = path.stem.removeprefix("calibration_").removesuffix(
            "_loss_and_gradient"
        )
        return f"Calibration optimizer · {method}"
    return LABELS.get(path.stem, path.name)


def _figure_exports(files: Iterable[Path]) -> list[Path]:
    by_stem = {}
    for path in sorted(files):
        if path.suffix in (".json", ".html"):
            # HTML sorts first, so a JSON export supersedes the matching HTML.
            by_stem[path.stem] = path
    return list(by_stem.values())


def _diagnostics(workspace: Workspace, section: str, cycle: int) -> list[Path]:
    if section == PERFORMANCE:
        st.caption(
            "Learning cycles split a growing base and score new batches. "
            "The test and batch trajectories describe changing evaluation populations."
        )
        return [AGGREGATE] if workspace.available(AGGREGATE) else []
    if section == DRIFT:
        st.caption(
            f"Compares all rows of input base {cycle - 1} with incoming batch {cycle}. "
            "Drift tests describe distribution differences, and performance requires separate evaluation."
        )
        return workspace.files(Path(f"drift_analysis/cycle_{cycle}"))
    evaluation = workspace.files(Path(f"evaluation/cycle_{cycle}"))
    if section == CALIBRATION:
        st.caption(
            "The reliability diagram reports final-test probabilities before and after calibration."
        )
        return sorted(
            (path for path in evaluation if path.name != "confusion_matrix.png"),
            key=lambda path: (path.name != "reliability_diagram.png", path.name),
        )
    return (
        [
            path
            for path in evaluation
            if path.name in {"confusion_matrix.png", "metrics.json"}
        ]
        + [
            path
            for path in workspace.files(Path(f"prediction/cycle_{cycle}"))
            if path.name
            in {"metrics.json", "confussion_matrix_.png", "confusion_matrix.png"}
        ]
        + workspace.files(Path(f"trained/cycle_{cycle}"))
    )


def _pick_diagnostic(choices: list[Path], section: str, cycle: int) -> Path | None:
    if not choices:
        return None
    # Diagnostic labels repeat across cycles and populations. Bind the widget to the file scope.
    path = st.selectbox(
        "Diagnostic",
        choices,
        format_func=_label,
        key=f"diagnostic:{section}:{cycle}:{choices[0].parent}",
    )
    return path if path in choices else choices[0]


def _interpret(path: Path) -> None:
    if path.stem == "tsne_comparison":
        st.caption(
            "Saved t-SNE views summarize class separation for several perplexities. "
            "The silhouette score uses raw features."
        )
    if path.stem == "reliability_diagram":
        st.caption(
            "Figure label revision unverified. Below the diagonal is overconfidence "
            "and above is underconfidence. Each bar sums sample weights in its bin "
            "(legacy figures label this axis “Sample Count”)."
        )
    if path.stem in {"metrics", "cycles_performance"} or path.stem.startswith(
        "training_plots_dataset_"
    ):
        st.caption(
            "The f1 field and legacy F1 legends refer to Macro F0.5 (macro Fβ with β = 0.5), "
            "computed with class-balanced sample weights. Figure label revision is unverified."
        )


def _inspect(cycle: int) -> None:
    state = dashboard_state()
    state.result_cycle = cycle
    state.result_mode = CYCLE_EXPLORER


def _remember_selection(field: str, key: str) -> None:
    setattr(dashboard_state(), field, st.session_state[key])


def _status(
    checked: Verification | None,
    applied: AppliedConfiguration | None,
    observed: set[int],
    aggregate: bool,
    summary: ExperimentSummary,
) -> str:
    if not observed and not aggregate:
        return "No local results"
    if (
        applied is not None
        and not aggregate
        and not observed.intersection(range(applied.cycles + 1))
    ):
        return "Historical files only · Outside the applied scope"
    if checked is None or not summary.evidence:
        return "Unverified correspondence"
    scoped = list(summary.evidence.values())
    if checked.aggregate.availability != Availability.NOT_APPLICABLE:
        scoped.append(checked.aggregate)
    if all(is_matching(item) for item in scoped):
        return "Verified matching"
    coverage = (
        "Partial coverage · "
        if any(
            item.availability in (Availability.MISSING, Availability.PARTIAL)
            for item in scoped
        )
        else ""
    )
    correspondence = (
        Correspondence.CONFLICT
        if any(item.correspondence == Correspondence.CONFLICT for item in scoped)
        else Correspondence.UNKNOWN
    )
    return f"{coverage}{correspondence}"


def _identity(
    applied: AppliedConfiguration | None,
    checked: Verification | None,
    summary: ExperimentSummary,
    observed: set[int],
    aggregate: bool,
) -> None:
    status = _status(checked, applied, observed, aggregate, summary)
    color = (
        "green"
        if status == "Verified matching"
        else "orange"
        if "Partial coverage" in status or Correspondence.CONFLICT in status
        else "gray"
    )
    badge = f":{color}-badge[{status}]"
    if applied:
        declared = applied.cycles
        coverage = f"{len(summary.learning)} of {declared} learning cycles verified"
        if declared == 0:
            coverage = (
                "Bootstrap verified"
                if 0 in summary.snapshots
                else "Bootstrap awaiting verification"
            )
        st.markdown(f"**Experiment evidence** · {badge}  ·  {coverage}")
        device = ""
        if summary.latest is not None:
            model = summary.learning[summary.latest].model
            if model and model.effective_device:
                device = f" · Recorded training device {model.effective_device} (cycle {summary.latest})"
        st.caption(
            f"Applied model {applied.model.upper()} · Requested device {applied.requested_device}{device}"
        )
    else:
        st.markdown(f"**Experiment evidence** · {badge}")
    if checked is None and (observed or aggregate):
        st.caption(UNVERIFIED)


def _details(
    applied: AppliedConfiguration | None, checked: Verification | None
) -> None:
    attempt = dashboard_state().last_run
    if applied is None and checked is None and attempt is None:
        return
    details = st.expander(
        "Evidence and execution details",
        key="_results_evidence_details",
        on_change="rerun",
    )
    if not details.open:
        return
    with details:
        st.caption(
            "Experiment correspondence covers the applied pipeline. Cycle completeness covers its "
            "required stages. Each diagnostic reports correspondence for its own producer. "
            "Prediction alignment checks the graph and score rows used together."
        )
        if applied:
            st.caption(
                "Applied params.yaml fingerprint · SHA-256 of the exact file bytes"
            )
            st.code(applied.fingerprint, language=None)
        if attempt:
            identity = (
                f" · Applied configuration at execution `{attempt.config_id}`"
                if attempt.config_id
                else ""
            )
            st.write(
                f"**Session execution outcome** · {attempt.status} · {attempt.scope}{identity}"
            )
        if checked:
            st.write(
                f"**Local evidence checked** · {checked.checked_at:%Y-%m-%d %H:%M:%S UTC} · {checked.elapsed:.2f} s"
            )
            for reason in checked.reasons:
                st.text(reason)
            if checked.stages:
                by_stage = dict(checked.stages)
                scope = st.selectbox("Evidence stage", by_stage)
                stage = by_stage[scope]
                st.write(f"{stage.availability} · {stage.correspondence}")
                if stage.locked_outputs_match:
                    st.caption("Declared output bytes match the local DVC lock.")
                for reason in stage.reasons:
                    st.text(reason)


def _cycle_evidence(
    checked: Verification | None,
    summary: ExperimentSummary,
    cycle: int,
    section: str,
    historical: bool,
) -> None:
    evidence = (
        checked.aggregate
        if checked is not None and section == PERFORMANCE
        else summary.evidence.get(cycle)
    )
    scope = (
        "Cross-cycle performance"
        if section == PERFORMANCE
        else "Bootstrap"
        if cycle == 0
        else f"Learning cycle {cycle}"
    )
    label = f"{scope} · {'Aggregate diagnostic' if section == PERFORMANCE else 'Cycle completeness'} · "
    if historical and section != PERFORMANCE:
        label += "Historical cycle outside the applied configuration · "
    if evidence is None:
        label += Correspondence.UNKNOWN
    elif evidence.availability == Availability.NOT_APPLICABLE:
        label += (
            f"{evidence.availability} · Existing files are historical for this scope"
        )
    else:
        label += f"{evidence.availability} · {evidence.correspondence}"
    st.markdown(label)
    if evidence and evidence.reasons:
        details = st.expander(
            "Selected scope evidence",
            key="_results_scope_evidence",
            on_change="rerun",
            type="compact",
        )
        if details.open:
            with details:
                for reason in evidence.reasons:
                    st.text(reason)


def _explore(
    workspace: Workspace,
    declared: int | None,
    observed: set[int],
    aggregate: bool,
    checked: Verification | None,
    summary: ExperimentSummary,
    *,
    folds: int,
    refresh: bool,
) -> None:
    state = dashboard_state()
    scope = sorted(set(range((declared or 0) + 1)) | observed)
    if state.result_cycle not in scope:
        state.result_cycle = summary.latest or next(
            (cycle for cycle in scope if cycle > 0), 0
        )
    st.session_state["_results_cycle"] = state.result_cycle
    cycle_column, section_column = st.columns(2, gap="medium")
    with cycle_column:
        cycle = st.selectbox(
            "Learning cycle",
            scope,
            format_func=lambda item: (
                "Bootstrap · cycle 0"
                if item == 0
                else cycle_label(item)
                + (" · existing files only" if item > (declared or 0) else "")
            ),
            key="_results_cycle",
            on_change=_remember_selection,
            args=("result_cycle", "_results_cycle"),
        )
    with section_column:
        sections = [DATA, MODEL, CALIBRATION, PREDICTION, DRIFT] if cycle else [DATA]
        if aggregate or (declared is not None and declared >= 2):
            sections.append(PERFORMANCE)
        st.session_state["_results_section"] = (
            state.result_section if state.result_section in sections else DATA
        )
        section = st.selectbox(
            "Result section",
            sections,
            key="_results_section",
            on_change=_remember_selection,
            args=("result_section", "_results_section"),
        )
    if section == DRIFT:
        from dashboard.drift_results import render_drift

        with st.container(key="results-drift", gap="small"):
            if render_drift(workspace, cycle, checked):
                return
    if section in (MODEL, CALIBRATION):
        from dashboard.model_results import (
            render_calibration,
            render_classification,
            render_training,
        )

        options = (
            ["Classification", "Training", "Original diagnostics"]
            if section == MODEL
            else ["Calibration overview", "Original diagnostics"]
        )
        with st.container(key="results-science-header", gap="xxsmall"):
            mode = st.segmented_control(
                "Scientific result view",
                options,
                default=options[0],
                required=True,
                key=f"scientific-view:{section}",
                label_visibility="collapsed",
            )
        if mode == "Classification":
            render_classification(
                workspace,
                cycle,
                checked,
                historical=cycle > (declared or 0),
                refresh=refresh,
            )
            return
        if mode == "Training":
            render_training(workspace, cycle, checked)
            return
        if mode == "Calibration overview":
            render_calibration(workspace, cycle, checked)
            return
    spatial = section in (DATA, PREDICTION)
    if spatial:
        from dashboard.spatial_explorer import choose_population, render_spatial

        with st.container(
            key="results-spatial-header",
            horizontal=True,
            gap="small",
            vertical_alignment="bottom",
        ):
            view_mode = st.segmented_control(
                "Spatial view",
                ["Interactive map", "Original diagnostics"],
                default="Interactive map",
                required=True,
                key=f"spatial-view:{section}",
                label_visibility="collapsed",
            )
            scope = choose_population(
                cycle, declared=declared, folds=folds, predictions=section == PREDICTION
            )
        status = summary.evidence.get(cycle)
        correspondence = (
            f"{status.availability} · {status.correspondence}"
            if status
            else Correspondence.UNKNOWN
        )
        if cycle:
            st.caption(
                f"Cycle completeness · {correspondence}  |  "
                f"Input base {cycle - 1} + incoming batch {cycle} → appended base {cycle}"
            )
        else:
            st.caption(f"Bootstrap completeness · {correspondence}")
        if view_mode == "Interactive map":
            render_spatial(
                workspace,
                scope,
                predictions=section == PREDICTION,
                historical=cycle > (declared or 0),
                refresh=refresh,
            )
            return
        original = workspace.files(scope.directory)
        choices = _figure_exports(original)
        if section == DATA:
            choices += [path for path in original if path.suffix == ".png"]
        else:
            choices += [
                path
                for path in workspace.files(Path(f"prediction/cycle_{cycle}"))
                if path.name in ("predictions.csv", "histograms.png")
            ]
    else:
        _cycle_evidence(checked, summary, cycle, section, cycle > (declared or 0))
        if cycle and section != PERFORMANCE:
            st.caption(
                f"**Lineage** · Input base {cycle - 1} + incoming batch {cycle} → appended base {cycle}"
            )
        choices = _diagnostics(workspace, section, cycle)
    chosen = _pick_diagnostic(choices, section, cycle)
    if chosen is None:
        st.warning(
            "No display files found for this selection. The output may be absent or partial."
        )
        return
    _diagnostic_evidence(chosen, checked, cycle, declared)
    _interpret(chosen)
    if chosen.suffix == ".csv" and section == PREDICTION:
        st.caption("Standalone prediction CSV · Producer row order")
        st.download_button(
            "Original producer CSV",
            lambda: workspace.read_bytes(chosen),
            chosen.name,
            "text/csv",
            on_click="ignore",
        )
    show_artifact(workspace, chosen)


def _diagnostic_evidence(
    path: Path, checked: Verification | None, cycle: int, declared: int | None
) -> None:
    if declared is not None and cycle > declared and path != AGGREGATE:
        st.caption("Selected diagnostic · Historical file outside the applied scope")
        return
    if path == AGGREGATE:
        producer = "analyze_cycles_performance"
    elif path.parts[:2] == ("data", "base"):
        snapshot = int(path.parts[2].removeprefix("cycle_"))
        producer = (
            "bootstrap" if snapshot == 0 else f"prepare_next_cycle_data_{snapshot}"
        )
    else:
        family = {
            "data": "generate_pred_data",
            "evaluation": "evaluate",
            "trained": "train",
            "prediction": "predict",
            "drift_analysis": "analyze_drift",
        }[path.parts[0]]
        producer = f"{family}_cycle_{cycle}"
    evidence = dict(checked.stages).get(producer) if checked else None
    tracked = checked is not None and any(
        f"results/{path}" == item or f"results/{path.parent}" == item
        for item in checked.paths
    )
    status = (
        f"{evidence.availability} · {evidence.correspondence}"
        if evidence
        and tracked
        and (not is_matching(evidence) or evidence.locked_outputs_match)
        else Correspondence.UNKNOWN
    )
    st.caption(f"Selected diagnostic · {status}")


def _compare(
    declared: int | None, observed: set[int], checked: Verification | None
) -> None:
    from dashboard.cycle_comparison import ComparisonCycle, render_comparison

    scope = sorted(set(range(1, (declared or 0) + 1)) | (observed - {0}))
    if not scope:
        st.info(
            "Two learning cycles are needed for a comparison. Bootstrap is a base snapshot."
        )
        return
    state = dashboard_state()
    with st.container(key="results-compare", gap="small"):
        st.markdown("### Compare two cycles")
        with st.container(key="compare-selectors"):
            columns = st.columns(2)
            selected = []
            for column, field, label, default in zip(
                columns,
                ("compare_first", "compare_second"),
                ("Learning cycle A", "Learning cycle B"),
                (scope[-2] if len(scope) > 1 else scope[0], scope[-1]),
                strict=True,
            ):
                if getattr(state, field) not in scope:
                    setattr(state, field, default)
                key = f"_results_{field}"
                st.session_state[key] = getattr(state, field)
                with column:
                    cycle = st.selectbox(
                        label,
                        scope,
                        format_func=lambda item: (
                            cycle_label(item)
                            + (
                                " · historical"
                                if declared is not None and item > declared
                                else ""
                            )
                        ),
                        key=key,
                        on_change=_remember_selection,
                        args=(field, key),
                    )
                    selected.append(
                        ComparisonCycle.from_evidence(cycle, declared, checked)
                    )
        render_comparison((selected[0], selected[1]))


def render() -> None:
    """Switch between the verified overview and one selected scientific diagnostic."""
    st.title("Results")
    workspace = Workspace(Path.cwd())
    state = dashboard_state()
    active = read_active(workspace.params)
    payload = active.contents if active else None
    applied = None
    if payload is not None:
        try:
            applied, _ = AppliedConfiguration.read(payload)
        except ValueError as error:
            st.caption(str(error))
    else:
        st.caption("No applied configuration.")

    st.session_state["_results_mode"] = state.result_mode
    with st.container(
        key="results-toolbar",
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        mode = st.segmented_control(
            "Results mode",
            [EXPERIMENT_OVERVIEW, CYCLE_EXPLORER, COMPARE_CYCLES],
            required=True,
            key="_results_mode",
            label_visibility="collapsed",
            on_change=_remember_selection,
            args=("result_mode", "_results_mode"),
        )
        refresh = st.button("Refresh result evidence")

    checked = state.result_verification
    if refresh:
        with st.spinner("Checking local DVC and scientific evidence…"):
            state.result_verification = checked = verify_workspace(workspace)
            state.result_evidence_notice = None
    if checked is not None and (
        (checked.applied is not None and checked.applied != applied)
        or (
            not refresh
            and checked.stamp is not None
            and not checked.still_observed(workspace)
        )
    ):
        state.result_verification = checked = None
        state.result_evidence_notice = (
            "Workspace evidence changed. Refresh to check correspondence again."
        )

    if checked is None or refresh:
        st.session_state.pop("_drift_raw_snapshot", None)

    observed = set(workspace.result_cycles(include_raw=True))
    aggregate = workspace.available(AGGREGATE)
    summary = ExperimentSummary.from_verification(checked)
    declared = applied.cycles if applied else None
    compact_header = mode == COMPARE_CYCLES or (
        mode == CYCLE_EXPLORER
        and state.result_section
        in (
            DATA,
            PREDICTION,
            MODEL,
            CALIBRATION,
            DRIFT,
        )
    )
    with st.container(key="results-evidence-context", gap="xxsmall"):
        if not compact_header:
            _identity(applied, checked, summary, observed, aggregate)
        if checked is not None and checked.stamp is None and checked.reasons:
            st.warning(checked.reasons[0])
        if state.result_evidence_notice:
            st.caption(state.result_evidence_notice)
        if state.has_unapplied_draft(payload):
            st.caption(
                "Unapplied draft edits · Results are scoped to the applied configuration."
            )
        _details(applied, checked)

    if mode == CYCLE_EXPLORER:
        _explore(
            workspace,
            declared,
            observed,
            aggregate,
            checked,
            summary,
            folds=applied.folds if applied else 0,
            refresh=refresh,
        )
    elif mode == COMPARE_CYCLES:
        _compare(declared, observed, checked)
    else:
        render_overview(
            summary, declared, observed, aggregate, state.result_cycle, _inspect
        )


render()
