"""Run the synthetic pipeline and show session-scoped execution evidence."""

from collections import Counter
from pathlib import Path

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException

from dashboard.artifacts import AGGREGATE, Workspace
from dashboard.components import (
    UNVERIFIED,
    applied_summary,
    read_active,
    show_validation,
)
from dashboard.execution import (
    DISPLAY_LIMIT,
    PHASES,
    REPRODUCING,
    RunAttempt,
    StageProgress,
    configuration_id,
    reproduce,
    stage_details,
)
from dashboard.state import CONFIGURE_PAGE, RESULTS_PAGE, dashboard_state

STAGES = {
    "Bootstrap": "bootstrap",
    "Training": "train_cycle_{}",
    "Evaluation": "evaluate_cycle_{}",
    "Prediction": "predict_cycle_{}",
    "Drift analysis": "analyze_drift_cycle_{}",
    "Cross-cycle performance aggregation": "analyze_cycles_performance",
}


def _advanced(runnable: bool, cycles: int | None) -> str | None:
    with st.expander("Advanced stage execution"):
        st.caption("DVC may also reproduce prerequisites for the selected stage.")
        with st.container(width=360):
            stage = st.selectbox("Stage", STAGES, key="_run_stage")
            target = STAGES[stage]
            enabled = runnable
            if "{}" in target:
                enabled = runnable and bool(cycles)
                if cycles:
                    cycle = st.number_input(
                        "Learning cycle to run", 1, cycles, key="_run_cycle"
                    )
                    target = target.format(cycle)
                else:
                    st.caption(
                        "This stage requires at least one applied learning cycle."
                    )
            elif stage == "Cross-cycle performance aggregation":
                enabled = runnable and cycles is not None and cycles >= 2
                if not cycles or cycles < 2:
                    st.caption(
                        "Aggregation requires at least two applied learning cycles."
                    )
            if st.button("Run stage", disabled=not enabled):
                return target
    return None


def _stage_progress(progress: StageProgress, *, running: bool) -> None:
    if progress.last_event:
        st.write(f"**Last observed** · {progress.last_event}")
    else:
        st.caption(
            "Waiting for identifiable DVC stage output."
            if running
            else "No identifiable DVC stage output was observed."
        )

    cycle = progress.last_learning_cycle
    if cycle is None:
        return
    selected = []
    other_cycles = set()
    other_statuses: Counter[str] = Counter()
    for stage, status in progress.stages.items():
        name, stage_cycle = stage_details(stage)
        if stage_cycle == cycle:
            display = "Start observed" if status == "Started" else status
            selected.append(f"{name} ({display.lower()})")
        elif stage_cycle:
            other_cycles.add(stage_cycle)
            other_statuses[status] += 1
    st.markdown(f"**Learning cycle {cycle}**")
    for index in range(0, len(selected), 3):
        st.caption(" · ".join(selected[index : index + 3]))
    if other_cycles:
        confirmed = other_statuses["Completed"] + other_statuses["Restored from cache"]
        st.caption(
            f"Other learning cycles observed · {len(other_cycles)} · "
            f"{confirmed} stages confirmed · {other_statuses['Skipped unchanged']} unchanged skips"
        )


def _status(
    slot: DeltaGenerator, attempt: RunAttempt | None, applied_id: str | None
) -> None:
    with slot.container():
        if attempt is None:
            st.info("Ready")
            st.caption(
                "Choose a full experiment or a targeted action using the applied settings."
            )
            return
        show = {
            "Running": st.info,
            "Completed": st.success,
            "Failed": st.error,
            "Up to date": st.success,
        }
        show[attempt.status](f"{attempt.status} · {attempt.scope}")
        st.caption(
            " → ".join(
                f"**{phase}**" if phase == attempt.phase else phase for phase in PHASES
            )
        )
        if attempt.config_id and attempt.config_id != applied_id:
            st.caption(f"Configuration used for this action · `{attempt.config_id}`")
        if attempt.error:
            st.error(attempt.error)
            if attempt.validation and not attempt.validation.is_valid:
                show_validation(attempt.validation, compact=True)
        if attempt.phase == REPRODUCING:
            if attempt.status == "Up to date":
                st.caption(
                    "DVC reported unchanged skips and reproduction exited successfully."
                )
            _stage_progress(attempt.progress, running=attempt.status == "Running")


def _logs(slot: DeltaGenerator, attempt: RunAttempt | None) -> None:
    with slot.container():
        if attempt is None:
            st.caption("Command output will appear here during an execution.")
            return
        if not attempt.commands:
            st.caption("No subprocess command was requested for this action.")
        if attempt.commands:
            commands = [
                f"{command.display} · "
                + (
                    f"exit {command.exit_status}"
                    if command.exit_status is not None
                    else "exit unavailable"
                )
                for command in attempt.commands
            ]
            st.code("\n".join(commands), language="text")
        if attempt.log:
            omitted = max(0, len(attempt.log) - DISPLAY_LIMIT)
            if omitted or attempt.discarded:
                st.caption(
                    "Showing the latest output. "
                    f"{omitted + attempt.discarded:,} earlier characters are omitted from this view. "
                    f"{attempt.discarded:,} are outside the retained session log."
                )
            st.code(attempt.log[-DISPLAY_LIMIT:], language="text", height=220)
        if attempt.commands and attempt.status != "Running":
            st.download_button(
                "Download retained session log",
                data=attempt.log_download(),
                file_name="mineral-classifier-last-action.log",
                mime="text/plain",
                on_click="ignore",
            )


def _availability(slot: DeltaGenerator, workspace: Workspace) -> None:
    with slot.container():
        if (
            workspace.available(AGGREGATE)
            or next(workspace.result_cycles(), None) is not None
        ):
            st.caption(f"Display files are available. {UNVERIFIED}")
        else:
            st.caption("No display files found in this workspace.")
        st.page_link(RESULTS_PAGE, label="Open Results")


def _execute(
    workspace: Workspace,
    attempt: RunAttempt,
    status: DeltaGenerator,
    logs: DeltaGenerator,
    applied_id: str | None,
) -> None:
    interrupted = None

    def refresh() -> None:
        nonlocal interrupted
        if interrupted is None:
            try:
                _status(status, attempt, applied_id)
                _logs(logs, attempt)
            except ScriptControlException as error:
                # Let the synchronous subprocess finish and release the lock even
                # when navigation stops rendering this ScriptRunner.
                interrupted = error

    reproduce(workspace.params, attempt, refresh)
    if interrupted is not None:
        raise interrupted


def render() -> None:
    """Show applied settings, one full action and the latest session outcome."""
    st.title("Run")
    workspace = Workspace(Path.cwd())
    state = dashboard_state()
    active = read_active(workspace.params)
    applied_id = (
        configuration_id(active.contents)
        if active and active.contents is not None
        else None
    )

    with st.container(width=1050):
        st.subheader("Applied configuration")
        cycles = applied_summary(active, show_calibration=True)
        if applied_id:
            st.caption(
                f"Applied configuration ID · `{applied_id}` · SHA-256 of params.yaml (12 characters)"
            )
        if active is not None and state.has_unapplied_draft(active.contents):
            st.warning(
                "Unapplied draft changes are saved in this session. Runs use the applied params.yaml."
            )
        st.page_link(CONFIGURE_PAGE, label="Review configuration")

        st.subheader("Experiment execution")
        running = state.last_run is not None and state.last_run.status == "Running"
        runnable = applied_id is not None and not running
        full_action = st.empty()
        full = full_action.button(
            "Run full experiment", type="primary", disabled=not runnable
        )
        status = st.empty()
        advanced = st.empty()
        with advanced.container():
            target = _advanced(runnable, cycles)
        with st.expander("Technical logs"):
            logs = st.empty()
        results = st.empty()

        if runnable and (full or target):
            full_action.empty()
            advanced.empty()
            attempt = RunAttempt(target=None if full else target)
            state.last_run = attempt
            state.result_verification = None
            _execute(workspace, attempt, status, logs, applied_id)
            st.rerun()
        else:
            _status(status, state.last_run, applied_id)
            _logs(logs, state.last_run)
        _availability(results, workspace)


render()
