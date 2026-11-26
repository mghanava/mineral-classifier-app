"""Run the synthetic pipeline from the applied workspace configuration."""

from pathlib import Path

import streamlit as st

from dashboard.artifacts import Workspace
from dashboard.components import applied_summary, read_active
from dashboard.execution import reproduce
from dashboard.state import CONFIGURE_PAGE, RESULTS_PAGE, dashboard_state

STAGES = {
    "Bootstrap": "bootstrap",
    "Training": "train_cycle_{}",
    "Evaluation": "evaluate_cycle_{}",
    "Prediction": "predict_cycle_{}",
    "Drift analysis": "analyze_drift_cycle_{}",
    "Cross-cycle performance aggregation": "analyze_cycles_performance",
}


def render() -> None:
    """Offer the existing full pipeline and targeted DVC actions."""
    st.title("Run")
    workspace = Workspace(Path.cwd())
    state = dashboard_state()
    active = read_active(workspace.params)
    st.subheader("Applied configuration")
    cycles = applied_summary(active)
    if active is not None and state.has_unapplied_draft(active.contents):
        st.warning(
            "Unapplied draft changes are saved in this session. Runs use the applied params.yaml."
        )
    st.page_link(CONFIGURE_PAGE, label="Review configuration")

    st.divider()
    st.subheader("Full experiment")
    st.write(
        "Regenerate the DVC pipeline and reproduce the stages for the applied settings."
    )
    runnable = active is not None and active.contents is not None
    if st.button("Generate & Run Full Pipeline", type="primary", disabled=not runnable):
        with st.spinner("Running pipeline..."):
            status = reproduce(workspace.params, announce_success=False)
        state.last_run = ("Full experiment", status)
        if status == 0:
            st.success("Pipeline completed successfully!")

    with st.expander("Advanced stage execution"):
        st.caption("DVC may also reproduce prerequisites for the selected stage.")
        stage = st.selectbox("Stage", STAGES, key="_run_stage")
        template = STAGES[stage]
        target = template
        enabled = runnable
        if "{}" in template:
            enabled = runnable and bool(cycles)
            if cycles:
                cycle = st.number_input(
                    "Learning cycle to run", 1, cycles, key="_run_cycle"
                )
                target = template.format(cycle)
            else:
                st.caption("This stage requires at least one applied learning cycle.")
        elif stage == "Cross-cycle performance aggregation":
            enabled = runnable and cycles is not None and cycles >= 2
            if not enabled:
                st.caption("Aggregation requires at least two applied learning cycles.")
        if st.button("Run stage", disabled=not enabled):
            status = reproduce(workspace.params, target)
            state.last_run = (target, status)

    if state.last_run is not None:
        label, status = state.last_run
        outcome = "completed" if status == 0 else f"failed (status {status})"
        st.caption(f"Most recent session action · {label} · {outcome}")
    st.page_link(RESULTS_PAGE, label="Open Results")


render()
