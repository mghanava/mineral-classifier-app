"""Introduce the synthetic workflow and show workspace availability."""

from pathlib import Path

import streamlit as st

from dashboard.artifacts import AGGREGATE, Workspace
from dashboard.components import UNVERIFIED, applied_summary, read_active
from dashboard.state import CONFIGURE_PAGE, RESULTS_PAGE, dashboard_state
from src.presets import load_preset


def render() -> None:
    """Open the workspace landing page without applying settings or launching work."""
    workspace = Workspace(Path.cwd())
    st.title("Mineral Classifier")
    st.write(
        "Explore synthetic spatial machine learning, calibrated mineral classification "
        "and geological experimentation with graph neural networks."
    )

    demo, custom = st.columns(2, gap="large")
    with demo, st.container(border=True, gap="small"):
        try:
            preset = load_preset("quick_demo")
        except ValueError as error:
            st.error(str(error))
        else:
            st.subheader(preset.display_name)
            st.write(preset.description)
            if st.button("Configure Quick demo", type="primary"):
                dashboard_state().preset = preset.id
                st.switch_page(CONFIGURE_PAGE)
            st.caption(
                "Unmodified Quick demo reference · 74.31 s for two CPU cycles on the "
                "M2.5 verification host. Runtime varies by machine."
            )
    with custom, st.container(border=True, gap="small"):
        st.subheader("Custom experiment")
        st.write(
            "Start from applied settings or load a preset, then edit guided controls or YAML."
        )
        st.page_link(CONFIGURE_PAGE, label="Configure an experiment")

    st.header("Workspace")
    config, results = st.columns(2, gap="large")
    with config:
        st.subheader("Applied configuration")
        applied_summary(read_active(workspace.params))
    with results:
        st.subheader("Results availability")
        if (
            workspace.available(AGGREGATE)
            or next(workspace.result_cycles(), None) is not None
        ):
            st.write("Existing result files are available.")
            st.caption(UNVERIFIED)
            st.page_link(RESULTS_PAGE, label="Open existing results")
        else:
            st.write("No display artifacts found in this workspace.")

    st.subheader("Synthetic workflow")
    st.write(
        "Validate and apply settings in Configure, reproduce the experiment in Run, "
        "then inspect graphs, calibrated predictions and drift diagnostics in Results. "
        "The simulator supplies labels and the pipeline uses random stratified splits."
    )
    st.caption(
        "Planned workflows · Training on user data and prediction with an existing model"
    )


render()
