"""Edit and review the existing expert YAML draft."""

from pathlib import Path

import streamlit as st

from dashboard.artifacts import Workspace
from dashboard.components import applied_summary, read_active, show_validation
from dashboard.state import RUN_PAGE, DashboardState, dashboard_state
from src.configuration_files import (
    ConfigurationSnapshot,
    persist_configuration,
    preview_configuration,
)
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml
from src.presets import (
    apply_preset,
    list_presets,
    load_preset,
    resolve_imported_configuration,
)


def _remember_draft() -> None:
    dashboard_state().update_draft(st.session_state["_configure_yaml"])


def _presets(state: DashboardState) -> None:
    try:
        presets = {preset_id: load_preset(preset_id) for preset_id in list_presets()}
        options = list(presets)
        selected = st.selectbox(
            "Experiment preset",
            options,
            index=options.index(state.preset) if state.preset in presets else 0,
            format_func=lambda preset_id: presets[preset_id].display_name,
            key="_configure_preset",
        )
        state.preset = selected
        st.caption(presets[selected].description)
        if st.button("Load preset into draft"):
            resolved = apply_preset(presets[selected])
            state.update_draft(dump_configuration_yaml(resolved.configuration))
            state.preview = None
            st.session_state.pop("_configure_yaml", None)
    except ValueError as error:
        st.error(str(error))


def _review(
    state: DashboardState,
    active: ConfigurationSnapshot,
    draft: str,
    preview_error: str | None,
) -> ConfigurationSnapshot | None:
    st.subheader("Preview")
    if preview_error:
        st.error(preview_error)
        return active
    preview = state.preview
    if preview is None:
        st.info(
            "Validate the current draft to review its resolved settings and changes."
        )
        return active

    stale = preview.active != active
    if stale:
        st.warning(
            "The active configuration changed since preview. Preview again before applying."
        )
    show_validation(preview.validation, compact=True)
    if preview.comparison_warning:
        st.warning(preview.comparison_warning)
    if preview.regeneration_paths:
        paths = ", ".join(preview.regeneration_paths)
        st.warning(
            f"DVC pipeline regeneration is required for changes to {paths}. "
            "Run regenerates the pipeline before reproduction."
        )
    with st.expander("Differences from active configuration"):
        if preview.diff:
            st.code(preview.diff, language="diff")
        else:
            st.write("No scientific configuration changes.")
    with st.expander("Resolved scientific YAML"):
        st.code(preview.yaml, language="yaml")

    if st.button(
        "Apply configuration",
        type="primary",
        disabled=stale or not preview.validation.is_valid,
    ):
        try:
            persist_configuration(preview, active, overwrite=True)
        except (OSError, ValueError) as error:
            st.error(f"Could not apply the configuration. {error}")
        else:
            state.applied_draft = (draft, preview.yaml.encode("utf-8"))
            state.preview = None
            st.success("Configuration applied to params.yaml.")
            st.page_link(RUN_PAGE, label="Go to Run")
            return read_active(active.path)
    return active


def render() -> None:
    """Keep the exact draft and its active-file binding across page changes."""
    st.title("Configure")
    st.write(
        "Load a preset or edit expert YAML, then validate and explicitly apply the draft."
    )
    state = dashboard_state()
    active = read_active(Workspace(Path.cwd()).params)
    if active is None:
        return
    if state.draft is None:
        state.update_draft((active.contents or b"").decode("utf-8", errors="replace"))

    editor, review = st.columns(2, gap="large")
    with editor:
        _presets(state)
        draft = st.text_area(
            "Draft YAML",
            value=state.draft,
            height=370,
            key="_configure_yaml",
            on_change=_remember_draft,
        )
        preview_error = None
        if st.button("Validate & Preview"):
            state.preview = None
            try:
                resolved = resolve_imported_configuration(
                    parse_configuration_yaml(draft, source="draft YAML")
                )
                state.preview = preview_configuration(resolved, active)
            except ValueError as error:
                preview_error = str(error)
    with review:
        applied = _review(state, active, draft, preview_error)
        st.divider()
        st.subheader("Applied configuration")
        applied_summary(applied)
        if applied is not None and state.has_unapplied_draft(applied.contents):
            st.caption("The editor has unapplied draft changes.")


render()
