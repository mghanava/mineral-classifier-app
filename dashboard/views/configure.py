"""Guide editing and explicit review of the session's scientific draft."""

from collections import Counter
from pathlib import Path

import streamlit as st

from dashboard.artifacts import Workspace
from dashboard.components import applied_summary, read_active, show_validation
from dashboard.guided_configuration import (
    GuidedFields,
    expert_guidance,
    render_advanced,
    render_basic,
)
from dashboard.state import RUN_PAGE, DashboardState, dashboard_state
from src.configuration_files import (
    ConfigurationSnapshot,
    persist_configuration,
    preview_configuration,
)
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml
from src.presets import (
    AppliedPreset,
    apply_preset,
    list_presets,
    load_preset,
    resolve_imported_configuration,
)

_LEVELS = ("Basic", "Advanced", "Expert")
_MEASURED_SECONDS = {"quick_demo": "74.31", "standard": "101.66"}


def _remember_draft(key: str) -> None:
    dashboard_state().update_draft(st.session_state[key], expert=True)


def _remember_level() -> None:
    dashboard_state().configure_level = st.session_state["_configure_level"]


def _presets(state: DashboardState) -> None:
    try:
        presets = {preset_id: load_preset(preset_id) for preset_id in list_presets()}
        options = list(presets)
        selection, action = st.columns([3, 2], vertical_alignment="bottom")
        with selection:
            selected = st.selectbox(
                "Experiment preset",
                options,
                index=options.index(state.preset) if state.preset in presets else 0,
                format_func=lambda preset_id: presets[preset_id].display_name,
                key="_configure_preset",
            )
        state.preset = selected
        preset = presets[selected]
        with action:
            if st.button("Load preset into draft"):
                resolved = apply_preset(preset)
                state.update_draft(dump_configuration_yaml(resolved.configuration))
                state.loaded_preset = (
                    preset.display_name,
                    dump_configuration_yaml(resolved.configuration, sort_keys=True),
                )
                state.preview = None
                state.preview_error = None
                state.expert_edited = False
                # Remount draft controls so the browser displays the replacement values.
                state.preset_generation += 1
        description = f"v{preset.version} · {preset.description}"
        if selected in _MEASURED_SECONDS:
            description += (
                f" Unmodified preset reference · {_MEASURED_SECONDS[selected]} s for two CPU cycles "
                "on the M2.5 host. Runtime varies by machine and settings."
            )
        st.caption(description)
    except ValueError as error:
        st.error(str(error))


def _resolve_draft(state: DashboardState) -> tuple[AppliedPreset | None, str | None]:
    try:
        return resolve_imported_configuration(
            parse_configuration_yaml(state.draft, source="draft YAML")
        ), None
    except ValueError as error:
        return None, str(error)


def _editor(state: DashboardState, resolved: AppliedPreset | None) -> None:
    level = st.radio(
        "Configuration level",
        _LEVELS,
        index=_LEVELS.index(state.configure_level),
        horizontal=True,
        key="_configure_level",
        on_change=_remember_level,
    )
    if level == "Expert":
        if resolved is not None:
            expert_guidance(resolved.configuration)
        key = f"_configure_yaml:{state.preset_generation}"
        st.text_area(
            "Draft YAML",
            value=state.draft,
            height=365,
            key=key,
            on_change=_remember_draft,
            args=(key,),
        )
    elif resolved is None:
        if state.draft.strip():
            st.error(
                "Correct the YAML in Expert or explicitly load a preset to use guided editing."
            )
        else:
            st.info(
                "Load a preset or enter a configuration in Expert to start a draft."
            )
    elif state.expert_edited and not resolved.validation.is_valid:
        st.error(
            "Correct the Expert draft or explicitly load a preset to resume guided editing."
        )
    else:
        fields = GuidedFields(resolved.configuration, state.preset_generation)
        if level == "Basic":
            render_basic(fields)
        else:
            render_advanced(fields, state)


def _review(
    state: DashboardState,
    active: ConfigurationSnapshot,
    resolved: AppliedPreset | None,
    draft_error: str | None,
) -> ConfigurationSnapshot | None:
    st.subheader("Review and apply")
    if st.button(
        "Validate & Preview", type="primary" if state.preview is None else "secondary"
    ):
        state.preview_error = draft_error
        state.preview = (
            preview_configuration(resolved, active) if resolved is not None else None
        )
    preview = state.preview
    if state.preview_error:
        st.write("**1 error · 0 warnings · 0 informational findings**")
        st.error(state.preview_error)
        return active
    if preview is None:
        st.caption("Review the draft before applying it to params.yaml.")
        return active

    stale = preview.active != active
    if stale:
        st.warning(
            "The active configuration changed since preview. Preview again before applying."
        )
    counts = Counter(issue.severity for issue in preview.validation.issues)
    findings = " · ".join(
        f"{counts[key]} {label}{'' if counts[key] == 1 else 's'}"
        for key, label in (
            ("error", "error"),
            ("warning", "warning"),
            ("info", "informational finding"),
        )
    )
    st.write(f"**{findings}**")
    show_validation(preview.validation, compact=True)
    if preview.comparison_warning:
        st.warning(preview.comparison_warning)
    if preview.regeneration_paths:
        paths = ", ".join(preview.regeneration_paths)
        st.warning(
            f"DVC pipeline regeneration is required for changes to {paths}. "
            "Run regenerates the pipeline before reproduction."
        )

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
            if active.contents != preview.yaml.encode("utf-8"):
                state.result_verification = None
                state.result_evidence_notice = (
                    "Applied configuration changed. Refresh to check existing results."
                )
            state.applied_draft = (state.draft, preview.yaml.encode("utf-8"))
            state.preview = None
            st.success("Configuration applied to params.yaml.")
            st.page_link(RUN_PAGE, label="Go to Run")
            return read_active(active.path)

    with st.expander("Differences from active configuration"):
        if preview.diff:
            st.code(preview.diff, language="diff")
        else:
            st.write("No scientific configuration changes.")
    with st.expander("Resolved scientific YAML"):
        st.code(preview.yaml, language="yaml")
    return active


def _status(
    state: DashboardState,
    applied: ConfigurationSnapshot | None,
    resolved: AppliedPreset | None,
) -> None:
    if applied is None:
        return
    if not state.draft.strip():
        status = "No draft yet"
    elif state.has_unapplied_draft(applied.contents):
        status = "Unapplied changes"
    else:
        status = "Matches applied configuration"
    if state.loaded_preset is not None:
        name, baseline = state.loaded_preset
        current = (
            dump_configuration_yaml(resolved.configuration, sort_keys=True)
            if resolved
            else None
        )
        status += f" · {name}" if current == baseline else f" · Customized from {name}"
    with st.container(horizontal=True, gap="large"):
        with st.container(gap="xxsmall", width=340):
            st.markdown("**Session draft**")
            st.write(status)
        with st.container(gap="xxsmall"):
            st.markdown("**Applied configuration · used by Run**")
            if applied.contents is not None:
                applied_summary(applied, show_targets=False)
            else:
                st.write("No applied configuration.")


def render() -> None:
    """Render one level and review the same scientific draft across all levels."""
    st.title("Configure")
    state = dashboard_state()
    active = read_active(Workspace(Path.cwd()).params)
    if active is None:
        return
    if state.draft is None:
        state.update_draft((active.contents or b"").decode("utf-8", errors="replace"))

    with st.container(gap="medium"):
        status = st.container()
        editor, review = st.columns([3, 2], gap="large")
        with editor:
            _presets(state)
            resolved, draft_error = _resolve_draft(state)
            _editor(state, resolved)
        with review, st.container(border=True, gap="small"):
            applied = _review(state, active, resolved, draft_error)
        with status:
            _status(state, applied, resolved)


render()
