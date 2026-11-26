"""Session-owned values that survive Streamlit's page widget cleanup."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from dashboard.execution import RunAttempt
from src.configuration_files import ConfigurationPreview

if TYPE_CHECKING:
    from dashboard.provenance import Verification

CONFIGURE_PAGE = "dashboard/views/configure.py"
RUN_PAGE = "dashboard/views/run.py"
RESULTS_PAGE = "dashboard/views/results.py"
EXPERIMENT_OVERVIEW = "Experiment overview"
CYCLE_EXPLORER = "Cycle explorer"


@dataclass
class DashboardState:
    """Hold the draft, its review binding, result selection and last run outcome."""

    draft: str | None = None
    preset: str | None = None
    loaded_preset: tuple[str, str] | None = None
    preset_generation: int = 0
    configure_level: str = "Basic"
    advanced_group: str | None = None
    expert_edited: bool = False
    preview_error: str | None = None
    preview: ConfigurationPreview | None = None
    applied_draft: tuple[str, bytes] | None = None
    result_mode: str = EXPERIMENT_OVERVIEW
    result_cycle: int | None = None
    result_section: str = "Data and spatial graphs"
    last_run: RunAttempt | None = None
    result_verification: "Verification | None" = None
    result_evidence_notice: str | None = None

    def update_draft(self, draft: str, *, expert: bool = False) -> None:
        """Discard reviews whenever the editor contents change."""
        if self.draft != draft:
            self.draft = draft
            self.preview = None
            self.preview_error = None
            self.expert_edited = expert

    def has_unapplied_draft(self, active: bytes | None) -> bool:
        """Recognize both the original editor text and a successfully applied review."""
        if self.draft is None:
            return False
        active_text = (active or b"").decode("utf-8", errors="replace")
        return self.draft != active_text and self.applied_draft != (self.draft, active)


def dashboard_state() -> DashboardState:
    """Return the non-widget application state for this session."""
    return st.session_state.setdefault("mineral_dashboard", DashboardState())
