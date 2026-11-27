"""Shared configuration summaries, validation findings and artifact presentation."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from plotly.graph_objects import Figure

from dashboard.artifacts import Workspace
from src.configuration import ValidationResult
from src.configuration_files import ConfigurationSnapshot, snapshot_configuration
from src.configuration_yaml import parse_configuration_yaml

UNVERIFIED = (
    "Correspondence with the applied configuration and source revision is unverified."
)


def show_validation(validation: ValidationResult, *, compact: bool = False) -> None:
    """Display path-specific findings and keep blocking issues visible."""
    display = {"error": st.error, "warning": st.warning, "info": st.info}
    secondary = []
    for issue in validation.issues:
        if compact and issue.severity != "error":
            secondary.append(issue)
        else:
            code = f" ({issue.code})" if compact else ""
            display[issue.severity](f"{issue.path} — {issue.message}{code}")
    if secondary:
        with st.expander(f"Warnings and information ({len(secondary)})"):
            for issue in secondary:
                display[issue.severity](
                    f"{issue.path} — {issue.message} ({issue.code})"
                )


def read_active(path: Path) -> ConfigurationSnapshot | None:
    """Read the active file and surface filesystem errors in the current page."""
    try:
        return snapshot_configuration(path)
    except (OSError, ValueError) as error:
        st.error(f"Cannot read the applied configuration. {error}")
        return None


def applied_summary(
    active: ConfigurationSnapshot | None,
    *,
    show_targets: bool = True,
    show_calibration: bool = False,
) -> int | None:
    """Show settings from the active file and return a usable cycle count."""
    if active is None:
        return None
    if active.contents is None:
        st.info("No applied configuration. Validate and apply a draft in Configure.")
        return None
    try:
        settings = parse_configuration_yaml(
            active.contents, source="active params.yaml"
        )
    except ValueError as error:
        st.warning(str(error))
        return None
    cycles = settings.get("cycles")
    count = cycles if type(cycles) is int and cycles >= 0 else None
    model = settings.get("default_model")
    device = settings.get("device")
    evaluation = settings.get("evaluate")
    method = (
        evaluation.get("calibration_method") if isinstance(evaluation, dict) else None
    )
    calibration = (
        f" · **Calibration** {method.capitalize() if isinstance(method, str) else 'Unavailable'}"
        if show_calibration
        else ""
    )
    st.write(
        f"**Model** {model.upper() if isinstance(model, str) else 'Unavailable'}"
        f" · **Requested device** {device if isinstance(device, str) else 'Unavailable'}"
        f" · **Learning cycles** {count if count is not None else 'Unavailable'}{calibration}"
    )
    data = settings.get("data")
    if show_targets and isinstance(data, dict):
        targets = [
            data.get(population, {}).get("n_samples")
            if isinstance(data.get(population), dict)
            else None
            for population in ("base", "pred")
        ]
        if all(type(target) is int for target in targets):
            st.caption(
                f"Requested observations · {targets[0]} bootstrap · {targets[1]} per batch"
            )
    return count


def show_artifact(
    workspace: Workspace, path: Path, *, image_width: int | None = None
) -> None:
    """Read and render one selected diagnostic, reporting intermittent files."""
    if path.suffix == ".html":
        st.caption("Original Plotly HTML may load Plotly from a CDN.")
        if not st.checkbox("Open trusted local legacy HTML", key=f"legacy:{path}"):
            return
    try:
        value = workspace.read(path)
        if isinstance(value, Figure):
            title = value.layout.title.text
            value.update_layout(
                title_text=" ",
                margin_t=8,
                height=500,
            )
            if st.context.theme.type == "dark":
                value.update_layout(template="plotly_dark")
            with st.container(width=1120):
                st.plotly_chart(value, width="stretch", key=f"figure:{path}")
                if title:
                    st.caption(title.replace("<br>", "  \n"))
        elif isinstance(value, pd.DataFrame):
            st.dataframe(value, height=340, width="stretch")
        elif isinstance(value, bytes):
            with Image.open(BytesIO(value)) as image:
                display_width = min(
                    920, max(1, round(540 * image.width / image.height))
                )
            st.image(value, width=image_width or display_width)
            st.download_button(
                "Download original figure",
                value,
                path.name,
                "image/png",
                on_click="ignore",
            )
        elif isinstance(value, dict):
            st.json(value)
        elif path.suffix == ".html":
            components.html(value, height=620, scrolling=True)
        else:
            st.text(value)
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        Image.DecompressionBombError,
    ) as error:
        st.warning(
            f"Could not display {path.name}. The file may be incomplete or malformed."
        )
        with st.expander("Read error"):
            st.text(str(error))
