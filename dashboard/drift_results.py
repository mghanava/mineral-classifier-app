"""Present verified drift findings and one selected diagnostic."""

from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from numpy.typing import NDArray

from dashboard.artifacts import Workspace
from dashboard.components import show_artifact
from dashboard.drift_summary import (
    METHOD_LABELS,
    MODEL_REPRESENTATION,
    RAW_REPRESENTATION,
    REPORT_THRESHOLD,
    DriftSummary,
)
from dashboard.experiment_summary import BLUE, ORANGE
from dashboard.model_results import scope_summary
from dashboard.provenance import Verification
from dashboard.spatial_data import PopulationKind, read_population

NATIVE = "Selected raw feature"
SETTINGS = "Technical settings"
REPORT = "Original test report"
DIAGNOSTICS = {
    "Original raw feature distributions": (
        "marginal_distributions.png",
        f"{RAW_REPRESENTATION} · Histograms and means use raw values. "
        "The KS and Mann-Whitney annotations use the corresponding transformed features.",
    ),
    "Within- and cross-set distances": (
        "distance_distributions.png",
        f"{MODEL_REPRESENTATION} · Sampled Euclidean distances within reference, "
        "within incoming and across the two populations.",
    ),
    "Mutual information": (
        "mutual_information.png",
        f"{MODEL_REPRESENTATION} · Normalized mutual information in each population "
        "and the reference-minus-incoming difference.",
    ),
    "PCA and Kernel PCA": (
        "pca_kpca_grid.png",
        f"{MODEL_REPRESENTATION} · Each projection is fitted jointly to the two "
        "populations in this cycle and has cycle-specific axes.",
    ),
}


def shared_histogram(
    reference: NDArray, incoming: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Return common edges and per-cohort proportions for raw display values."""
    edges = np.histogram_bin_edges(np.concatenate([reference, incoming]), bins=30)
    return (
        edges,
        np.histogram(reference, bins=edges)[0] / reference.size,
        np.histogram(incoming, bins=edges)[0] / incoming.size,
    )


def _native(workspace: Workspace, checked: Verification, drift: DriftSummary) -> None:
    key = "_drift_raw_snapshot"
    cached = st.session_state.get(key)
    try:
        if cached is None or cached[:2] != (checked, drift.cycle):
            reference = read_population(
                workspace, PopulationKind.INPUT_BASE, cycle=drift.cycle
            )
            incoming = read_population(
                workspace, PopulationKind.INCOMING, cycle=drift.cycle
            )
            if (
                reference.dataset_sha256 != drift.reference_sha256
                or reference.snapshot != drift.cycle - 1
                or incoming.dataset_sha256 != drift.incoming_sha256
                or incoming.parent_sha256 != reference.dataset_sha256
                or (reference.count, incoming.count)
                != (drift.reference_count, drift.incoming_count)
                or reference.feature_names != drift.feature_names
                or incoming.feature_names != drift.feature_names
            ):
                raise ValueError(
                    "The raw populations differ from the verified drift cohorts."
                )
            cached = (
                checked,
                drift.cycle,
                reference.raw_features,
                incoming.raw_features,
            )
        if not checked.still_observed(workspace):
            raise ValueError(
                "Workspace evidence changed. Refresh before reopening the raw comparison."
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        AttributeError,
        EOFError,
        UnpicklingError,
    ) as error:
        st.session_state.pop(key, None)
        st.info(str(error))
        return
    st.session_state[key] = cached
    _, _, reference, incoming = cached
    with st.container(width=360):
        feature = st.selectbox("Raw feature", drift.feature_names, key="drift-feature")
    column = drift.feature_names.index(feature)
    edges, ref_proportion, in_proportion = shared_histogram(
        reference[:, column], incoming[:, column]
    )
    figure = go.Figure()
    for label, proportions, color, pattern in (
        (f"Reference · n={drift.reference_count:,}", ref_proportion, BLUE, ""),
        (f"Incoming · n={drift.incoming_count:,}", in_proportion, ORANGE, "/"),
    ):
        figure.add_bar(
            x=(edges[:-1] + edges[1:]) / 2,
            y=proportions,
            name=label,
            marker={"color": color, "pattern": {"shape": pattern, "solidity": 0.18}},
            customdata=np.column_stack((edges[:-1], edges[1:])),
            hovertemplate="%{customdata[0]:.3g} to %{customdata[1]:.3g}<br>%{y:.2%}",
        )
    figure.update_layout(
        height=320,
        margin={"l": 6, "r": 12, "t": 6, "b": 8},
        barmode="group",
        bargap=0.12,
        bargroupgap=0.04,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom"},
    )
    figure.update_xaxes(title=feature, automargin=True)
    figure.update_yaxes(
        title="Cohort proportion per bin",
        tickformat=".0%",
        rangemode="tozero",
        automargin=True,
    )
    st.caption(
        f"{RAW_REPRESENTATION} · Shared bin edges. Heights show the fraction of each "
        "cohort in each bin. Paired bars use the same interval. This descriptive display uses no statistical test."
    )
    with st.container(width=900):
        st.plotly_chart(figure, config={"displayModeBar": False})
        with st.expander("Histogram values"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Bin start": edges[:-1],
                        "Bin end": edges[1:],
                        "Reference": ref_proportion,
                        "Incoming": in_proportion,
                    }
                ),
                hide_index=True,
                height=240,
                column_config={
                    name: st.column_config.NumberColumn(format="percent")
                    for name in ("Reference", "Incoming")
                },
            )


def _settings(drift: DriftSummary, workspace: Workspace, metadata_path: Path) -> None:
    st.markdown("#### Effective settings")
    with st.container(key="drift-settings"):
        settings, interpretation = st.columns([1.2, 1], gap="large")
        with settings:
            st.markdown(
                f"**Representation** · {MODEL_REPRESENTATION} (float64) · {drift.scaler or 'Identity'}  \n"
                f"**Effective RBF gamma** · {drift.effective_gamma:.8g}  \n"
                f"**Feature order** · {', '.join(drift.feature_names)}  \n"
                f"**Transport** · {drift.wasserstein_context}  \n"
                f"**Permutation budget per test** · {drift.budget:,}  \n"
                f"**Configured / effective seed** · {drift.configured_seed if drift.configured_seed is not None else 'Unseeded'} / "
                f"{drift.effective_seed if drift.effective_seed is not None else 'Unseeded'}  \n"
                f"**Early stopping** · {drift.early_stopping.describe(drift.budget)}"
            )
        with interpretation:
            st.markdown(
                "**Permutation assumptions**  \n"
                "The tests shuffle individual observations and assume exchangeability under the null. "
                "Spatial dependence can violate that assumption and limit interpretation of the p-values. "
                "Small permutation budgets yield coarse, variable p-values, especially near the report cutoff."
            )
            st.caption(
                "The report cutoff applies to the published p-value. The early-stop threshold "
                "applies to the running p-value during computation. The artifacts record "
                "the budget and policy. Realized counts and null distributions are unavailable."
            )
    st.download_button(
        "Download original settings JSON",
        lambda: workspace.read_bytes(metadata_path),
        metadata_path.name,
        "application/json",
        on_click="ignore",
    )


def render_drift(
    workspace: Workspace, cycle: int, checked: Verification | None
) -> bool:
    """Render checked findings, leaving the existing fallback for other evidence states."""
    summary, evidence = scope_summary(checked, cycle, "analyze_drift")
    drift = summary.drift if summary else None
    if checked is None or drift is None:
        st.session_state.pop("_drift_raw_snapshot", None)
        st.info(
            "Verified drift findings are unavailable. Original diagnostics retain their evidence labels."
        )
        return False

    st.markdown("### Distribution drift")
    st.caption(
        f"Cycle {cycle} · {evidence.correspondence} · {MODEL_REPRESENTATION} (float64) · {drift.scaler or 'Identity'}"
    )
    with st.container(
        key="drift-populations", horizontal=True, gap="medium", width=620
    ):
        st.metric(
            f"Reference · base snapshot {cycle - 1}",
            f"{drift.reference_count:,}",
            help=f"Dataset SHA-256 {drift.reference_sha256}",
        )
        st.metric(
            f"Incoming · batch {cycle}",
            f"{drift.incoming_count:,}",
            help=f"Dataset SHA-256 {drift.incoming_sha256}",
        )

    st.markdown(f"#### Permutation tests · report cutoff `p < {REPORT_THRESHOLD:g}`")
    with st.container(width=900):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Method": METHOD_LABELS[test.method],
                        "Statistic": test.statistic,
                        "p-value": test.p_value,
                        "Finding": test.finding,
                    }
                    for test in drift.tests
                ]
            ),
            hide_index=True,
            column_config={
                "Method": st.column_config.TextColumn(width=106),
                "Statistic": st.column_config.NumberColumn(format="%.6g", width=76),
                "p-value": st.column_config.NumberColumn(format="%.6g", width=65),
                "Finding": st.column_config.TextColumn(width=93),
            },
        )
    st.markdown(
        f"**Interpretation limits** · Budget {drift.budget:,} permutations per test. Small budgets give coarse "
        "p-values, and spatial dependence can violate the row-shuffling assumption. Failure to detect drift "
        "leaves distribution equality unresolved. Detection alone gives no evidence of predictive degradation."
    )
    directory = Path(f"drift_analysis/cycle_{cycle}")
    with st.container(width=400):
        diagnostic = st.selectbox(
            "Supporting diagnostic",
            ["Overview only", NATIVE, *DIAGNOSTICS, SETTINGS, REPORT],
            key="verified-drift-diagnostic",
        )
    if diagnostic == NATIVE:
        _native(workspace, checked, drift)
    elif diagnostic == SETTINGS:
        _settings(drift, workspace, directory / "drift_metadata.json")
    elif diagnostic == REPORT:
        path = directory / "drift_results.txt"
        show_artifact(workspace, path)
        st.download_button(
            "Download original test report",
            lambda: workspace.read_bytes(path),
            path.name,
            "text/plain",
            on_click="ignore",
        )
    elif diagnostic in DIAGNOSTICS:
        filename, caption = DIAGNOSTICS[diagnostic]
        st.caption(caption)
        show_artifact(workspace, directory / filename, image_width=1040)
    return True
