"""Present one existing result diagnostic at a time."""

from collections.abc import Iterable
from pathlib import Path

import streamlit as st

from dashboard.artifacts import AGGREGATE, Workspace
from dashboard.components import UNVERIFIED, applied_summary, read_active, show_artifact
from dashboard.state import CONFIGURE_PAGE, RUN_PAGE, dashboard_state

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


def _data(workspace: Workspace, cycle: int) -> list[Path]:
    if cycle == 0:
        st.caption(
            "Bootstrap creates the initial base snapshot used by learning cycle 1."
        )
        directory = Path("data/base/cycle_0")
    else:
        populations = {
            f"Input base · snapshot {cycle - 1}": Path(f"data/base/cycle_{cycle - 1}"),
            f"Incoming batch {cycle}": Path(f"data/prediction/cycle_{cycle}"),
            f"Post-append base · snapshot {cycle}": Path(f"data/base/cycle_{cycle}"),
        }
        population = st.selectbox("Population", populations)
        directory = populations[population]
        if population.startswith("Post-append"):
            st.caption("The append uses simulator labels from the incoming batch.")
    files = workspace.files(directory)
    return [*_figure_exports(files), *(path for path in files if path.suffix == ".png")]


def _diagnostics(workspace: Workspace, section: str, cycle: int) -> list[Path]:
    if section == DATA:
        return _data(workspace, cycle)
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
    if section == PREDICTION:
        st.caption(
            "The stored graph shows simulator labels. The standalone prediction table has no coordinates."
        )
        return _figure_exports(
            workspace.files(Path(f"data/prediction/cycle_{cycle}"))
        ) + [
            path
            for path in workspace.files(Path(f"prediction/cycle_{cycle}"))
            if path.name
            not in {"metrics.json", "confussion_matrix_.png", "confusion_matrix.png"}
        ]
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


def _interpret(path: Path) -> None:
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


def _remember_section() -> None:
    dashboard_state().result_section = st.session_state["_results_section"]


def render() -> None:
    """Route the selected cycle, scientific section and existing diagnostic."""
    st.title("Results")
    workspace = Workspace(Path.cwd())
    state = dashboard_state()
    active = read_active(workspace.params)
    cycles = applied_summary(active, show_targets=False)
    observed = set(workspace.result_cycles())
    aggregate = workspace.available(AGGREGATE)
    if observed or aggregate:
        st.info(f"Existing workspace artifacts. {UNVERIFIED}")
    else:
        st.info("No display artifacts found in this workspace.")
        if active is not None and active.contents is not None:
            st.page_link(RUN_PAGE, label="Go to Run")
        else:
            st.page_link(CONFIGURE_PAGE, label="Go to Configure")

    scope = sorted(set(range((cycles or 0) + 1)) | observed)
    if state.result_cycle not in scope:
        state.result_cycle = next((cycle for cycle in scope if cycle > 0), 0)
    cycle_column, section_column = st.columns(2)
    with cycle_column:
        cycle = st.selectbox(
            "Learning cycle",
            scope,
            index=scope.index(state.result_cycle),
            format_func=lambda item: (
                "Bootstrap · cycle 0"
                if item == 0
                else f"Cycle {item}"
                + (" · existing files only" if item > (cycles or 0) else "")
            ),
            key="_results_cycle",
        )
        state.result_cycle = cycle
    with section_column:
        sections = [DATA, MODEL, CALIBRATION, PREDICTION, DRIFT] if cycle else [DATA]
        if aggregate or (cycles is not None and cycles >= 2):
            sections.append(PERFORMANCE)
        section = st.selectbox(
            "Result section",
            sections,
            index=sections.index(state.result_section)
            if state.result_section in sections
            else 0,
            key="_results_section",
            on_change=_remember_section,
        )
    if cycle and section != PERFORMANCE:
        st.caption(
            f"Input base {cycle - 1}  ·  Incoming batch {cycle}  ·  Post-append base {cycle}"
        )
    if section == DATA and cycle:
        population_column, diagnostic_column = st.columns(2)
        with population_column:
            choices = _data(workspace, cycle)
        with diagnostic_column:
            chosen = st.selectbox("Diagnostic", choices, format_func=_label)
    else:
        choices = _diagnostics(workspace, section, cycle)
        chosen = (
            st.selectbox("Diagnostic", choices, format_func=_label) if choices else None
        )
    if chosen is None:
        st.warning(
            "No display files found for this selection. The output may be absent or partial."
        )
        return
    _interpret(chosen)
    show_artifact(workspace, chosen)


render()
