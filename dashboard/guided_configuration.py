"""Bind native guided controls to the authoritative scientific YAML draft."""

import json
from collections.abc import Callable
from typing import Any

import streamlit as st

from dashboard.state import DashboardState, dashboard_state
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml
from src.presets import apply_preset, load_preset, resolve_imported_configuration

GROUPS = (
    "Synthetic environment and class schema",
    "Partitions and validation",
    "Spatial graph and preprocessing",
    "Model training",
    "Probability calibration",
    "Drift analysis",
    "Reproducibility",
)
_SCALERS = {name: name for name in ("RobustScaler", "StandardScaler", "MinMaxScaler")}
_CALIBRATORS = {
    "Temperature scaling": "temperature",
    "Platt scaling": "platt",
    "Beta calibration": "beta",
    "Dirichlet": "dirichlet",
    "Isotonic": "isotonic",
}
_RULES = {
    "K-nearest neighbors": "k_nearest",
    "Fixed radius": "connection_radius",
    "Distance percentile": "distance_percentile",
}


def _read(configuration: dict, path: str) -> Any:
    value = configuration
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _change(path: str, value: object) -> dict:
    keys = path.split(".")
    nested = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        nested = {key: nested}
    return nested


def _patch(overrides: dict) -> None:
    state = dashboard_state()
    try:
        document = parse_configuration_yaml(state.draft, source="draft YAML")
    except ValueError:
        return
    if (
        state.expert_edited
        and not resolve_imported_configuration(document).validation.is_valid
    ):
        return
    updated = resolve_imported_configuration(document, overrides)
    state.update_draft(dump_configuration_yaml(updated.configuration))


def _number(text: str, *, integer: bool = False) -> int | float | str | None:
    if not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        if not integer:
            try:
                return float(text)
            except ValueError:
                pass
    return text


def _names(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return text


def _canonical(path: str) -> Any:
    return _read(apply_preset(load_preset("research")).configuration, path)


class GuidedFields:
    """Render short controls and route each change through the shared resolver."""

    def __init__(self, configuration: dict, preset_generation: int) -> None:
        """Use a resolved view of the active session draft for widget defaults."""
        self.configuration = configuration
        self.preset_generation = preset_generation

    def read(self, path: str) -> Any:
        """Look up a scientific field without inserting a value."""
        return _read(self.configuration, path)

    def _binding(
        self, path: str, convert: Callable = lambda value: value
    ) -> dict[str, Any]:
        return self._callback(path, lambda value: _patch(_change(path, convert(value))))

    def _callback(self, path: str, change: Callable) -> dict[str, Any]:
        key = f"_guided:{self.preset_generation}:{path}"

        def remember() -> None:
            change(st.session_state[key])

        return {"key": key, "on_change": remember}

    def number(
        self, path: str, label: str, *, integer: bool = False, help: str | None = None
    ) -> None:
        """Use exact text for real, nullable, or non-browser-sized values."""
        value = self.read(path)
        if (
            integer
            and type(value) is int
            and abs(value) < 2**53 - 1
            and not path.endswith(".seed")
        ):
            st.number_input(
                label, value=value, step=1, help=help, **self._binding(path)
            )
        else:
            st.text_input(
                label,
                value="" if value is None else str(value),
                help=help,
                **self._binding(path, lambda text: _number(text, integer=integer)),
            )

    def choice(
        self,
        path: str,
        label: str,
        options: dict,
        *,
        selected: object = None,
        change: Callable | None = None,
    ) -> None:
        """Show user-facing tokens while keeping scientific values typed."""
        if change is None:
            selected = self.read(path)
        labels = list(options)
        index = next(
            (i for i, value in enumerate(options.values()) if value == selected), None
        )
        callback = (
            change if change is not None else lambda value: _patch(_change(path, value))
        )

        def remember(label: str | None) -> None:
            if label is not None:
                callback(options[label])

        st.selectbox(label, labels, index=index, **self._callback(path, remember))

    def flag(self, path: str, label: str, *, change: Callable | None = None) -> None:
        """Expose an existing boolean field."""
        callback = (
            self._binding(path) if change is None else self._callback(path, change)
        )
        st.checkbox(label, value=self.read(path) is True, **callback)

    def class_names(self) -> None:
        """Support ordered names and both generated-name values."""
        path = "data.base.class_names"
        st.text_input(
            "Class names",
            value=json.dumps(self.read(path), ensure_ascii=False),
            help='Use null or [] for generated names, or list custom names in class order, e.g. ["Quartz", "Pyrite"].',
            **self._binding(path, _names),
        )


def _device_index(value: str) -> None:
    _patch({"device": f"cuda:{value.strip()}" if value.strip() else "cuda"})


def render_basic(fields: GuidedFields) -> None:
    """Show five common settings and the optional visible CUDA device index."""
    model, device, cycles = st.columns(3)
    with model:
        fields.choice("default_model", "Model", {"GCN": "gcn", "GAT": "gat"})
    with device:
        requested = fields.read("device")
        cuda = isinstance(requested, str) and (
            requested == "cuda" or requested.startswith("cuda:")
        )
        fields.choice(
            "device",
            "Computing device",
            {"Auto": "auto", "CPU": "cpu", "CUDA": "cuda"},
            selected="cuda" if cuda else requested,
            change=lambda value: _patch({"device": value}),
        )
        if cuda:
            st.text_input(
                "CUDA device index",
                value=requested.partition(":")[2],
                help="Leave blank for the first visible CUDA device (index 0).",
                **fields._callback("cuda_index", _device_index),
            )
    with cycles:
        fields.number(
            "cycles", "Learning cycles", integer=True, help="Zero runs bootstrap only."
        )
    bootstrap, batch = st.columns(2)
    with bootstrap:
        fields.number(
            "data.base.n_samples", "Requested bootstrap observations", integer=True
        )
    with batch:
        fields.number(
            "data.pred.n_samples",
            "Requested observations per prediction batch",
            integer=True,
        )
    if type(fields.read("cycles")) is int and fields.read("cycles") == 0:
        st.caption("Bootstrap only · No training or incoming prediction batches.")
    st.caption(
        "Placement and per-class top-ups can change the realized observation counts."
    )


def _environment(fields: GuidedFields) -> None:
    radius, depth, spacing = st.columns(3)
    with radius:
        fields.number("data.base.radius", "Generator radius")
    with depth:
        fields.number("data.base.depth", "Generator depth")
    with spacing:
        fields.number("data.base.spacing", "Minimum XY spacing")
    features, classes, floor = st.columns(3)
    with features:
        fields.number("data.base.n_features", "Features", integer=True)
    with classes:
        fields.number("data.base.n_classes", "Classes", integer=True)
    with floor:
        fields.number(
            "data.base.min_samples_per_class",
            "Bootstrap class floor",
            integer=True,
            help="Clear the field to disable the minimum class top-up.",
        )
    fields.class_names()


def _partitions(fields: GuidedFields) -> None:
    folds, holdout, calibration = st.columns(3)
    with folds:
        fields.number("data.base.n_splits", "Cross-validation folds", integer=True)
    with holdout:
        fields.number("data.base.test_size", "Holdout size of the full base")
    with calibration:
        fields.number("data.base.calib_size", "Calibration size of the holdout")
    st.caption(
        "For either size, enter an integer count (e.g. 25), a fraction (e.g. 0.25), "
        "or leave blank for the existing 25% default. Development uses the rest of the "
        "full base. Final test uses the rest of the holdout."
    )


def _graph(fields: GuidedFields) -> None:
    labels = {
        "k_nearest": "Neighbors per node",
        "connection_radius": "Connection radius",
        "distance_percentile": "Distance percentile",
    }
    base = "data.base"
    active = [key for key in labels if fields.read(f"{base}.{key}") is not None]
    rule = active[0] if len(active) == 1 else None

    def set_rule(selected: str) -> None:
        start = {
            "k_nearest": 1,
            "connection_radius": 1.0,
            "distance_percentile": _canonical(f"{base}.distance_percentile"),
        }
        _patch(
            {
                "data": {
                    "base": {
                        key: start[selected] if key == selected else None
                        for key in labels
                    }
                }
            }
        )

    method, parameter = st.columns(2)
    with method:
        fields.choice(
            "connectivity", "Connectivity rule", _RULES, selected=rule, change=set_rule
        )
    with parameter:
        if rule:
            fields.number(f"{base}.{rule}", labels[rule], integer=rule == "k_nearest")
    fields.flag("add_self_loops", "Add graph self-loops")

    def set_affinity(method: str) -> None:
        path = "data.edge_affinity"
        scale = None if method == "unit" else _canonical(f"{path}.scale")
        _patch({"data": {"edge_affinity": {"method": method, "scale": scale}}})

    affinity, scale = st.columns(2)
    with affinity:
        fields.choice(
            "data.edge_affinity.method",
            "Edge affinity",
            {"Unit weights": "unit", "Inverse quadratic": "inverse_quadratic"},
            selected=fields.read("data.edge_affinity.method"),
            change=set_affinity,
        )
    with scale:
        if fields.read("data.edge_affinity.method") == "inverse_quadratic":
            fields.number("data.edge_affinity.scale", "Affinity scale")
    st.caption(
        "Graphs are built independently per partition. GCN uses edge affinity and GAT uses the topology."
    )
    with st.container(width=330):
        fields.choice(
            "data.scaler_type",
            "Feature scaler",
            {**_SCALERS, "Identity (no scaling)": None},
        )
    st.caption(
        "Changing the scaler resets its options to constructor defaults. Customize those options in Expert."
    )


def _training(fields: GuidedFields) -> None:
    model = fields.read("default_model")
    if isinstance(model, str):
        st.write(f"Selected model · **{model.upper()}**")
    with st.container(width=330):
        fields.number("train.n_epochs", "Training epoch limit", integer=True)
    st.caption(
        "The cross-validation runs inform the training duration for a fresh refit on the full development pool."
    )


def _calibration(fields: GuidedFields) -> None:
    method, bins, budget = st.columns(3)
    with method:
        fields.choice("evaluate.calibration_method", "Calibration method", _CALIBRATORS)
    with bins:
        fields.number("evaluate.n_bins", "Reliability bins", integer=True)
    with budget:
        if fields.read("evaluate.calibration_method") != "isotonic":
            fields.number(
                "evaluate.n_epochs", "Calibration fit epoch budget", integer=True
            )
    if fields.read("evaluate.calibration_method") == "isotonic":
        st.caption(
            "Isotonic fits on the reserved calibration partition. Parametric settings remain in the draft."
        )
    else:
        st.caption(
            "Parametric calibration uses a fit and monitoring split within the reserved calibration partition."
        )


def _drift(fields: GuidedFields) -> None:
    sliced_path = "analyze_drift.use_max_sliced_wasserstein"
    sinkhorn_path = "analyze_drift.use_sinkhorn_wasserstein"
    sliced, sinkhorn = fields.read(sliced_path), fields.read(sinkhorn_path)
    method = "sliced" if sliced else "sinkhorn" if sinkhorn else "exact"
    budget, transport = st.columns(2)
    with budget:
        fields.number(
            "analyze_drift.n_permutations", "Permutation budget", integer=True
        )
    with transport:
        fields.choice(
            "wasserstein",
            "Wasserstein method",
            {"Exact": "exact", "Max-sliced": "sliced", "Sinkhorn": "sinkhorn"},
            selected=method,
            change=lambda value: _patch(
                {
                    "analyze_drift": {
                        "use_max_sliced_wasserstein": value == "sliced",
                        "use_sinkhorn_wasserstein": value == "sinkhorn",
                    }
                }
            ),
        )
    if sliced and sinkhorn:
        st.warning(
            "Both legacy flags are enabled. Max-sliced takes precedence. Choosing a different method updates both flags."
        )
    stop = "analyze_drift.early_stopping"

    def set_stopping(enabled: bool) -> None:
        if enabled and not fields.read(stop):
            _patch(_change(stop, {**_canonical(stop), "enabled": True}))
        else:
            _patch(_change(f"{stop}.enabled", enabled))

    fields.flag(
        f"{stop}.enabled", "Enable permutation early stopping", change=set_stopping
    )
    st.caption(
        "The early-stop threshold and the report cutoff (0.05) serve different purposes."
    )
    if fields.read("data.base.n_features") != 1 and method in {"sliced", "sinkhorn"}:
        option = "Projection count" if method == "sliced" else "Sinkhorn regularization"
        st.caption(f"{option} is available in Expert for this multivariate method.")


def _seeds(fields: GuidedFields) -> None:
    st.caption(
        "Clear any seed to leave that operation unseeded. Bootstrap and partitioning share a seed."
    )
    for seeds in (
        (
            ("data.base.seed", "Bootstrap and partition seed"),
            ("data.pred.seed", "Incoming batch seed"),
        ),
        (("train.seed", "Training seed"), ("evaluate.seed", "Calibration seed")),
        (("analyze_drift.seed", "Drift seed"),),
    ):
        for column, (path, label) in zip(st.columns(2), seeds, strict=False):
            with column:
                if (
                    path != "evaluate.seed"
                    or fields.read("evaluate.calibration_method") != "isotonic"
                ):
                    fields.number(path, label, integer=True)


def render_advanced(fields: GuidedFields, state: DashboardState) -> None:
    """Render only the selected Advanced group."""
    with st.container(width=385):
        group = st.selectbox(
            "Advanced group",
            GROUPS,
            key="_configure_advanced_group",
            index=GROUPS.index(state.advanced_group)
            if state.advanced_group in GROUPS
            else 0,
        )
    state.advanced_group = group
    panels = (
        _environment,
        _partitions,
        _graph,
        _training,
        _calibration,
        _drift,
        _seeds,
    )
    panels[GROUPS.index(group)](fields)


def expert_guidance(configuration: dict) -> None:
    """Identify the specialist parameters applicable to the draft."""
    model = _read(configuration, "default_model")
    scaler = _read(configuration, "data.scaler_type")
    method = _read(configuration, "evaluate.calibration_method")
    topics = []
    if model in ("gcn", "gat"):
        topics.append(f"`models.{model}` controls the selected model architecture.")
    if isinstance(scaler, str) and scaler in _SCALERS:
        topics.append(f"`data.scaler_params` holds {scaler} constructor options.")
    if method in ("temperature", "platt", "beta", "dirichlet"):
        options = {
            "temperature": "`initial_temperature` sets the starting temperature",
            "platt": "`eps` clamps probabilities for the fit",
            "beta": "`eps` clamps probabilities for the fit",
            "dirichlet": "`reg_mu` sets the bias penalty",
        }
        topics.append(
            f"Under `evaluate`, {options[method]}. The block also holds the parametric optimizer and stopping settings."
        )
    if _read(configuration, "analyze_drift.early_stopping.enabled") is True:
        topics.append(
            "`analyze_drift.early_stopping` holds its minimum permutations and threshold."
        )
    if _read(configuration, "data.base.n_features") != 1:
        if _read(configuration, "analyze_drift.use_max_sliced_wasserstein") is True:
            topics.append(
                "`analyze_drift.n_projections` sets the max-sliced projection count."
            )
        elif _read(configuration, "analyze_drift.use_sinkhorn_wasserstein") is True:
            topics.append(
                "`analyze_drift.sinkhorn_wasserstein_reg` configures regularization."
            )
    if topics:
        with st.expander("Expert settings for the selected methods"):
            for topic in topics:
                st.markdown(f"- {topic}")
