"""Resolve and verify the production independent spatial graph contract."""

import hashlib
import json
from collections.abc import Mapping
from numbers import Real

import torch
from torch_geometric.data import Data

GRAPH_PROTOCOL_VERSION = 1


def json_digest(value: dict) -> str:
    """Hash metadata with canonical JSON ordering."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_affinity_scale(method: str, scale: object) -> None:
    """Check the scale contract for the configured affinity method."""
    if method == "unit":
        if scale is not None:
            raise ValueError("Unit edge affinity requires scale: null.")
    elif method == "inverse_quadratic":
        if (
            not isinstance(scale, Real)
            or isinstance(scale, bool)
            or not 0 < scale < float("inf")
        ):
            raise ValueError(
                "Inverse-quadratic edge affinity requires a finite positive scale."
            )
    else:
        raise ValueError("Use edge_affinity.method inverse_quadratic or unit.")


def resolve_connectivity(base: Mapping[str, object]) -> dict[str, object]:
    """Return the sole active connectivity rule without building a graph."""
    keys = ("k_nearest", "connection_radius", "distance_percentile")
    connectivity = {key: base[key] for key in keys if base.get(key) is not None}
    if len(connectivity) != 1:
        raise ValueError(
            "Set exactly one of k_nearest, connection_radius, distance_percentile; set the others to null."
        )
    return connectivity


def resolve_graph_protocol(params: dict) -> dict:
    """Resolve the versioned, model-independent graph settings from YAML."""
    data = params["data"]
    if {"make_edge_weight", "make_edge_weight_method"} & data.keys():
        raise ValueError(
            "Legacy edge weighting is incompatible. Remove data.make_edge_weight and "
            "data.make_edge_weight_method, then set data.edge_affinity to "
            "{method: inverse_quadratic, scale: 460.0} or {method: unit, scale: null}."
        )
    affinity = data.get("edge_affinity")
    if not isinstance(affinity, dict) or set(affinity) != {"method", "scale"}:
        raise ValueError("Set data.edge_affinity with explicit method and scale keys.")
    method, scale = affinity["method"], affinity["scale"]
    validate_affinity_scale(method, scale)
    connectivity = resolve_connectivity(data["base"])
    loops = params.get("add_self_loops")
    if type(loops) is not bool:
        raise ValueError("Set add_self_loops to true or false.")
    return {
        "version": GRAPH_PROTOCOL_VERSION,
        "coordinates": "canonical_float64_xyz",
        "metric": "euclidean",
        "connectivity": connectivity,
        "symmetrization": "undirected_union_unique_coo",
        "ties": "knn_stable_id_percentile_inclusive_linear",
        "affinity": {
            "method": method,
            "scale": float(scale) if scale is not None else None,
        },
        "self_loops": {
            "owner": "graph_builder",
            "enabled": loops,
            "stored_affinity": 1.0,
            "improved_gcn_affinity": 2.0,
        },
    }


def validate_model_graph_settings(model_name: str, model: Mapping[str, object]) -> None:
    """Check model-specific requirements of the independent graph protocol."""
    if "add_self_loops" in model:
        raise ValueError(
            "Remove models.*.add_self_loops. Use the top-level add_self_loops setting."
        )
    if model_name == "gcn" and model.get("normalize") is not True:
        raise ValueError("Independent GCN graphs require models.gcn.normalize: true.")


def model_graph_settings(protocol: dict, model_name: str, params: dict) -> dict:
    """Resolve model-specific settings without changing graph topology."""
    if model_name not in ("gcn", "gat"):
        raise ValueError(f"Unsupported graph model {model_name!r}.")
    model = params["models"][model_name]
    validate_model_graph_settings(model_name, model)
    loops = protocol["self_loops"]
    improved = model_name == "gcn" and model.get("improved", False)
    return {
        "protocol": protocol,
        "model": model_name,
        "model_parameters": {**model, "add_self_loops": False},
        "effective_loop_affinity": (
            loops["improved_gcn_affinity"] if improved else loops["stored_affinity"]
        )
        if loops["enabled"]
        else None,
    }


def graph_digest(graph: Data) -> str:
    """Bind graph policy, row provenance and exact stored topology and weights."""
    digest = hashlib.sha256()
    digest.update(
        json_digest(
            {
                "protocol": graph.graph_protocol,
                "dataset": graph.dataset_sha256,
                "split": getattr(graph, "split_sha256", None),
                "role": graph.split_type,
                "source_parent": getattr(graph, "source_parent_sha256", None),
                "sample_ids": graph.sample_ids,
                "effective_k": getattr(graph, "effective_k", None),
            }
        ).encode()
    )
    for key in (
        "coordinates",
        "edge_index",
        "edge_weight",
        "edge_attr",
        "edge_distance",
    ):
        values = graph[key]
        if not isinstance(values, torch.Tensor) or values.device.type != "cpu":
            raise ValueError(f"Graph field {key} must be a stored CPU tensor.")
        digest.update(str((values.dtype, tuple(values.shape))).encode())
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def verify_graph(graph: Data, expected: dict, role: str) -> None:
    """Reject stale or legacy graph payloads before model consumption."""
    if (
        getattr(graph, "graph_protocol", None) != expected
        or getattr(graph, "split_type", None) != role
    ):
        raise ValueError(
            "Incompatible graph protocol or role. Regenerate independent graph artifacts."
        )
    try:
        valid = graph.graph_sha256 == graph_digest(graph)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid graph provenance. Regenerate independent graph artifacts."
        ) from exc
    if not valid:
        raise ValueError(
            "Graph payload digest mismatch. Regenerate independent graph artifacts."
        )
