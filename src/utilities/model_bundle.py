"""Publish and verify local model, preprocessing and calibration artifacts."""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import Module

from src.utilities.graph_protocol import json_digest
from src.utilities.preprocessing import (
    FittedPreprocessing,
    load_preprocessing,
    save_preprocessing,
)
from src.utilities.training_schedule import FoldTrainingResult, final_training_plan

GRAPH_BINDING = "graph_binding.json"
CALIBRATION_BINDING = "calibration_binding.json"


def publish_model(
    directory: Path,
    model: Module,
    state: FittedPreprocessing,
    settings: dict,
    training_graph_sha256: str,
    training: dict,
) -> None:
    """Publish the model and preprocessing with their graph binding last."""
    _verify_training(training, state)
    with TemporaryDirectory(dir=directory) as temporary:
        weights = Path(temporary) / "model.pt"
        sidecar = weights.with_name("preprocessing.json")
        torch.save(model.state_dict(), weights)
        save_preprocessing(sidecar, state, weights)
        graph = weights.with_name(GRAPH_BINDING)
        binding = {
            "version": 2,
            "training": training,
            "settings": settings,
            "training_graph_sha256": training_graph_sha256,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "preprocessing_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        }
        graph.write_text(
            json.dumps(binding, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        weights.replace(directory / weights.name)
        sidecar.replace(directory / sidecar.name)
        graph.replace(directory / graph.name)


def _read_artifact(path: Path, recovery: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read required artifact {path}. {recovery}") from exc


def _verify_training(training: dict, state: FittedPreprocessing) -> None:
    try:
        if (
            state.fold_index is not None
            or training["strategy"] != "full_development_pool"
            or training["dataset_sha256"] != state.dataset_sha256
            or training["partition_sha256"] != state.partition_sha256
            or type(training["observation_count"]) is not int
            or training["observation_count"] < 1
        ):
            raise ValueError("Training population does not match preprocessing.")
        folds = [FoldTrainingResult(**fold) for fold in training["cv_folds"]]
        plan = final_training_plan(folds, training["epoch_budget"])
        if json_digest({key: training[key] for key in plan}) != json_digest(plan):
            raise ValueError(
                "Training duration or schedule does not match CV diagnostics."
            )
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid development-pool training provenance. Retrain the final model: {exc}"
        ) from exc


def read_model_bundle(
    directory: Path,
    feature_names: tuple[str, ...],
    class_names: tuple[str, ...],
    protocol: dict,
    *,
    settings: dict | None = None,
) -> tuple[FittedPreprocessing, dict[str, str | int], bytes]:
    """Verify the exact checkpoint and sidecar bytes without loading a model."""
    weights = _read_artifact(
        directory / "model.pt", "Retrain the development-pool model."
    )
    sidecar = _read_artifact(
        directory / "preprocessing.json", "Retrain the development-pool model."
    )
    state = load_preprocessing(
        sidecar, weights, feature_names=feature_names, class_names=class_names
    )
    digests = {
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(sidecar).hexdigest(),
    }
    payload = _read_artifact(
        directory / GRAPH_BINDING, "Retrain using independent graphs."
    )
    try:
        binding = json.loads(payload)
    except ValueError as exc:
        raise ValueError(
            "Malformed model graph binding. Retrain using independent graphs."
        ) from exc
    if (
        not isinstance(binding, dict)
        or binding.keys()
        != {"version", "settings", "training_graph_sha256", "training", *digests}
        or type(binding["version"]) is not int
        or binding["version"] != 2
        or any(binding[key] != value for key, value in digests.items())
    ):
        raise ValueError(
            "Model graph binding is stale or incompatible. Retrain using independent graphs."
        )
    saved = binding["settings"]
    training_graph = binding["training_graph_sha256"]
    if (
        not isinstance(saved, dict)
        or saved.get("protocol") != protocol
        or (settings is not None and saved != settings)
        or not isinstance(training_graph, str)
        or len(training_graph) != 64
        or any(char not in "0123456789abcdef" for char in training_graph)
    ):
        raise ValueError(
            "Model graph protocol does not match the requested graph settings. Use a matching bundle."
        )
    training = binding["training"]
    _verify_training(training, state)
    digests.update(
        training_strategy=training["strategy"],
        training_observation_count=training["observation_count"],
        graph_binding_sha256=hashlib.sha256(payload).hexdigest(),
        graph_protocol_sha256=json_digest(protocol),
        training_graph_sha256=training_graph,
    )
    return state, digests, weights


def load_model_bundle(
    directory: Path,
    model: Module,
    feature_names: tuple[str, ...],
    class_names: tuple[str, ...],
    settings: dict,
) -> tuple[FittedPreprocessing, dict[str, str | int]]:
    """Verify the sidecar against the same checkpoint bytes loaded into the model."""
    state, digests, weights = read_model_bundle(
        directory, feature_names, class_names, settings["protocol"], settings=settings
    )
    model.load_state_dict(
        torch.load(BytesIO(weights), weights_only=True, map_location="cpu")
    )
    return state, digests


def write_calibration_binding(
    directory: Path,
    bundle: dict[str, str | int],
    dataset_sha256: str,
    split_sha256: str,
) -> None:
    """Record the model bundle, exact calibrator bytes and fitted split."""
    calibrator = (directory / "calibrator.pt").read_bytes()
    binding = {
        "version": 1,
        **bundle,
        "dataset_sha256": dataset_sha256,
        "calibration_split_sha256": split_sha256,
        "calibrator_sha256": hashlib.sha256(calibrator).hexdigest(),
    }
    (directory / CALIBRATION_BINDING).write_text(
        json.dumps(binding, indent=2) + "\n", encoding="utf-8"
    )


def verified_calibrator(
    directory: Path, bundle: dict[str, str | int], dataset_sha256: str
) -> BytesIO:
    """Verify a trusted local calibrator before returning its exact byte snapshot."""
    payload = _read_artifact(directory / CALIBRATION_BINDING, "Re-run evaluation.")
    try:
        binding = json.loads(payload)
    except ValueError as exc:
        raise ValueError("Malformed calibration binding. Re-run evaluation.") from exc
    expected_fields = {
        "version",
        *bundle,
        "dataset_sha256",
        "calibration_split_sha256",
        "calibrator_sha256",
    }
    if (
        not isinstance(binding, dict)
        or binding.keys() != expected_fields
        or type(binding.get("version")) is not int
        or binding["version"] != 1
        or binding.get("dataset_sha256") != dataset_sha256
        or any(binding.get(key) != value for key, value in bundle.items())
    ):
        raise ValueError(
            "Calibration binding does not match the model bundle. Re-run evaluation."
        )
    split = binding.get("calibration_split_sha256")
    if (
        not isinstance(split, str)
        or len(split) != 64
        or any(char not in "0123456789abcdef" for char in split)
    ):
        raise ValueError("Invalid calibration split provenance. Re-run evaluation.")
    calibrator = _read_artifact(directory / "calibrator.pt", "Re-run evaluation.")
    if binding.get("calibrator_sha256") != hashlib.sha256(calibrator).hexdigest():
        raise ValueError("Calibrator digest mismatch. Re-run evaluation.")
    return BytesIO(calibrator)
