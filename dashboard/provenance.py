"""Verify result correspondence within one explicitly refreshed local workspace."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from graphlib import TopologicalSorter
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, TypeVar

from dashboard.artifacts import Workspace
from dashboard.dvc_evidence import (
    LocalHashes,
    WorkspaceStamp,
    dvc_mapping,
    local_status,
    plain_path,
    same_value,
)
from src.configuration import validate_configuration
from src.configuration_files import configuration_operation
from src.configuration_yaml import parse_configuration_yaml

if TYPE_CHECKING:
    from dashboard.scientific_summary import BasePopulation, CycleSummary

T = TypeVar("T")


class Availability(StrEnum):
    """Describe local completeness independently of correspondence."""

    AVAILABLE = "Available"
    PARTIAL = "Partial"
    MISSING = "Missing"
    NOT_APPLICABLE = "Not applicable"


class Correspondence(StrEnum):
    """Describe what local evidence supports for an applied scope."""

    MATCHING = "Verified matching"
    UNKNOWN = "Unverified correspondence"
    CONFLICT = "Outdated or conflicting"


@dataclass(frozen=True)
class AppliedConfiguration:
    """Small applied summary with a fingerprint of the exact YAML bytes."""

    fingerprint: str
    model: str
    requested_device: str
    cycles: int
    folds: int
    requested_bootstrap: int
    requested_batch: int
    calibration: str

    @classmethod
    def read(cls, payload: bytes) -> tuple["AppliedConfiguration", dict]:
        """Strictly validate the scientific interpretation of the active file."""
        settings = parse_configuration_yaml(payload, source="applied params.yaml")
        validation = validate_configuration(settings)
        if not validation.is_valid:
            first = next(
                issue for issue in validation.issues if issue.severity == "error"
            )
            raise ValueError(
                f"Applied configuration is invalid: {first.path} — {first.message}"
            )
        base, batch = settings["data"]["base"], settings["data"]["pred"]
        return cls(
            sha256(payload).hexdigest(),
            settings["default_model"],
            settings["device"],
            settings["cycles"],
            base["n_splits"],
            base["n_samples"],
            batch["n_samples"],
            settings["evaluate"]["calibration_method"],
        ), settings


@dataclass(frozen=True)
class Evidence:
    """Record availability, applied correspondence and independent output integrity."""

    availability: Availability
    correspondence: Correspondence = Correspondence.UNKNOWN
    locked_outputs_match: bool | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CycleEvidence:
    """Keep cohort metrics beside their scope and required producer evidence."""

    cycle: int
    evidence: Evidence
    local_summary: "CycleSummary"

    @property
    def matching_summary(self) -> "CycleSummary | None":
        """Expose the cycle to current comparisons only when complete and matching."""
        return (
            self.local_summary
            if self.evidence.correspondence == Correspondence.MATCHING
            else None
        )


@dataclass(frozen=True)
class Verification:
    """Point-in-time verification of the applied pipeline, including gaps."""

    root: Path
    checked_at: datetime
    elapsed: float
    applied: AppliedConfiguration | None
    discovered_cycles: tuple[int, ...]
    bootstrap: "BasePopulation | None" = None
    cycles: tuple[CycleEvidence, ...] = ()
    stages: tuple[tuple[str, Evidence], ...] = ()
    aggregate: Evidence = Evidence(Availability.NOT_APPLICABLE)
    reasons: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    stamp: WorkspaceStamp | None = None

    @property
    def declared_cycles(self) -> range:
        """Keep the active learning range even when all outputs are missing."""
        return range(1, self.applied.cycles + 1 if self.applied else 1)

    @property
    def latest_complete_matching_cycle(self) -> int | None:
        """Return the latest complete matching learning cycle, leaving gaps."""
        return next(
            (
                cycle.cycle
                for cycle in reversed(self.cycles)
                if cycle.matching_summary is not None
            ),
            None,
        )

    def still_observed(self, workspace: Workspace) -> bool:
        """Invalidate displayed evidence after normal workspace changes."""
        if self.stamp is None or workspace.root.resolve() != self.root:
            return False
        try:
            return self.stamp == WorkspaceStamp.capture(self.root, self.paths)
        except (OSError, ValueError, RuntimeError):
            return False


def _cycle_stages(cycle: int) -> tuple[str, ...]:
    return (
        f"train_cycle_{cycle}",
        f"evaluate_cycle_{cycle}",
        f"generate_pred_data_cycle_{cycle}",
        f"predict_cycle_{cycle}",
        f"analyze_drift_cycle_{cycle}",
        f"prepare_next_cycle_data_{cycle}",
    )


def _members(settings: dict, cycle: int) -> dict[str, tuple[str, ...]]:
    trained, evaluation, _, prediction, drift, _ = _cycle_stages(cycle)
    method = settings["evaluate"]["calibration_method"]
    optional = (
        ("optimum_params.txt", f"calibration_{method}_loss_and_gradient.png")
        if method != "isotonic" and settings["evaluate"]["verbose"]
        else ()
    )
    return {
        trained: (
            "model.pt",
            "preprocessing.json",
            "graph_binding.json",
            *(
                f"training_plots_dataset_{fold}.png"
                for fold in range(1, settings["data"]["base"]["n_splits"] + 1)
            ),
        ),
        evaluation: (
            "calibrator.pt",
            "calibration_binding.json",
            "metrics.json",
            "reliability_diagram.png",
            "confusion_matrix.png",
            *optional,
        ),
        prediction: (
            "predictions.csv",
            "metrics.json",
            "histograms.png",
            "confussion_matrix_.png",
        ),
        drift: (
            "drift_metadata.json",
            "drift_results.txt",
            "distance_distributions.png",
            "marginal_distributions.png",
            "mutual_information.png",
            "pca_kpca_grid.png",
        ),
    }


def _entry_map(record: dict, key: str) -> dict:
    files = record.get(key, [])
    if not isinstance(files, list):
        raise ValueError(f"Invalid lock {key} declaration.")
    entries = {}
    for item in files:
        name = item.get("path") if isinstance(item, dict) else None
        if not isinstance(name, str) or name in entries:
            raise ValueError(f"Invalid or duplicate lock {key} path.")
        entries[name] = item
    return entries


def _tracked(settings: dict, paths: list[str]) -> dict:
    values = {}
    for path in paths:
        item = settings
        for key in path.split("."):
            item = item[key]
        values[path] = item
    return values


def _availability(
    root: Path, files: tuple[str, ...], members: tuple[str, ...]
) -> Availability:
    paths = [plain_path(root, file) for file in files]
    if not any(path.exists() for path in paths):
        return Availability.MISSING
    if members:
        complete = paths[0].is_dir() and all(
            (paths[0] / member).is_file() for member in members
        )
    else:
        complete = all(path.is_file() for path in paths)
    return Availability.AVAILABLE if complete else Availability.PARTIAL


def _match_files(hashes: LocalHashes, entries: dict) -> tuple[str, ...]:
    def matches(entry: dict) -> bool:
        try:
            return hashes.matches(entry)
        except FileNotFoundError:
            return False

    return tuple(name for name, entry in entries.items() if not matches(entry))


def _dvc_stage(
    name: str,
    spec: dict,
    availability: Availability,
    record: object,
    hashes: LocalHashes,
    settings: dict,
    dvc_params: dict,
    stage_status: object,
    topology: bool | None,
    status_ok: bool,
) -> Evidence:
    if record is None:
        return Evidence(availability, reasons=(f"{name}: no DVC lock record.",))
    if not isinstance(record, dict):
        return Evidence(availability, reasons=(f"{name}: malformed DVC lock record.",))
    try:
        outputs, deps = (_entry_map(record, key) for key in ("outs", "deps"))
        output_mismatch = _match_files(
            hashes, {key: outputs[key] for key in spec["outs"] if key in outputs}
        )
        locked_outputs_match = (
            outputs.keys() == set(spec["outs"]) and not output_mismatch
        )
        state = Correspondence.MATCHING
        reasons = []
        if (
            topology is False
            or record.get("cmd") != spec["cmd"]
            or outputs.keys() != set(spec["outs"])
            or deps.keys() != set(spec["deps"])
        ):
            reasons.append(
                f"{name}: generated topology, lock command or declared paths differ."
            )
            state = Correspondence.CONFLICT
        parameters = spec.get("params", [])
        expected = {"params.yaml": _tracked(settings, parameters)} if parameters else {}
        if not same_value(expected, record.get("params", {})) or not same_value(
            _tracked(settings, parameters), _tracked(dvc_params, parameters)
        ):
            reasons.append(f"{name}: tracked parameter values or types differ.")
            state = Correspondence.CONFLICT
        dep_mismatch = _match_files(
            hashes, {key: deps[key] for key in spec["deps"] if key in deps}
        )
        if dep_mismatch or output_mismatch:
            reasons.extend(
                f"{name}: DVC lock content differs or is missing at {path}."
                for path in (*dep_mismatch, *output_mismatch)
            )
            state = Correspondence.CONFLICT
        if availability != Availability.AVAILABLE:
            reasons.append(f"{name}: required output files are missing.")
            if state == Correspondence.MATCHING:
                state = Correspondence.UNKNOWN
        if not status_ok or stage_status is not None:
            cache_only = (
                isinstance(stage_status, list)
                and bool(stage_status)
                and all(
                    isinstance(change, dict)
                    and change.keys() == {"changed outs"}
                    and isinstance(change["changed outs"], dict)
                    and change["changed outs"]
                    and all(
                        value == "not in cache"
                        for value in change["changed outs"].values()
                    )
                    for change in stage_status
                )
            )
            if not status_ok or not cache_only:
                reasons.append(
                    f"{name}: local DVC status was unavailable or reported outstanding changes."
                )
                if state == Correspondence.MATCHING:
                    state = Correspondence.UNKNOWN
        return Evidence(availability, state, locked_outputs_match, tuple(reasons))
    except (OSError, ValueError, TypeError, KeyError) as error:
        return Evidence(
            availability, reasons=(f"{name}: unreadable lock evidence. {error}",)
        )


def _combine(evidence: tuple[Evidence, ...]) -> Evidence:
    states = {item.correspondence for item in evidence}
    availabilities = {item.availability for item in evidence}
    identity = {item.locked_outputs_match for item in evidence}
    availability = (
        next(iter(availabilities)) if len(availabilities) == 1 else Availability.PARTIAL
    )
    state = (
        Correspondence.CONFLICT
        if Correspondence.CONFLICT in states
        else Correspondence.MATCHING
        if states == {Correspondence.MATCHING}
        else Correspondence.UNKNOWN
    )
    return Evidence(
        availability,
        state,
        False if False in identity else True if identity == {True} else None,
        tuple(dict.fromkeys(reason for item in evidence for reason in item.reasons)),
    )


def _dependencies(pipeline: dict) -> dict[str, tuple[str, ...]]:
    owners = {out: name for name, spec in pipeline.items() for out in spec["outs"]}
    return {
        name: tuple(
            dict.fromkeys(
                owners[parent.as_posix()]
                for dep in spec["deps"]
                for parent in (Path(dep), *Path(dep).parents)
                if parent.as_posix() in owners
            )
        )
        for name, spec in pipeline.items()
    }


def _propagate(stages: dict[str, Evidence], dependencies: dict) -> None:
    for name in TopologicalSorter(dependencies).static_order():
        bad = [
            parent
            for parent in dependencies[name]
            if stages[parent].correspondence != Correspondence.MATCHING
        ]
        if bad:
            stage = stages[name]
            state = _combine(
                (stage, *(stages[parent] for parent in bad))
            ).correspondence
            stages[name] = replace(
                stage,
                correspondence=state,
                reasons=(
                    *stage.reasons,
                    f"{name}: upstream correspondence incomplete for {', '.join(bad)}.",
                ),
            )


def _read_summaries(root: Path, settings: dict, stages: dict[str, Evidence]):
    from dashboard.scientific_summary import (
        CycleSummary,
        ScientificReader,
        classification_metrics,
        complete_metrics,
        evaluation_metrics,
        read_json,
    )

    def issue(
        name: str, message: str, *, conflict: bool = False, partial: bool = False
    ) -> None:
        stage = stages[name]
        stages[name] = replace(
            stage,
            availability=Availability.PARTIAL if partial else stage.availability,
            correspondence=Correspondence.CONFLICT
            if conflict
            else (
                Correspondence.UNKNOWN
                if stage.correspondence == Correspondence.MATCHING
                else stage.correspondence
            ),
            reasons=(*stage.reasons, f"{name}: {message}"),
        )

    def checked(name: str, read: Callable[[], T], *dependencies: object) -> T | None:
        if not stages[name].locked_outputs_match:
            return None
        if any(value is None for value in dependencies):
            issue(name, "Scientific binding prerequisites are unavailable.")
            return None
        try:
            return read()
        except FileNotFoundError as error:
            issue(
                name,
                f"Required scientific field or file missing. {error}",
                partial=True,
            )
        except Exception as error:
            # Trusted but legacy, truncated or otherwise incompatible artifacts fail closed.
            issue(
                name,
                f"Scientific artifact or binding could not be verified. {error}",
                conflict=True,
                partial=True,
            )
        return None

    reader = ScientificReader(root, settings)
    input_base = checked("bootstrap", lambda: reader.base(0))
    bootstrap = input_base.population if input_base else None

    def cycle_summary(cycle: int, input_base):
        trained, evaluation, incoming, prediction, drift, appended = _cycle_stages(
            cycle
        )
        batch = checked(incoming, lambda: reader.batch(cycle, input_base), input_base)
        after = checked(
            appended, lambda: reader.append(cycle, input_base, batch), input_base, batch
        )
        if after is None and stages[appended].locked_outputs_match:
            # Its local partition counts can still support the next cycle's lineage check.
            after = checked(appended, lambda: reader.base(cycle))
        model = checked(trained, lambda: reader.model(cycle, input_base), input_base)
        calibration = checked(
            evaluation,
            lambda: reader.calibration(cycle, input_base, model[1]),
            input_base,
            model,
        )
        drift_summary = checked(
            drift,
            lambda: reader.drift(cycle, input_base, batch, model[1], model[2]),
            input_base,
            batch,
            model,
        )
        eval_metrics = checked(
            evaluation,
            lambda: evaluation_metrics(
                read_json(reader.path(f"evaluation/cycle_{cycle}/metrics.json"))
            ),
        )
        batch_metrics = checked(
            prediction,
            lambda: classification_metrics(
                read_json(reader.path(f"prediction/cycle_{cycle}/metrics.json"))
            ),
        )
        for name, metrics in ((evaluation, eval_metrics), (prediction, batch_metrics)):
            if metrics is not None and not complete_metrics(metrics):
                issue(
                    name,
                    "Published metrics have missing or invalid values.",
                    partial=True,
                )
        summary = CycleSummary(
            input_base.population if input_base else None,
            len(batch.sample_ids) if batch else None,
            after.population if after else None,
            eval_metrics,
            batch_metrics,
            model[0] if model else None,
            calibration,
            drift_summary,
            input_base.raw.feature_names if input_base else (),
            input_base.raw.class_names if input_base else (),
        )
        return summary, after

    summaries = []
    for cycle in range(1, settings["cycles"] + 1):
        summary, input_base = cycle_summary(cycle, input_base)
        summaries.append(summary)
    return bootstrap, summaries


def stage_paths(scope: dict) -> tuple[str, ...]:
    """List declared stage paths and their ancestor ignore files for stamping."""
    files = {path for spec in scope.values() for path in (*spec["deps"], *spec["outs"])}
    ignore_files = {
        (parent / ".dvcignore").as_posix()
        for path in files
        for parent in Path(path).parents
    }
    return tuple(sorted(files | ignore_files))


def producer_scope(pipeline: dict, stage: str) -> dict:
    """Select a declaration and all upstream producers, including directory owners."""
    dependencies = _dependencies(pipeline)
    pending = [stage]
    scope = {}
    while pending:
        name = pending.pop()
        if name not in scope:
            scope[name] = pipeline[name]
            pending.extend(dependencies[name])
    return scope


def check_dvc_stages(
    root: Path,
    settings: dict,
    pipeline: dict,
    scope: dict,
    controls: tuple[bytes | None, ...],
    *,
    snapshots: dict[str, bytes] | None = None,
) -> tuple[dict[str, Evidence], list[str]]:
    """Check fresh local stage content against the applied topology and lock."""
    params_bytes, definition_bytes, lock_bytes, *_ = controls
    problems = []
    topology = None
    if definition_bytes is None:
        problems.append("No generated dvc.yaml is available.")
    else:
        try:
            topology = same_value(
                dvc_mapping(definition_bytes, "dvc.yaml"), {"stages": pipeline}
            )
            if not topology:
                problems.append(
                    "Generated DVC definition differs from the expected applied topology."
                )
        except ValueError as error:
            problems.append(str(error))
    records = {}
    if lock_bytes is None:
        problems.append("No dvc.lock is available.")
    else:
        try:
            lock = dvc_mapping(lock_bytes, "dvc.lock")
            if lock.get("schema") != "2.0" or not isinstance(lock.get("stages"), dict):
                raise ValueError("Unsupported DVC lock schema.")
            records = lock["stages"]
        except ValueError as error:
            problems.append(str(error))
    status = {}
    status_ok = False
    if topology and records:
        try:
            status = local_status(root)
            status_ok = not (status.keys() - pipeline.keys())
            if not status_ok:
                problems.append("Local DVC status included unidentified stages.")
        except Exception as error:
            problems.append(str(error))
    hashes = LocalHashes(root, snapshots=snapshots)
    dvc_params = dvc_mapping(params_bytes, "params.yaml")
    members = {
        key: value
        for cycle in range(1, settings["cycles"] + 1)
        for key, value in _members(settings, cycle).items()
    }
    stages = {
        name: _dvc_stage(
            name,
            spec,
            _availability(root, tuple(spec["outs"]), members.get(name, ())),
            records.get(name),
            hashes,
            settings,
            dvc_params,
            status.get(name),
            topology,
            status_ok,
        )
        for name, spec in scope.items()
    }
    return stages, problems


def verify_workspace(workspace: Workspace) -> Verification:
    """Check declared local evidence under the cooperative lock on explicit refresh."""
    from scripts.setup_dvc import pipeline_definition

    started = perf_counter()
    root = workspace.root.resolve()
    applied = None
    observed = ()
    try:
        with configuration_operation(root / "params.yaml"):
            observed = tuple(
                sorted(set(Workspace(root).result_cycles(include_raw=True)) - {0})
            )
            controls = WorkspaceStamp.capture(root, ()).controls
            params_bytes = controls[0]
            if params_bytes is None:
                raise ValueError("An applied configuration is required.")
            applied, settings = AppliedConfiguration.read(params_bytes)
            pipeline = pipeline_definition(settings)["stages"]
            paths = stage_paths(pipeline)
            stamp = WorkspaceStamp.capture(root, paths)
            if stamp.controls != controls:
                raise ValueError(
                    "Workspace controls changed during verification. Refresh again."
                )
            stages, problems = check_dvc_stages(
                root, settings, pipeline, pipeline, controls
            )
            bootstrap, summaries = _read_summaries(root, settings, stages)
            _propagate(stages, _dependencies(pipeline))
            cycles = tuple(
                CycleEvidence(
                    cycle,
                    _combine(
                        tuple(
                            stages[name]
                            for name in (
                                "bootstrap"
                                if cycle == 1
                                else f"prepare_next_cycle_data_{cycle - 1}",
                                *_cycle_stages(cycle),
                            )
                        )
                    ),
                    summary,
                )
                for cycle, summary in enumerate(summaries, start=1)
            )
            if (
                stamp != WorkspaceStamp.capture(root, paths)
                or stamp.controls != WorkspaceStamp.capture(root, ()).controls
            ):
                raise ValueError(
                    "Workspace files changed during verification. Refresh again."
                )
            return Verification(
                root,
                datetime.now(UTC),
                perf_counter() - started,
                applied,
                observed,
                bootstrap,
                cycles,
                tuple(stages.items()),
                stages.get(
                    "analyze_cycles_performance", Evidence(Availability.NOT_APPLICABLE)
                ),
                tuple(problems),
                paths,
                stamp,
            )
    except Exception as error:
        return Verification(
            root,
            datetime.now(UTC),
            perf_counter() - started,
            applied,
            observed,
            reasons=(str(error),),
        )
