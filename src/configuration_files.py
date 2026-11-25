"""Preview and explicitly persist a resolved scientific configuration."""

import difflib
import fcntl
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.configuration import ValidationResult, validate_configuration
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml
from src.presets import AppliedPreset

_TOPOLOGY_PATHS = ("cycles", "default_model", "data.base.n_splits")
_MISSING = object()


@dataclass(frozen=True)
class ConfigurationSnapshot:
    """Capture the target path and exact bytes for optimistic concurrency checks."""

    path: Path
    contents: bytes | None


@dataclass(frozen=True)
class ConfigurationPreview:
    """Hold the resolved YAML, findings and comparison against the active file."""

    yaml: str
    validation: ValidationResult
    active: ConfigurationSnapshot | None
    diff: str
    regeneration_paths: tuple[str, ...]
    comparison_warning: str | None = None


def snapshot_configuration(path: Path) -> ConfigurationSnapshot:
    """Read an existing regular file, or record that the file is absent."""
    path = path.parent.resolve() / path.name
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"Use a regular configuration file at {path}.")
        contents = path.read_bytes()
    except FileNotFoundError:
        contents = None
    return ConfigurationSnapshot(path, contents)


def _get_path(configuration: Mapping, path: str) -> object:
    value = configuration
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def pipeline_regeneration_paths(before: Mapping, after: Mapping) -> tuple[str, ...]:
    """Report changes that affect generated commands, stages or outputs."""
    changes = []
    for path in _TOPOLOGY_PATHS:
        old, new = _get_path(before, path), _get_path(after, path)
        if type(old) is not type(new) or old != new:
            changes.append(path)
    return tuple(changes)


def preview_configuration(
    resolved: AppliedPreset, active: ConfigurationSnapshot | None = None
) -> ConfigurationPreview:
    """Render complete YAML and a content diff without updating any file."""
    serialized = dump_configuration_yaml(resolved.configuration)
    if active is None:
        return ConfigurationPreview(serialized, resolved.validation, None, "", ())
    before, old_yaml, warning = {}, "", None
    if active.contents is not None:
        try:
            before = parse_configuration_yaml(active.contents, source=str(active.path))
            old_yaml = dump_configuration_yaml(before, sort_keys=True)
        except ValueError as error:
            warning = f"{error} Comparing with the active file as text."
            old_yaml = active.contents.decode("utf-8", errors="replace")
    new_yaml = dump_configuration_yaml(resolved.configuration, sort_keys=True)
    diff = "\n".join(
        difflib.unified_diff(
            old_yaml.splitlines(),
            new_yaml.splitlines(),
            fromfile=str(active.path),
            tofile="resolved configuration",
            lineterm="",
        )
    )
    return ConfigurationPreview(
        serialized,
        resolved.validation,
        active,
        diff,
        pipeline_regeneration_paths(before, resolved.configuration),
        warning,
    )


@contextmanager
def configuration_operation(path: Path) -> Iterator[None]:
    """Reserve the configuration directory for a write or dashboard run."""
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "A pipeline run is active or a configuration write is in progress. "
                "Try again after it finishes."
            ) from error
        yield
    finally:
        os.close(directory)


def persist_configuration(
    preview: ConfigurationPreview,
    target: ConfigurationSnapshot,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish validated YAML if the target still matches its snapshot."""
    if not preview.validation.is_valid:
        raise ValueError("Fix validation errors before writing a configuration.")
    # Check the exact serialized document that will be handed to stage consumers.
    document = parse_configuration_yaml(preview.yaml, source="resolved configuration")
    if not validate_configuration(document).is_valid:
        raise ValueError("The serialized configuration failed validation.")
    if preview.active is not None and preview.active.path == target.path:
        target = preview.active

    path = target.path
    with configuration_operation(path):
        _publish_configuration(preview.yaml, target, overwrite=overwrite)


def _publish_configuration(
    yaml: str, target: ConfigurationSnapshot, *, overwrite: bool
) -> None:
    path = target.path
    temporary = None
    try:
        current = snapshot_configuration(path)
        if current.contents != target.contents:
            raise ValueError(
                f"{path} changed since preview. Preview again before writing."
            )
        if current.contents is not None and not overwrite:
            raise ValueError(
                f"{path} already exists. Explicitly authorize replacement."
            )
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            if current.contents is not None:
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            stream.write(yaml.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if snapshot_configuration(path).contents != current.contents:
            raise ValueError(
                f"{path} changed since preview. Preview again before writing."
            )
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic create-if-absent even if another tool ignores the lock.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
