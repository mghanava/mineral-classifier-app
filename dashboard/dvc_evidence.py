"""Inspect local DVC declarations and hash workspace contents using DVC's codec."""

import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

CONTROL_PATHS = (
    "params.yaml",
    "dvc.yaml",
    "dvc.lock",
    ".dvcignore",
    ".dvc/config",
    ".dvc/config.local",
)


def plain_path(root: Path, name: str) -> Path:
    """Allow only confined local regular files and directories, without symlinks."""
    relative = Path(name)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or path.resolve() != path:
        raise ValueError(f"Unsupported local path: {name}")
    return path


def _signature(path: Path) -> tuple[int, ...] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError(f"Unsupported workspace file: {path}")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True)
class WorkspaceStamp:
    """Cheap invalidation guard over controls and declared result/source paths."""

    controls: tuple[bytes | None, ...]
    files: tuple[tuple[str, tuple[int, ...] | None], ...]

    @classmethod
    def capture(cls, root: Path, paths: tuple[str, ...]) -> "WorkspaceStamp":
        """Capture control bytes and recursively enumerate declared directories."""

        def entries(path: Path):
            sig = _signature(path)
            yield path.relative_to(root).as_posix(), sig
            if sig and stat.S_ISDIR(sig[2]):
                for child in sorted(path.iterdir()):
                    yield from entries(child)

        controls = []
        for name in CONTROL_PATHS:
            path = plain_path(root, name)
            sig = _signature(path)
            if sig and not stat.S_ISREG(sig[2]):
                raise ValueError(f"Expected a regular DVC control file: {name}")
            controls.append(path.read_bytes() if sig else None)
        files = {}
        for name in paths:
            files.update(entries(plain_path(root, name)))
        return cls(tuple(controls), tuple(sorted(files.items())))


def dvc_mapping(payload: bytes, name: str) -> dict:
    """Decode a DVC document using its YAML 1.2 loader."""
    from dvc.utils.serialize import loads_yaml
    from ruamel.yaml import YAMLError

    try:
        value = loads_yaml(payload.decode("utf-8"))
    except (ValueError, YAMLError, RecursionError) as error:
        raise ValueError(f"Cannot decode {name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {name}.")
    return value


def same_value(left: object, right: object) -> bool:
    """Compare nested scientific settings without equating bool, int and float."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same_value(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def local_status(root: Path) -> dict:
    """Get the supported CLI's local status for the generated pipeline only."""
    with TemporaryDirectory(prefix="mineral-dvc-status-") as site_cache:
        process = subprocess.run(
            [sys.executable, "-m", "dvc", "status", "--json", "dvc.yaml"],
            cwd=root,
            env={
                **os.environ,
                "DVC_SITE_CACHE_DIR": site_cache,
                "DVC_NO_ANALYTICS": "1",
            },
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    if process.returncode:
        raise ValueError(
            f"Local DVC status failed (exit {process.returncode}): {process.stderr.strip()[-500:]}"
        )
    try:
        status = json.loads(process.stdout)
    except ValueError as error:
        raise ValueError("Local DVC status did not return JSON.") from error
    if not isinstance(status, dict):
        raise ValueError("Local DVC status did not return a stage mapping.")
    return status


class LocalHashes:
    """Compute DVC workspace hashes with no persistent hash state or ODB writes."""

    def __init__(self, root: Path) -> None:
        """Build against fresh local metadata using DVC's directory manifest format."""
        from dvc.ignore import DvcIgnoreFilter
        from dvc_objects.fs import LocalFileSystem

        self.root = root
        self.fs = LocalFileSystem()
        self.ignore = DvcIgnoreFilter(self.fs, str(root))
        self.cache: dict[str, tuple[str, int | None, int | None]] = {}

    def matches(self, entry: dict) -> bool:
        """Check regular file or directory content and the lock's size metadata."""
        from dvc_data.hashfile.build import _build_tree
        from dvc_data.hashfile.hash import hash_file

        name = entry["path"]
        checksum = entry.get("md5")
        if (
            entry.get("hash") != "md5"
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{32}(?:\.dir)?", checksum)
        ):
            raise ValueError(f"Missing or unsupported DVC digest for {name}.")
        path = plain_path(self.root, name)
        if name not in self.cache:
            info = self.fs.info(str(path))
            if info["type"] == "directory":
                # The locked DVC tree builder delegates file hashing and the
                # canonical .dir serialization to dvc-data without a disk ODB.
                meta, tree = _build_tree(
                    str(path),
                    self.fs,
                    info,
                    "md5",
                    dry_run=True,
                    ignore=self.ignore,
                    checksum_jobs=2,
                )
                digest = tree.hash_info
                tree.fs.rm(tree.path)
            else:
                meta, digest = hash_file(str(path), self.fs, "md5", info=info)
            self.cache[name] = (digest.value, meta.size, meta.nfiles)
        actual, size, nfiles = self.cache[name]
        return (
            actual == checksum
            and same_value(entry.get("size"), size)
            and (
                not checksum.endswith(".dir") or same_value(entry.get("nfiles"), nfiles)
            )
        )
