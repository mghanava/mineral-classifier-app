"""Test read-only previews, stale checks and atomic scientific YAML export."""

from dataclasses import replace

import pytest

from src import configuration_files
from src.configuration_files import (
    persist_configuration,
    pipeline_regeneration_paths,
    preview_configuration,
    snapshot_configuration,
)
from src.configuration_yaml import parse_configuration_yaml
from src.presets import apply_preset, load_preset, resolve_new_configuration
from src.utilities.general_utils import load_params


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("cycles: [1", "Cannot parse fixture"),
        ("data:\n  base:\n    seed: 1\n    seed: null", "Duplicate mapping key 'seed'"),
        ("[]", "one YAML mapping"),
        ("cycles: 1\n---\ncycles: 2", "Cannot parse fixture"),
        ("data: &block {base: *block}", "anchors and aliases"),
        ("data: {<<: {base: {}}}", "Unsupported YAML tag"),
        ("data: !!python/object/apply:os.getcwd []", "Unsupported YAML tag"),
        ("data: {5: 1}", "string mapping key"),
        ("data: 2020-01-01", "Unsupported YAML tag"),
        ("data: !!bool bogus", "Cannot parse fixture"),
    ],
)
def test_scientific_yaml_rejects_ambiguous_documents(contents, message):
    """Reject unsafe tags, repeated keys, non-mappings and invalid syntax."""
    with pytest.raises(ValueError, match=message):
        parse_configuration_yaml(contents, source="fixture")


@pytest.fixture
def active_file(tmp_path):
    """Provide an isolated active configuration with formatting to preserve on failure."""
    path = tmp_path / "params.yaml"
    baseline = preview_configuration(apply_preset(load_preset("research")))
    path.write_text("# Active experiment\n" + baseline.yaml, encoding="utf-8")
    return path


def test_preview_and_apply_are_separate_and_allow_warnings(active_file):
    """Compare against active settings and round trip with the stage's loader."""
    before = active_file.read_bytes()
    snapshot = snapshot_configuration(active_file)
    resolved = resolve_new_configuration(
        load_preset("quick_demo"),
        {
            "cycles": 0,
            "default_model": "gat",
            "data": {
                "base": {
                    "n_splits": 4,
                    "seed": None,
                    "test_size": 100,
                    "calib_size": 10,
                    "class_names": [],
                },
                "scaler_params": {},
            },
            "models": {"gat": {"batch_norm": False}},
        },
    )
    preview = preview_configuration(resolved, snapshot)
    assert preview.validation.is_valid
    assert any(issue.severity == "warning" for issue in preview.validation.issues)
    assert any(issue.severity == "info" for issue in preview.validation.issues)
    assert preview.regeneration_paths == (
        "cycles",
        "default_model",
        "data.base.n_splits",
    )
    assert "-cycles: 3" in preview.diff and "+cycles: 0" in preview.diff
    assert active_file.read_bytes() == before
    assert parse_configuration_yaml(preview.yaml) == resolved.configuration

    persist_configuration(preview, snapshot, overwrite=True)
    loaded = load_params(active_file)
    assert loaded == resolved.configuration
    assert type(loaded["data"]["base"]["test_size"]) is int
    assert type(loaded["data"]["edge_affinity"]["scale"]) is float
    assert loaded["models"]["gat"]["batch_norm"] is False
    assert "id" not in loaded and "version" not in loaded


def test_diff_ignores_comments_and_pipeline_regeneration_ignores_other_changes(
    active_file,
):
    """Compare scientific values independently of document presentation."""
    baseline = apply_preset(load_preset("research"))
    preview = preview_configuration(baseline, snapshot_configuration(active_file))
    assert not preview.diff
    assert not preview.regeneration_paths
    changed = resolve_new_configuration(load_preset("research"), {"train": {"seed": 0}})
    assert (
        pipeline_regeneration_paths(baseline.configuration, changed.configuration) == ()
    )


def test_invalid_preview_and_failed_serialization_leave_active_file(active_file):
    """Block both invalid input and an invalid serialized document."""
    snapshot = snapshot_configuration(active_file)
    invalid = resolve_new_configuration(load_preset("research"), {"cycles": -1})
    with pytest.raises(ValueError, match="Fix validation errors"):
        persist_configuration(
            preview_configuration(invalid, snapshot), snapshot, overwrite=True
        )
    valid = preview_configuration(apply_preset(load_preset("standard")), snapshot)
    for contents in ("broken: [", "cycles: 0\n"):
        with pytest.raises(ValueError, match="configuration"):
            persist_configuration(
                replace(valid, yaml=contents), snapshot, overwrite=True
            )
    assert active_file.read_bytes() == snapshot.contents


def test_replace_failure_is_atomic_and_cleans_up(active_file, monkeypatch):
    """Leave original bytes in place if the publication step fails."""
    snapshot = snapshot_configuration(active_file)
    preview = preview_configuration(apply_preset(load_preset("standard")), snapshot)

    def failed_replace(source, target):
        assert active_file.read_bytes() == snapshot.contents
        assert source.read_text() == preview.yaml
        assert target == active_file
        raise OSError("fixture replace failed")

    monkeypatch.setattr(configuration_files.os, "replace", failed_replace)
    with pytest.raises(OSError, match="fixture replace failed"):
        persist_configuration(preview, snapshot, overwrite=True)
    assert active_file.read_bytes() == snapshot.contents
    assert list(active_file.parent.iterdir()) == [active_file]


def test_stale_preview_cannot_overwrite_even_with_a_new_snapshot(active_file):
    """Require a fresh comparison before overwriting someone else's edit."""
    snapshot = snapshot_configuration(active_file)
    preview = preview_configuration(apply_preset(load_preset("standard")), snapshot)
    active_file.write_bytes(snapshot.contents + b"# New edit\n")
    newer = snapshot_configuration(active_file)
    with pytest.raises(ValueError, match="changed since preview"):
        persist_configuration(preview, newer, overwrite=True)
    assert active_file.read_bytes() == newer.contents


def test_export_requires_overwrite_authorization(active_file, tmp_path):
    """Create only the specified output and require authorization for replacement."""
    preview = preview_configuration(apply_preset(load_preset("standard")))
    existing = snapshot_configuration(active_file)
    with pytest.raises(ValueError, match="Explicitly authorize"):
        persist_configuration(preview, existing)
    assert active_file.read_bytes() == existing.contents
    target = snapshot_configuration(tmp_path / "experiment.yaml")
    persist_configuration(preview, target)
    assert target.path.read_text() == preview.yaml
