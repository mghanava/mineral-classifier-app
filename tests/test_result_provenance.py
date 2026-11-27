"""Exercise local evidence with genuine DVC and retained scientific fixtures."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from dashboard import provenance
from dashboard.artifacts import Workspace
from dashboard.dvc_evidence import LocalHashes, dvc_mapping, local_status
from dashboard.provenance import Availability, Correspondence, verify_workspace
from dashboard.scientific_summary import (
    ScientificReader,
    classification_metrics,
    complete_metrics,
    evaluation_metrics,
)
from scripts.setup_dvc import pipeline_definition
from src.configuration_files import configuration_operation
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml

ROOT = Path(__file__).resolve().parents[1]
RETAINED = Path("/tmp/mineral-m25.SkdwUb/a")


@pytest.fixture(scope="module")
def toy_dvc(tmp_path_factory):
    """Create one real DVC lock from a tiny non-scientific stage."""
    root = tmp_path_factory.mktemp("dvc-evidence-toy")
    (root / ".dvc").mkdir()
    (root / ".dvc/config").write_text("[core]\n    no_scm = True\n")
    (root / "params.yaml").write_text("value: 1.0\nmissing: null\nsmall: 1.0e-08\n")
    (root / "producer.py").write_text(
        "from pathlib import Path\n"
        "out = Path('bundle')\n"
        "out.mkdir()\n"
        "(out / 'table.csv').write_text('value\\n1\\n')\n"
        "Path('one.txt').write_text('one')\n"
    )
    (root / "dvc.yaml").write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "toy": {
                        "cmd": "python producer.py",
                        "deps": ["producer.py"],
                        "params": ["value", "missing", "small"],
                        "outs": ["bundle", "one.txt"],
                    }
                }
            }
        )
    )
    process = subprocess.run(
        [sys.executable, "-m", "dvc", "repro"],
        cwd=root,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "DVC_SITE_CACHE_DIR": str(root / "site"),
            "DVC_NO_ANALYTICS": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return root


def _toy_evidence(root):
    spec = dvc_mapping((root / "dvc.yaml").read_bytes(), "dvc.yaml")["stages"]["toy"]
    lock = root / "dvc.lock"
    record = (
        dvc_mapping(lock.read_bytes(), "dvc.lock")["stages"]["toy"]
        if lock.exists()
        else None
    )
    params = (root / "params.yaml").read_bytes()
    status = local_status(root)
    return provenance._dvc_stage(
        "toy",
        spec,
        Availability.AVAILABLE
        if (root / "one.txt").is_file()
        else Availability.PARTIAL,
        record,
        LocalHashes(root),
        parse_configuration_yaml(params),
        dvc_mapping(params, "params.yaml"),
        status.get("toy"),
        True,
        True,
    )


# The toy directory and separate file use different availability schemas. The
# production adapter always declares directory members for scientific bundles.
@pytest.mark.parametrize(
    ("change", "expected", "matches"),
    [
        ("none", Correspondence.MATCHING, True),
        ("no-lock", Correspondence.UNKNOWN, None),
        ("missing-output", Correspondence.CONFLICT, False),
        ("modified-file", Correspondence.CONFLICT, False),
        ("modified-member", Correspondence.CONFLICT, False),
        ("added-member", Correspondence.CONFLICT, False),
        ("source", Correspondence.CONFLICT, True),
        ("parameter-type", Correspondence.CONFLICT, True),
    ],
)
def test_real_dvc_status_and_fresh_hashes(toy_dvc, tmp_path, change, expected, matches):
    """Inspect real status and verify all bytes independently of cache state."""
    root = shutil.copytree(
        toy_dvc, tmp_path / "workspace", ignore=shutil.ignore_patterns("site")
    )
    if change == "no-lock":
        (root / "dvc.lock").unlink()
    elif change == "missing-output":
        (root / "one.txt").unlink()
    elif change == "modified-file":
        (root / "one.txt").write_text("two")
    elif change == "modified-member":
        (root / "bundle/table.csv").write_text("value\n9\n")
    elif change == "added-member":
        (root / "bundle/extra.csv").write_text("value\n2\n")
    elif change == "source":
        (root / "producer.py").write_text("# source changed")
    elif change == "parameter-type":
        (root / "params.yaml").write_text("value: 1\nmissing: null\nsmall: 1.0e-08\n")

    evidence = _toy_evidence(root)
    assert evidence.correspondence == expected
    assert evidence.locked_outputs_match is matches
    assert evidence.availability == (
        Availability.PARTIAL if change == "missing-output" else Availability.AVAILABLE
    )


def _write_settings(root: Path, cycles: int, *, topology_cycles: int | None = None):
    settings = parse_configuration_yaml((ROOT / "params.yaml").read_bytes())
    settings["cycles"] = cycles
    settings["data"]["base"]["test_size"] = 20
    settings["data"]["base"]["calib_size"] = None
    settings["evaluate"]["seed"] = None
    (root / "params.yaml").write_text(dump_configuration_yaml(settings))
    expected = pipeline_definition(
        {**settings, "cycles": cycles if topology_cycles is None else topology_cycles}
    )
    (root / "dvc.yaml").write_text(yaml.safe_dump(expected))


@pytest.mark.parametrize("cycles", [0, 1, 25])
def test_applied_scope_reports_gaps_and_out_of_scope_raw_artifacts(tmp_path, cycles):
    """Do not promote discovered leftover directories or requested counts."""
    _write_settings(tmp_path, cycles)
    old = tmp_path / "results/data/base/cycle_31/base_data.pt"
    old.parent.mkdir(parents=True)
    old.write_text("not a serialized graph")
    result = verify_workspace(Workspace(tmp_path))
    assert result.declared_cycles == range(1, cycles + 1)
    assert result.discovered_cycles == (31,)
    assert len(result.cycles) == cycles
    assert all(
        cycle.evidence.availability == Availability.MISSING
        and cycle.matching_summary is None
        for cycle in result.cycles
    )
    assert result.bootstrap is None and result.latest_complete_matching_cycle is None
    assert result.aggregate.availability == (
        Availability.NOT_APPLICABLE if cycles < 2 else Availability.MISSING
    )


def test_stale_topology_and_busy_workspace(tmp_path):
    """A valid applied file can coexist with a stale definition or another action."""
    _write_settings(tmp_path, 1, topology_cycles=2)
    workspace = Workspace(tmp_path)
    result = verify_workspace(workspace)
    assert any("expected applied topology" in message for message in result.reasons)
    assert result.latest_complete_matching_cycle is None
    with configuration_operation(tmp_path / "params.yaml"):
        busy = verify_workspace(workspace)
        assert busy.stamp is None
        assert any("pipeline run is active" in message for message in busy.reasons)


def test_metrics_reject_nonfinite_wrong_types_and_missing_fields():
    """Expose absent fields and keep batch metrics separate from calibration."""
    raw = {"acc": float("inf"), "f1": 0.25, "mcc": False, "ece": 0.03}
    batch = classification_metrics(raw)
    assert batch.macro_f05 == 0.25
    assert batch.accuracy is None and batch.mcc is None
    assert not hasattr(batch, "ece")
    final_test = evaluation_metrics({"calibrated": raw})
    assert final_test.after_calibration.ece == 0.03
    assert final_test.before_calibration.classification.accuracy is None
    assert not complete_metrics(final_test)


@pytest.fixture
def retained_workspace(tmp_path):
    """Copy historical scientific outputs and source without sharing a cache."""
    retained = Path(
        os.environ.get("MINERAL_HISTORICAL_QUICK_DEMO_WORKSPACE", RETAINED)
    ).expanduser()
    if not (retained / "dvc.lock").is_file():
        pytest.skip(
            f"Retained local M2.5 verification workspace unavailable at {retained}."
        )
    for name in ("params.yaml", "dvc.yaml", "dvc.lock"):
        shutil.copy2(retained / name, tmp_path / name)
    for name in ("src", "results"):
        shutil.copytree(
            retained / name,
            tmp_path / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    (tmp_path / ".dvc").mkdir()
    shutil.copy2(retained / ".dvc/config", tmp_path / ".dvc/config")
    return Workspace(tmp_path)


@pytest.mark.integration
def test_historical_integrity_is_independent_of_current_sources(retained_workspace):
    """Verify historical evidence before checking identical outputs under the newer sources."""
    workspace = retained_workspace
    old = verify_workspace(workspace)
    assert old.latest_complete_matching_cycle == 2
    assert old.bootstrap.total == 440
    cycle = old.cycles[0].matching_summary
    assert (cycle.input_base.total, cycle.incoming_batch, cycle.post_append.total) == (
        440,
        100,
        540,
    )
    assert (
        cycle.input_base.development,
        cycle.input_base.calibration,
        cycle.input_base.final_test,
    ) == (308, 40, 92)
    assert cycle.model.effective_device == "cpu"
    assert cycle.calibration.method == "dirichlet"
    assert complete_metrics(cycle.evaluation) and complete_metrics(cycle.prediction)
    assert old.still_observed(workspace)

    for name in ("train_utils.py", "eval_utils.py", "cycles_performance_utils.py"):
        relative = Path("src/utilities") / name
        shutil.copy2(ROOT / relative, workspace.root / relative)
    assert not old.still_observed(workspace)
    changed = verify_workspace(workspace)
    assert changed.latest_complete_matching_cycle is None
    assert all(
        cycle.evidence.correspondence == Correspondence.CONFLICT
        for cycle in changed.cycles
    )
    assert all(cycle.evidence.locked_outputs_match for cycle in changed.cycles)
    assert (
        dict(changed.stages)["predict_cycle_1"].correspondence
        == Correspondence.CONFLICT
    )
    assert any(
        "src/utilities/train_utils.py" in reason
        for reason in dict(changed.stages)["train_cycle_1"].reasons
    )


@pytest.mark.integration
def test_mixed_cycle_and_invalid_bound_data(retained_workspace):
    """Keep a verified earlier cycle while rejecting replaced later artifacts and bindings."""
    workspace = retained_workspace
    old = verify_workspace(workspace)
    path = workspace.results / "prediction/cycle_2/metrics.json"
    shutil.copy2(workspace.results / "prediction/cycle_1/metrics.json", path)
    assert not old.still_observed(workspace)
    mixed = verify_workspace(workspace)
    assert mixed.latest_complete_matching_cycle == 1
    assert mixed.cycles[1].matching_summary is None
    assert not dict(mixed.stages)["predict_cycle_2"].locked_outputs_match

    settings = parse_configuration_yaml((workspace.root / "params.yaml").read_bytes())
    reader = ScientificReader(workspace.root, settings)
    base = reader.base(0)
    _, bundle, _ = reader.model(1, base)
    binding = workspace.results / "evaluation/cycle_1/calibration_binding.json"
    binding.write_text("{}")
    with pytest.raises(ValueError, match="Calibration binding"):
        reader.calibration(1, base, bundle)
    binding.unlink()
    missing = verify_workspace(workspace)
    assert dict(missing.stages)["evaluate_cycle_1"].availability == Availability.PARTIAL
    assert missing.latest_complete_matching_cycle is None


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename", ["params.yaml", "results/trained/cycle_1/graph_binding.json"]
)
def test_workspace_changes_during_evidence_collection(
    retained_workspace, monkeypatch, filename
):
    """Suppress publication if controls or a result member change after status."""
    workspace = retained_workspace

    def changing(root):
        result = local_status(root)
        path = root / filename
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(provenance, "local_status", changing)
    result = verify_workspace(workspace)
    assert result.stamp is None and result.latest_complete_matching_cycle is None
    assert any("changed during verification" in reason for reason in result.reasons)


def test_results_refresh_is_explicit_and_cached_scope_is_invalidated(
    tmp_path, monkeypatch
):
    """Opening Results and changing selections use metadata without DVC checks."""
    from streamlit.testing.v1 import AppTest

    _write_settings(tmp_path, 1)
    monkeypatch.chdir(tmp_path)
    reads = []
    refresh = provenance.verify_workspace

    def count(workspace):
        reads.append(workspace.root)
        return refresh(workspace)

    monkeypatch.setattr(provenance, "verify_workspace", count)
    app = (
        AppTest.from_file(str(ROOT / "app.py"))
        .switch_page("dashboard/views/results.py")
        .run()
    )
    assert not app.exception and not reads
    app.button[0].click().run()
    assert not app.exception and len(reads) == 1
    assert any("No local results" in text.value for text in app.markdown)
    app.segmented_control[0].select("Cycle explorer").run()
    assert any(
        "Learning cycle 1 · Missing · Unverified" in text.value for text in app.markdown
    )
    app.selectbox[0].select(0).run()
    assert not app.exception and len(reads) == 1
    path = tmp_path / "params.yaml"
    path.write_bytes(path.read_bytes() + b"# Edited externally\n")
    app.run()
    assert not app.exception and len(reads) == 1
    assert any("Workspace evidence changed" in item.value for item in app.caption)
