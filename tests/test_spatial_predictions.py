"""Verify spatial readers with disposable fixture artifacts and optional retained data."""

import json
import os
import shutil
from hashlib import md5
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from dashboard import provenance
from dashboard.artifacts import Workspace
from dashboard.dvc_evidence import LocalHashes, dvc_mapping
from dashboard.provenance import Availability, Correspondence, verify_workspace
from dashboard.spatial_data import Connectivity, PopulationKind, read_population
from dashboard.spatial_predictions import ProbabilityFamily, verify_prediction_join
from scripts.setup_dvc import pipeline_definition
from src.configuration_files import configuration_operation
from src.configuration_yaml import dump_configuration_yaml, parse_configuration_yaml
from src.utilities.graph_protocol import (
    graph_digest,
    model_graph_settings,
    resolve_graph_protocol,
)
from src.utilities.model_bundle import (
    publish_model,
    read_model_bundle,
    write_calibration_binding,
)
from src.utilities.preprocessing import (
    FittedPreprocessing,
    RawDataset,
    TrainingPartition,
)
from src.utilities.raw_graphs import (
    build_cycle_graphs,
    build_development_graph,
    build_raw_graph,
    read_raw_graph,
    synthetic_schema,
)
from src.utilities.training_schedule import FoldTrainingResult, final_training_plan

ROOT = Path(__file__).resolve().parents[1]
RETAINED = Path("/tmp/mineral-m35aii-quick.JeKROY")


def _graph_file(workspace, cycle):
    return workspace.results / f"data/prediction/cycle_{cycle}/pred_data.pt"


def _prediction_file(workspace, cycle):
    return Path(f"prediction/cycle_{cycle}/predictions.csv")


def _load(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def _persist_graphs(workspace, raw, settings, snapshot):
    directory = workspace.results / f"data/base/cycle_{snapshot}"
    directory.mkdir(parents=True)
    protocol = resolve_graph_protocol(settings)
    graph, folds, test, calibration = build_cycle_graphs(
        raw, protocol, settings["data"]["base"]
    )
    for name, graph_data in (
        ("base", graph),
        ("fold", folds),
        ("test", test),
        ("calib", calibration),
    ):
        torch.save(graph_data, directory / f"{name}_data.pt")
    return folds, test, calibration


def _write_bundle(workspace, settings, cycle, parent, folds, test, calibration):
    protocol = resolve_graph_protocol(settings)
    name = settings["default_model"]
    development = build_development_graph(parent, folds, test, calibration, protocol)
    partition = TrainingPartition(
        parent.dataset_sha256, None, development.original_node_indices.numpy()
    )
    fitted_state = FittedPreprocessing(
        settings["data"]["scaler_type"],
        parent.feature_names,
        parent.class_names,
        parent.dataset_sha256,
        partition.partition_sha256,
        None,
    )
    model_settings = model_graph_settings(protocol, name, settings)
    plan = final_training_plan(
        [FoldTrainingResult(1.0, 1, None, [0.01])]
        * settings["data"]["base"]["n_splits"],
        1,
    )
    training = {
        "strategy": "full_development_pool",
        "dataset_sha256": parent.dataset_sha256,
        "partition_sha256": partition.partition_sha256,
        "observation_count": development.num_nodes,
        **plan,
    }
    trained = workspace.results / f"trained/cycle_{cycle}"
    trained.mkdir(parents=True)
    publish_model(
        trained,
        torch.nn.Module(),
        fitted_state,
        model_settings,
        development.graph_sha256,
        training,
    )
    _, bundle, _ = read_model_bundle(
        trained, parent.feature_names, parent.class_names, protocol
    )
    evaluation = workspace.results / f"evaluation/cycle_{cycle}"
    evaluation.mkdir(parents=True)
    calibrator = evaluation / "calibrator.pt"
    calibrator.write_bytes(f"Calibration binding fixture {cycle}".encode())
    write_calibration_binding(
        evaluation, bundle, parent.dataset_sha256, calibration.split_sha256
    )


def _raw(settings, prefix, count, *, start=0):
    features, classes = synthetic_schema(settings["data"]["base"])
    rng = np.random.default_rng(start)
    labels = np.arange(count, dtype=np.int64) % len(classes)
    return RawDataset(
        coordinates=rng.normal(size=(count, 3)),
        features=rng.normal(size=(count, len(features))),
        sample_ids=tuple(f"{prefix}-{index}" for index in range(count)),
        feature_names=features,
        class_names=classes,
        labels=labels,
    )


def _write_prediction(workspace, raw, cycle):
    path = workspace.results / _prediction_file(workspace, cycle)
    path.parent.mkdir(parents=True)
    count = len(raw.class_names)
    rng = np.random.default_rng(cycle)
    calibrated = rng.random((len(raw.sample_ids), count), dtype=np.float32)
    uncalibrated = rng.random(calibrated.shape, dtype=np.float32)
    calibrated[0] = np.eye(count, dtype=np.float32)[0]
    calibrated[1] = 1
    families = (calibrated, uncalibrated)
    for scores in families:
        scores /= scores.sum(axis=1, keepdims=True)
    columns = {
        f"{prefix}_prob_class_{i}": scores[:, i]
        for prefix, scores in zip(("calib", "uncalib"), families, strict=True)
        for i in range(count)
    }
    columns.update(predicted_label=calibrated.argmax(axis=1), true_label=raw.labels)
    for family, scores in zip(ProbabilityFamily, families, strict=True):
        scores = torch.from_numpy(scores)
        columns[f"{family}_entropy"] = (
            -(scores * (scores + 1e-8).log()).sum(dim=1).numpy()
        )
    pd.DataFrame(columns).to_csv(path, index=False)


def _record_fixture_lock(workspace, settings, pipeline):
    """Record fixture file identities with DVC's locked codec, without running a stage."""
    root = workspace.root
    hashes = LocalHashes(root)
    records = {}

    def entry(relative):
        # Calculate metadata with the same codec used by DVC evidence checks.
        hashes.matches({"path": relative, "hash": "md5", "md5": "0" * 32})
        digest, size, nfiles = hashes.cache[relative]
        value = {"path": relative, "hash": "md5", "md5": digest, "size": size}
        if nfiles is not None:
            value["nfiles"] = nfiles
        return value

    for stage, spec in pipeline.items():
        record = {
            "cmd": spec["cmd"],
            "deps": [entry(path) for path in spec["deps"]],
            "outs": [entry(path) for path in spec["outs"]],
        }
        if spec.get("params"):
            record["params"] = {
                "params.yaml": provenance._tracked(settings, spec["params"])
            }
        records[stage] = record
    (root / "dvc.lock").write_text(yaml.safe_dump({"schema": "2.0", "stages": records}))


@pytest.fixture(scope="module")
def spatial_fixture(tmp_path_factory):
    """Create two tiny known-order graph/CSV fixtures without fitting or inference."""
    workspace = Workspace(tmp_path_factory.mktemp("spatial-fixture"))
    root = workspace.root
    settings = parse_configuration_yaml((ROOT / "params.yaml").read_bytes())
    settings["cycles"] = 2
    settings["evaluate"]["verbose"] = False
    (root / ".dvc").mkdir()
    (root / ".dvc/config").write_text("[core]\n    no_scm = True\n")
    (root / "params.yaml").write_text(dump_configuration_yaml(settings))
    protocol = resolve_graph_protocol(settings)

    parent = _raw(settings, "base", 96)
    for cycle in (1, 2):
        partitions = _persist_graphs(workspace, parent, settings, cycle - 1)
        batch = _raw(settings, f"batch-{cycle}", 16, start=cycle)
        graph = build_raw_graph(
            batch, protocol, "prediction", source_parent=parent.dataset_sha256
        )
        path = _graph_file(workspace, cycle)
        path.parent.mkdir(parents=True)
        torch.save(graph, path)
        _write_prediction(workspace, batch, cycle)
        _write_bundle(workspace, settings, cycle, parent, *partitions)
        metrics = {"acc": 0.5, "f1": 0.5, "mcc": 0.25}
        pred_metrics = workspace.results / f"prediction/cycle_{cycle}/metrics.json"
        pred_metrics.write_text(json.dumps(metrics))
        eval_metrics = workspace.results / f"evaluation/cycle_{cycle}/metrics.json"
        scores = {**metrics, "ece": 0.1, "mce": 0.2}
        eval_metrics.write_text(
            json.dumps({"calibrated": scores, "uncalibrated": scores})
        )
        parent = RawDataset(
            coordinates=np.vstack((parent.coordinates, batch.coordinates)),
            features=np.vstack((parent.features, batch.features)),
            sample_ids=parent.sample_ids + batch.sample_ids,
            labels=np.r_[parent.labels, batch.labels],
            class_names=parent.class_names,
            feature_names=parent.feature_names,
        )
    _persist_graphs(workspace, parent, settings, 2)
    pipeline = pipeline_definition(settings)["stages"]
    (root / "dvc.yaml").write_text(yaml.safe_dump({"stages": pipeline}))
    for stage, spec in pipeline.items():
        for path in spec["deps"]:
            if path.startswith("src/"):
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, destination)
        members = (
            provenance._members(settings, 1) | provenance._members(settings, 2)
        ).get(stage, ())
        for output in spec["outs"]:
            for path in [Path(output) / name for name in members] or [Path(output)]:
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(b"Fixture diagnostic\n")
    _record_fixture_lock(workspace, settings, pipeline)
    return workspace


@pytest.fixture
def workspace(spatial_fixture, tmp_path, monkeypatch):
    """Use an independent copy for all routine cases and keep content checks real."""
    monkeypatch.setattr(provenance, "local_status", lambda root: {})
    return Workspace(shutil.copytree(spatial_fixture.root, tmp_path / "workspace"))


def _relock_fixture(workspace):
    settings = parse_configuration_yaml(workspace.params.read_bytes())
    _record_fixture_lock(workspace, settings, pipeline_definition(settings)["stages"])


def _check_alignment(workspace, cycle):
    results = verify_prediction_join(workspace, cycle)
    assert results.evidence.correspondence == Correspondence.MATCHING, (
        results.evidence.reasons
    )
    joined = results.joined
    assert joined is not None
    raw = read_raw_graph(_load(_graph_file(workspace, cycle)))
    csv = workspace.read(_prediction_file(workspace, cycle))
    np.testing.assert_array_equal(joined.population.coordinates, raw.coordinates)
    np.testing.assert_array_equal(joined.population.raw_features, raw.features)
    np.testing.assert_array_equal(joined.population.observed_class_ids, raw.labels)
    assert joined.population.sample_ids == raw.sample_ids
    assert joined.population.feature_names == raw.feature_names
    assert joined.population.class_names == raw.class_names
    assert (
        joined.population.coordinates.dtype
        == joined.population.raw_features.dtype
        == np.float64
    )
    assert joined.population.kind == PopulationKind.INCOMING
    assert joined.still_observed(workspace)
    for family, prefix in zip(ProbabilityFamily, ("calib", "uncalib"), strict=True):
        scores = joined.scores(family)
        expected = csv.filter(regex=f"^{prefix}_prob_class_").to_numpy()
        np.testing.assert_allclose(scores.probabilities, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            scores.entropy_nats, csv[f"{family}_entropy"], rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            scores.entropy_nats,
            -(expected * np.log(expected + 1e-8)).sum(axis=1),
            atol=1e-6,
        )
        np.testing.assert_allclose(scores.maximum_probability(), expected.max(axis=1))
        np.testing.assert_allclose(scores.selected_probability(0), expected[:, 0])
        np.testing.assert_array_equal(
            joined.correctness(family), expected.argmax(axis=1) == raw.labels
        )
    return joined


@pytest.mark.parametrize("cycle", [1, 2])
def test_fixture_alignment(workspace, cycle):
    """Verify both score families and exact raw observations for each fixture cycle."""
    _check_alignment(workspace, cycle)


@pytest.mark.integration
def test_fixture_uses_actual_dvc_status(spatial_fixture, tmp_path):
    """Use the installed DVC status command and tolerate absent DVC cache objects."""
    workspace = Workspace(shutil.copytree(spatial_fixture.root, tmp_path / "fixture"))
    _check_alignment(workspace, 2)


@pytest.mark.integration
@pytest.mark.parametrize("cycle", [1, 2])
def test_retained_quick_demo(cycle, tmp_path):
    """Verify each genuine retained Quick demo cycle through the full gate."""
    configured = os.environ.get("MINERAL_QUICK_DEMO_WORKSPACE")
    retained = Path(configured).expanduser() if configured else RETAINED
    if not (retained / "dvc.lock").is_file():
        message = f"Retained Quick demo workspace is unavailable at {retained}."
        if configured:
            pytest.fail(message)
        pytest.skip(message)
    root = tmp_path / "retained"
    root.mkdir()
    for name in ("params.yaml", "dvc.yaml", "dvc.lock", ".dvcignore"):
        if (retained / name).exists():
            shutil.copy2(retained / name, root / name)
    for name in ("src", "results"):
        shutil.copytree(
            retained / name, root / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    (root / ".dvc").mkdir()
    shutil.copy2(retained / ".dvc/config", root / ".dvc/config")
    workspace = Workspace(root)
    joined = _check_alignment(workspace, cycle)
    assert joined.population.count == 100
    assert joined.population.membership == ("incoming_batch",) * 100
    base = read_population(workspace, PopulationKind.INPUT_BASE, cycle=cycle)
    assert joined.population.parent_sha256 == base.dataset_sha256
    assert set(joined.population.sample_ids).isdisjoint(base.sample_ids)
    appended = read_population(workspace, PopulationKind.SNAPSHOT, cycle=cycle)
    assert appended.sample_ids == base.sample_ids + joined.population.sample_ids
    final = read_population(workspace, PopulationKind.FINAL_APPEND, cycle=2)
    assert final.count == 640
    assert final.membership == ("input_base",) * 540 + ("incoming_batch",) * 100

    full = joined.export()
    assert (
        full.scope == "full_population"
        and full.row_count == full.population_count == 100
    )
    table = pd.read_csv(BytesIO(full.csv_bytes), float_precision="round_trip")
    assert tuple(table["sample_id"]) == joined.population.sample_ids
    assert set(table["membership"]) == {"incoming_batch"}
    assert set(table["learning_cycle"]) == {cycle}
    np.testing.assert_array_equal(table[["x", "y", "z"]], joined.population.coordinates)
    np.testing.assert_array_equal(
        table["observed_class_name"], joined.population.observed_class_names
    )
    for family in ProbabilityFamily:
        scores = joined.scores(family)
        np.testing.assert_array_equal(
            table.filter(regex=f"^{family}_probability_class_"), scores.probabilities
        )
        np.testing.assert_array_equal(
            table[f"{family}_entropy_nats"], scores.entropy_nats
        )
        np.testing.assert_array_equal(
            table[f"{family}_predicted_class_name"],
            joined.predicted_class_names(family),
        )
    sample_id = joined.population.sample_ids[-1]
    filtered = joined.export(joined.population.view(sample_ids={sample_id}))
    assert filtered.scope == "filtered_view" and filtered.row_count == 1
    assert filtered.population_count == 100
    assert pd.read_csv(BytesIO(filtered.csv_bytes))["sample_id"].tolist() == [sample_id]


def test_partition_selection_and_final_append(workspace):
    """Keep independent partitions, development membership and full base topology distinct."""
    base = read_population(workspace, PopulationKind.SNAPSHOT, cycle=0)
    assert base.topology.kind == Connectivity.DESCRIPTIVE
    assert (
        base.sample_ids
        == read_population(workspace, PopulationKind.INPUT_BASE, cycle=1).sample_ids
    )
    assert read_population(workspace, PopulationKind.SNAPSHOT, cycle=1).count == 112
    development = read_population(workspace, PopulationKind.DEVELOPMENT, cycle=1)
    assert development.topology is None
    test = read_population(workspace, PopulationKind.FINAL_TEST, cycle=1)
    calibration = read_population(workspace, PopulationKind.CALIBRATION, cycle=1)
    memberships = [
        set(population.sample_ids) for population in (development, test, calibration)
    ]
    assert set.union(*memberships) == set(base.sample_ids)
    assert sum(map(len, memberships)) == base.count
    assert calibration.topology.kind == test.topology.kind == Connectivity.INDEPENDENT
    folds = []
    for fold in range(3):
        train = read_population(workspace, PopulationKind.CV_TRAIN, cycle=1, fold=fold)
        val = read_population(
            workspace, PopulationKind.CV_VALIDATION, cycle=1, fold=fold
        )
        assert train.fold == val.fold == fold
        assert set(train.sample_ids).isdisjoint(val.sample_ids)
        assert set(train.sample_ids) | set(val.sample_ids) == memberships[0]
        folds.extend(val.sample_ids)
        rows = [base.sample_ids.index(sid) for sid in val.sample_ids]
        np.testing.assert_array_equal(val.raw_features, base.raw_features[rows])
        np.testing.assert_array_equal(val.coordinates, base.coordinates[rows])
        np.testing.assert_array_equal(
            val.observed_class_ids, base.observed_class_ids[rows]
        )
    assert sorted(folds) == sorted(development.sample_ids)
    with pytest.raises(ValueError, match="fold"):
        read_population(workspace, PopulationKind.CV_VALIDATION, cycle=1, fold=3)
    final = read_population(workspace, PopulationKind.FINAL_APPEND, cycle=2)
    inputs = read_population(workspace, PopulationKind.INPUT_BASE, cycle=2)
    incoming = read_population(workspace, PopulationKind.INCOMING, cycle=2)
    assert final.sample_ids == inputs.sample_ids + incoming.sample_ids
    assert final.count == 128
    assert (
        final.membership
        == ("input_base",) * inputs.count + ("incoming_batch",) * incoming.count
    )


def test_filter_graph_edges_and_statistics(workspace):
    """Show unique physical links with visible endpoints and keep original graph statistics."""
    population = read_population(workspace, PopulationKind.INCOMING, cycle=1)
    graph = _load(_graph_file(workspace, 1))
    edges = population.view().edge_ids()
    stats = population.topology.statistics
    ids = population.sample_ids
    expected = {
        frozenset((ids[a], ids[b])) for a, b in graph.edge_index.T.tolist() if a != b
    }
    assert len(edges) == len(expected) == stats.undirected_edge_count
    assert stats.self_loop_count == population.count
    assert {frozenset(edge) for edge in edges} == expected
    np.testing.assert_array_equal(
        population.topology.edge_indices.array(), graph.edge_index.numpy()
    )
    a, b = edges[0]
    filtered = population.view(sample_ids={a, b})
    assert filtered.edge_ids() == ((a, b),)
    assert filtered.count == 2
    assert filtered.population.count == stats.node_count == 16
    z = population.coordinates[0, 2]
    only_one = population.view(
        depth=(z, z), class_ids={int(population.observed_class_ids[0])}
    )
    assert only_one.count == 1
    assert only_one.edge_ids() == ()


def test_export_and_in_memory_snapshot(workspace):
    """Keep exports scoped, immutable across time, and detached from caller edits."""
    joined = _check_alignment(workspace, 1)
    raw_path = workspace.results / _prediction_file(workspace, 1)
    original = raw_path.read_bytes()
    full = joined.export()
    assert (
        full.scope == "full_population"
        and full.row_count == full.population_count == 16
    )
    table = pd.read_csv(BytesIO(full.csv_bytes))
    assert tuple(table["sample_id"]) == joined.population.sample_ids
    assert set(table["learning_cycle"]) == {1}
    assert set(table["export_scope"]) == {"full_population"}
    assert json.loads(table["class_names_in_probability_order"][0]) == list(
        joined.population.class_names
    )
    for family in ProbabilityFamily:
        np.testing.assert_array_equal(
            table[f"{family}_predicted_class_name"],
            joined.predicted_class_names(family),
        )
        np.testing.assert_allclose(
            table[f"{family}_entropy_nats"], joined.scores(family).entropy_nats
        )
    view = joined.population.view(sample_ids={joined.population.sample_ids[2]})
    filtered = joined.export(view)
    assert (
        filtered.scope == "filtered_view"
        and filtered.row_count == 1
        and filtered.population_count == 16
    )
    assert pd.read_csv(BytesIO(filtered.csv_bytes))["sample_id"].tolist() == [
        joined.population.sample_ids[2]
    ]
    empty = joined.export(joined.population.view(sample_ids=set()))
    assert empty.scope == "filtered_view" and empty.row_count == 0
    assert pd.read_csv(BytesIO(empty.csv_bytes)).empty
    assert (
        joined.export(
            joined.population.view(sample_ids=joined.population.sample_ids)
        ).scope
        == "filtered_view"
    )
    with pytest.raises(ValueError, match="different verified"):
        joined.export(
            read_population(workspace, PopulationKind.INCOMING, cycle=1).view()
        )
    coordinates = joined.population.coordinates
    with pytest.raises(ValueError):
        coordinates.setflags(write=True)
    assert coordinates is not joined.population.coordinates
    assert joined.population.coordinates.dtype == np.float64
    changed = joined.enriched_table()
    changed.loc[0, "x"] = 6
    assert joined.export() == full
    assert raw_path.read_bytes() == original
    raw_path.write_bytes(b"modified")
    assert not joined.still_observed(workspace)
    assert joined.export() == full
    assert verify_prediction_join(workspace, 1).joined is None


@pytest.mark.parametrize(
    "change",
    [
        "same-class-reorder",
        "cross-cycle-csv",
        "cross-cycle-graph",
        "directory-member",
        "added-member",
        "source",
    ],
)
def test_changed_workspace_never_reuses_overview_evidence(workspace, change):
    """Refuse stale evidence and keep independent diagnostic readers available."""
    # A real overview-check snapshot guards the exact workspace state independently.
    previous = verify_workspace(workspace)
    assert previous.still_observed(workspace)
    assert (
        dict(previous.stages)["predict_cycle_1"].correspondence
        == Correspondence.MATCHING
    )
    assert verify_prediction_join(workspace, 1).joined is not None
    path = workspace.results / _prediction_file(workspace, 1)
    if change == "same-class-reorder":
        table = pd.read_csv(path)
        assert table.loc[0, "true_label"] == table.loc[4, "true_label"]
        order = np.arange(len(table))
        order[[0, 4]] = order[[4, 0]]
        rows = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join([rows[0], *(rows[i + 1] for i in order)]))
    elif change == "cross-cycle-csv":
        shutil.copyfile(workspace.results / _prediction_file(workspace, 2), path)
    elif change == "cross-cycle-graph":
        shutil.copyfile(_graph_file(workspace, 2), _graph_file(workspace, 1))
    elif change == "directory-member":
        (path.parent / "histograms.png").write_bytes(b"new")
    elif change == "added-member":
        (path.parent / "extra.csv").write_bytes(b"new")
    else:
        (workspace.root / "src/utilities/pred_utils.py").write_text(
            "# changed producer"
        )
    assert not previous.still_observed(workspace)
    result = verify_prediction_join(workspace, 1)
    assert (
        result.joined is None
        and result.evidence.correspondence == Correspondence.CONFLICT
    )
    assert "DVC lock content differs" in " ".join(result.evidence.reasons)
    assert len(workspace.read(_prediction_file(workspace, 1))) == 16
    if change != "cross-cycle-graph":
        assert read_population(workspace, PopulationKind.INCOMING, cycle=1).count == 16


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("parent", "base lineage"),
        ("duplicate-ids", "Sample IDs"),
        ("graph-version", "graph protocol version"),
        ("graph-shape", "edge weights, attributes or distances"),
        ("calibration-binding", "fitting partition"),
        ("model-binding", "graph binding"),
        ("producer-order", "Producer row order"),
    ],
)
def test_bad_scientific_bindings_even_when_fixture_lock_matches(
    workspace, change, reason
):
    """Fail semantic or producer-order checks after local DVC content matches."""
    if change in {"parent", "duplicate-ids", "graph-version", "graph-shape"}:
        path = _graph_file(workspace, 1)
        graph = _load(path)
        if change == "parent":
            graph.source_parent_sha256 = "0" * 64
        elif change == "duplicate-ids":
            graph.sample_ids = (graph.sample_ids[0], *graph.sample_ids[:-1])
        elif change == "graph-version":
            graph.graph_protocol["version"] = True
        else:
            graph.edge_distance = graph.edge_distance[:-1]
        graph.graph_sha256 = graph_digest(graph)
        torch.save(graph, path)
    elif change == "producer-order":
        path = workspace.root / "src/utilities/pred_utils.py"
        path.write_text(path.read_text() + "\n# Unreviewed revision\n")
    else:
        directory = "evaluation" if change == "calibration-binding" else "trained"
        binding_name = (
            "calibration_binding"
            if change == "calibration-binding"
            else "graph_binding"
        )
        path = workspace.results / f"{directory}/cycle_1/{binding_name}.json"
        binding = json.loads(path.read_bytes())
        key = (
            "calibration_split_sha256"
            if change == "calibration-binding"
            else "weights_sha256"
        )
        binding[key] = "0" * 64
        path.write_text(json.dumps(binding))
    _relock_fixture(workspace)
    result = verify_prediction_join(workspace, 1)
    assert result.joined is None and result.evidence.locked_outputs_match
    assert reason in " ".join(result.evidence.reasons)


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("calib_prob_class_0", "nan", "finite and within"),
        ("uncalib_prob_class_0", "-0.01", "finite and within"),
        ("calib_prob_class_0", "1.01", "finite and within"),
        ("calib_prob_class_0", "0.9", "sum to one"),
        ("uncalib_prob_class_0", "0.9", "sum to one"),
        ("uncalib_prob_class_0", "bad", "invalid number"),
        ("true_label", "1", "ordered labels"),
        ("predicted_label", "1", "argmax"),
        ("calibrated_entropy", "0.01", "calibrated entropy"),
        ("uncalibrated_entropy", "0.01", "uncalibrated entropy"),
    ],
)
def test_bad_numeric_csv_after_hash_checks(workspace, column, value, reason):
    """Check scientific CSV contents even when a matching fixture lock is present."""
    path = workspace.results / _prediction_file(workspace, 1)
    headers, row, tail = path.read_text().split("\n", maxsplit=2)
    fields = row.split(",")
    fields[headers.split(",").index(column)] = value
    path.write_text("\n".join((headers, ",".join(fields), tail)))
    _relock_fixture(workspace)
    result = verify_prediction_join(workspace, 1)
    assert result.joined is None and result.evidence.locked_outputs_match
    assert reason in " ".join(result.evidence.reasons)


@pytest.mark.parametrize("change", ["header", "row-count", "row-width"])
def test_invalid_csv_shape_after_hash_checks(workspace, change):
    """Refuse malformed CSV columns, row counts and field widths."""
    path = workspace.results / _prediction_file(workspace, 1)
    headers, row, tail = path.read_text().split("\n", maxsplit=2)
    if change == "header":
        headers = headers.replace("calib_prob_class_0", "wrong")
    elif change == "row-count":
        tail = row + "\n" + tail
    else:
        row += ",0"
    path.write_text("\n".join((headers, row, tail)))
    _relock_fixture(workspace)
    result = verify_prediction_join(workspace, 1)
    assert result.joined is None and result.evidence.locked_outputs_match
    assert "CSV" in " ".join(result.evidence.reasons)


@pytest.mark.parametrize("change_at", ["status", "graph-load", "raw-load"])
def test_mid_read_change_fails_closed(workspace, monkeypatch, change_at):
    """Reject files changed or replaced during verification or loading."""
    path = workspace.results / _prediction_file(workspace, 1)
    if change_at == "status":

        def modify(root):
            lock = root / "dvc.lock"
            lock.write_bytes(lock.read_bytes() + b"\n")
            return {}

        monkeypatch.setattr(provenance, "local_status", modify)
    else:
        original = torch.load

        def modify(source, *args, **kwargs):
            result = original(source, *args, **kwargs)
            if change_at == "graph-load" and isinstance(source, BytesIO):
                path.write_bytes(path.read_bytes() + b"\n")
            elif change_at == "raw-load" and source == _graph_file(workspace, 1):
                # Rewrite identical content. Identity metadata still invalidates the read.
                source.write_bytes(source.read_bytes())
            return result

        monkeypatch.setattr(torch, "load", modify)
    if change_at == "raw-load":
        with pytest.raises(ValueError, match="Workspace files changed"):
            read_population(workspace, PopulationKind.INCOMING, cycle=1)
    else:
        assert "Workspace files changed" in " ".join(
            verify_prediction_join(workspace, 1).evidence.reasons
        )


def test_captured_bytes_must_match_lock_even_without_stamp_change(
    workspace, monkeypatch
):
    """Validate the CSV snapshot itself against the directory manifest."""
    target = workspace.results / _prediction_file(workspace, 1)
    read_bytes = Path.read_bytes

    def read(path):
        payload = read_bytes(path)
        return payload + b"\n" if path == target else payload

    monkeypatch.setattr(Path, "read_bytes", read)
    result = verify_prediction_join(workspace, 1)
    assert result.joined is None
    assert "Loaded bytes differ from DVC evidence" in " ".join(result.evidence.reasons)


def test_missing_busy_historic_and_partial(workspace):
    """Prompt retries for busy workspaces and preserve standalone diagnostics."""
    with configuration_operation(workspace.params):
        busy = verify_prediction_join(workspace, 1)
        assert busy.joined is None
        assert "pipeline run is active" in " ".join(busy.evidence.reasons)
    historical = verify_prediction_join(workspace, 3)
    assert historical.joined is None
    assert "declared" in " ".join(historical.evidence.reasons)
    lock = workspace.root / "dvc.lock"
    lock.unlink()
    missing = verify_prediction_join(workspace, 1)
    assert (
        missing.joined is None
        and missing.evidence.correspondence == Correspondence.UNKNOWN
    )
    assert read_population(workspace, PopulationKind.INCOMING, cycle=1).count == 16
    path = workspace.results / _prediction_file(workspace, 1)
    assert workspace.read(_prediction_file(workspace, 1)).shape[0] == 16
    path.unlink()
    partial = verify_prediction_join(workspace, 1)
    assert (
        partial.joined is None and partial.evidence.availability == Availability.PARTIAL
    )


def test_prediction_lock_requires_exact_graph_dependency(workspace):
    """Reject a lock with the wrong graph dependency even when rows look plausible."""
    path = workspace.root / "dvc.lock"
    lock = dvc_mapping(path.read_bytes(), "dvc.lock")
    dependencies = lock["stages"]["predict_cycle_1"]["deps"]
    graph_path = _graph_file(workspace, 1).relative_to(workspace.root).as_posix()
    other = _graph_file(workspace, 2).read_bytes()
    entry = next(entry for entry in dependencies if entry["path"] == graph_path)
    entry["md5"] = md5(other, usedforsecurity=False).hexdigest()
    entry["size"] = len(other)
    path.write_text(yaml.safe_dump(lock))
    result = verify_prediction_join(workspace, 1)
    assert (
        result.joined is None
        and result.evidence.correspondence == Correspondence.CONFLICT
    )
    assert graph_path in " ".join(result.evidence.reasons)


def test_prediction_scope_excludes_unrelated_results(workspace):
    """A join checks its producer ancestry independently of drift and later predictions."""
    (workspace.results / "drift_analysis/cycle_1/drift_metadata.json").unlink()
    (workspace.results / "prediction/cycle_2/predictions.csv").write_text("bad")
    assert verify_prediction_join(workspace, 1).joined is not None
    assert verify_prediction_join(workspace, 2).joined is None


def test_ignored_prediction_member_blocks_join(workspace):
    """Check prediction directory membership even for files ignored by DVC."""
    (workspace.root / ".dvcignore").write_text(
        "/results/prediction/cycle_1/ignored.txt\n"
    )
    (workspace.results / "prediction/cycle_1/ignored.txt").write_text("unexpected")
    result = verify_prediction_join(workspace, 1)
    assert result.joined is None
    assert "Directory files differ from the DVC manifest" in " ".join(
        result.evidence.reasons
    )
