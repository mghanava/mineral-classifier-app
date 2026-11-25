"""Exercise the CLI contract without DVC or training."""

import subprocess
import sys

import pytest

from scripts.configure import main
from src.configuration_files import preview_configuration
from src.configuration_yaml import load_configuration_yaml, parse_configuration_yaml
from src.presets import apply_preset, load_preset


@pytest.fixture
def existing(tmp_path):
    """Create a complete configuration for CLI tests."""
    path = tmp_path / "params.yaml"
    path.write_text(preview_configuration(apply_preset(load_preset("research"))).yaml)
    return path


def test_module_lists_versioned_presets():
    """Expose a runnable module and return success."""
    process = subprocess.run(
        [sys.executable, "-m", "scripts.configure", "list"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.splitlines() == [
        "quick_demo\tv1\tQuick demo",
        "standard\tv1\tStandard",
        "research\tv1\tResearch",
    ]


def test_preview_stdout_is_yaml_with_separate_diagnostics(existing, tmp_path, capsys):
    """Render an override and semantic diff without writing an active file."""
    overrides = tmp_path / "edits.yaml"
    overrides.write_text("cycles: 0\ndata:\n  base:\n    seed: null\n")
    original = existing.read_bytes()
    assert (
        main(
            [
                "resolve",
                "quick_demo",
                "--overrides",
                str(overrides),
                "--against",
                str(existing),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    result = parse_configuration_yaml(captured.out)
    assert result["cycles"] == 0 and result["data"]["base"]["seed"] is None
    assert "WARNING [unreachable_budget]" in captured.err
    assert "INFO [dvc_regeneration]" in captured.err
    assert "-cycles: 3" in captured.err and "+cycles: 0" in captured.err
    assert existing.read_bytes() == original


def test_validate_import_export_and_overwrite(existing, tmp_path, capsys):
    """Validate strictly, then resolve an omission only via the import command."""
    assert main(["validate", str(existing)]) == 0
    existing.write_text(existing.read_text().replace("    seed: 110\n", ""))
    assert main(["validate", str(existing)]) == 1
    assert "ERROR [missing_key] data.base.seed" in capsys.readouterr().err
    output = tmp_path / "resolved.yaml"
    args = ["import", str(existing), "--output", str(output)]
    assert main(args) == 0
    assert not capsys.readouterr().out
    assert load_configuration_yaml(output)["data"]["base"]["seed"] == 42
    output.write_text("existing output\n")
    assert main(args) == 1
    assert output.read_text() == "existing output\n"
    assert "Explicitly authorize replacement" in capsys.readouterr().err
    assert main([*args, "--overwrite"]) == 0
    assert load_configuration_yaml(output)["data"]["base"]["seed"] == 42


def test_cli_errors_do_not_write(tmp_path, capsys):
    """Keep diagnostics separate and use exit 2 for bad command arguments."""
    output = tmp_path / "experiment.yaml"
    for contents, expected in (
        ("cycles: 1\ncycles: 2\n", "Duplicate mapping key"),
        ("cycles: -1\n", "ERROR [out_of_range] cycles"),
    ):
        overrides = tmp_path / "overrides.yaml"
        overrides.write_text(contents)
        assert (
            main(
                [
                    "resolve",
                    "standard",
                    "--overrides",
                    str(overrides),
                    "--output",
                    str(output),
                ]
            )
            == 1
        )
        captured = capsys.readouterr()
        assert not captured.out
        assert expected in captured.err
        assert not output.exists()
    with pytest.raises(SystemExit) as invalid_arguments:
        main(["resolve", "standard", "--overwrite"])
    assert invalid_arguments.value.code == 2
