"""List, validate, resolve and export experiment configurations."""

import argparse
import sys
from pathlib import Path

from src.configuration import ValidationResult, validate_configuration
from src.configuration_files import (
    persist_configuration,
    preview_configuration,
    snapshot_configuration,
)
from src.configuration_yaml import load_configuration_yaml
from src.presets import (
    list_presets,
    load_preset,
    resolve_imported_configuration,
    resolve_new_configuration,
)


def _report(validation: ValidationResult) -> None:
    for issue in validation.issues:
        print(
            f"{issue.severity.upper()} [{issue.code}] {issue.path} — {issue.message}",
            file=sys.stderr,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List distributed presets.")
    validate = commands.add_parser("validate", help="Validate a complete YAML file.")
    validate.add_argument("file", type=Path)

    resolve = commands.add_parser(
        "resolve", help="Resolve a new preset-based experiment."
    )
    resolve.add_argument("preset", choices=list_presets())
    imported = commands.add_parser("import", help="Resolve an existing experiment.")
    imported.add_argument("file", type=Path)
    for command in (resolve, imported):
        command.add_argument(
            "--overrides", type=Path, help="Apply a YAML mapping of edits."
        )
        command.add_argument(
            "--against", type=Path, help="Compare against an active YAML file."
        )
        command.add_argument(
            "--output", type=Path, help="Export to a file instead of stdout."
        )
        command.add_argument(
            "--overwrite", action="store_true", help="Allow replacing --output."
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the configuration CLI and return a shell exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for preset_id in list_presets():
                preset = load_preset(preset_id)
                print(f"{preset.id}\tv{preset.version}\t{preset.display_name}")
            return 0
        if args.command == "validate":
            validation = validate_configuration(load_configuration_yaml(args.file))
            _report(validation)
            return 0 if validation.is_valid else 1

        if args.overwrite and args.output is None:
            parser.error("--overwrite requires --output.")
        target = (
            snapshot_configuration(args.output) if args.output is not None else None
        )
        active = (
            snapshot_configuration(args.against) if args.against is not None else None
        )
        if target is not None and (
            (active is None and target.contents is not None)
            or (active is not None and active.path == target.path)
        ):
            active = target
        overrides = load_configuration_yaml(args.overrides) if args.overrides else None
        if args.command == "resolve":
            resolved = resolve_new_configuration(load_preset(args.preset), overrides)
        else:
            resolved = resolve_imported_configuration(
                load_configuration_yaml(args.file), overrides
            )

        preview = preview_configuration(resolved, active)
        _report(preview.validation)
        if preview.comparison_warning:
            print(f"WARNING [comparison] {preview.comparison_warning}", file=sys.stderr)
        if active is not None:
            print(
                preview.diff or "No scientific configuration changes.", file=sys.stderr
            )
        if preview.regeneration_paths:
            paths = ", ".join(preview.regeneration_paths)
            print(
                f"INFO [dvc_regeneration] Regenerate the pipeline for changes to {paths}.",
                file=sys.stderr,
            )
        if not preview.validation.is_valid:
            return 1
        if target is None:
            print(preview.yaml, end="")
        else:
            persist_configuration(preview, target, overwrite=args.overwrite)
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR [configuration_file] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
