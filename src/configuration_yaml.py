"""Read scientific YAML using the same scalar types as stage consumers."""

from pathlib import Path

import yaml
from yaml.events import AliasEvent


class _ConfigurationLoader(yaml.SafeLoader):
    _supported_tags = frozenset(
        f"tag:yaml.org,2002:{kind}"
        for kind in ("bool", "float", "int", "map", "null", "seq", "str")
    )

    def compose_node(self, parent, index):
        event = self.peek_event()
        if isinstance(event, AliasEvent) or event.anchor is not None:
            raise ValueError(
                f"YAML anchors and aliases are unsupported at {event.start_mark}."
            )
        node = super().compose_node(parent, index)
        if node.tag not in self._supported_tags:
            raise ValueError(f"Unsupported YAML tag {node.tag!r} at {node.start_mark}.")
        return node

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError(f"Expected a YAML mapping at {node.start_mark}.")
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError(f"Use a string mapping key at {key_node.start_mark}.")
            if key in mapping:
                raise ValueError(
                    f"Duplicate mapping key {key!r} at {key_node.start_mark}."
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def parse_configuration_yaml(contents: str | bytes, *, source: str = "YAML") -> dict:
    """Parse one plain YAML mapping or raise ValueError with source context."""
    try:
        document = yaml.load(contents, Loader=_ConfigurationLoader)
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
        yaml.YAMLError,
    ) as error:
        raise ValueError(f"Cannot parse {source}. {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{source} must contain one YAML mapping.")
    return document


def load_configuration_yaml(path: Path) -> dict:
    """Read and parse a YAML file without accessing other files."""
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read {str(path)!r}. {error}") from error
    return parse_configuration_yaml(contents, source=str(path))


def dump_configuration_yaml(configuration: dict, *, sort_keys: bool = False) -> str:
    """Serialize scientific YAML with safe, stage-compatible scalar types."""
    try:
        return yaml.safe_dump(configuration, sort_keys=sort_keys, allow_unicode=True)
    except (yaml.YAMLError, RecursionError) as error:
        raise ValueError(
            f"Cannot serialize the configuration as scientific YAML. {error}"
        ) from error
