"""Discover and read display artifacts in the trusted local results workspace."""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.io as pio
from PIL import Image
from plotly.graph_objects import Figure

CYCLE_FAMILIES = (
    "data/base",
    "data/prediction",
    "trained",
    "evaluation",
    "prediction",
    "drift_analysis",
)
DISPLAY_FORMATS = {".json", ".html", ".png", ".csv", ".txt"}
AGGREGATE = Path("cycles_performance_analysis/cycles_performance.png")


@dataclass(frozen=True)
class Workspace:
    """Use the same working directory as the configuration and DVC entry points."""

    root: Path

    @property
    def params(self) -> Path:
        """Return the applied configuration path."""
        return self.root / "params.yaml"

    @property
    def results(self) -> Path:
        """Return the display artifact root."""
        return self.root / "results"

    def _local(self, relative: Path) -> Path:
        path = (self.results / relative).resolve()
        # Keep display reads inside the workspace, including through directory links.
        if not path.is_relative_to(self.root.resolve() / "results"):
            raise ValueError("The artifact path leaves the local results directory.")
        return path

    def _entries(self, relative: Path) -> list[Path]:
        try:
            return sorted(self._local(relative).iterdir())
        except (OSError, RuntimeError, ValueError):
            return []

    def available(self, relative: Path) -> bool:
        """Check file presence without reading or verifying its contents."""
        try:
            return self._local(relative).is_file()
        except (OSError, RuntimeError, ValueError):
            return False

    def files(self, directory: Path) -> list[Path]:
        """List selectable display files as results-relative paths."""
        return [
            directory / path.name
            for path in self._entries(directory)
            if path.suffix in DISPLAY_FORMATS and self.available(directory / path.name)
        ]

    def result_cycles(self) -> Iterator[int]:
        """Yield cycles with display files, including cycles beyond applied settings."""
        for family in CYCLE_FAMILIES:
            for directory in self._entries(Path(family)):
                match = re.fullmatch(r"cycle_(0|[1-9][0-9]*)", directory.name)
                if match and self.files(Path(family) / directory.name):
                    yield int(match[1])

    def read(self, relative: Path) -> Figure | pd.DataFrame | bytes | dict | str:
        """Read one chosen display artifact without loading executable model objects."""
        path = self._local(relative)
        if relative.suffix not in DISPLAY_FORMATS or not path.is_file():
            raise ValueError("The display file is unavailable.")
        contents = path.read_bytes()
        if not contents:
            raise ValueError("The display file is empty.")
        if relative.suffix == ".png":
            with Image.open(BytesIO(contents), formats=["PNG"]) as image:
                image.verify()
            return contents
        if relative.suffix == ".csv":
            return pd.read_csv(BytesIO(contents))
        text = contents.decode("utf-8")
        if relative.suffix == ".json":
            document = json.loads(text)
            if not isinstance(document, dict):
                raise ValueError("Expected a JSON object.")
            if relative.parent.parts[:1] == ("data",):
                if not isinstance(document.get("data"), list) or not document["data"]:
                    raise ValueError("Expected a Plotly figure with trace data.")
                return pio.from_json(text)
            return document
        return text
