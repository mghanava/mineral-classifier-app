"""Read the saved drift report without running statistical analysis."""

import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from dashboard.dvc_evidence import same_value

# analyze_drift publishes a strict p-value cutoff independently of early stopping.
REPORT_THRESHOLD = 0.05
RAW_REPRESENTATION = "Canonical raw features"
MODEL_REPRESENTATION = "Frozen model feature transform"
REPRESENTATIONS = {
    "marginal": "canonical_raw_float64",
    "multivariate": "frozen_model_transform_float64",
}
METHOD_LABELS = {
    "mmd": "MMD",
    "energy": "Energy distance",
    "wasserstein": "Wasserstein",
    "max_sliced_wasserstein": "Max-sliced Wasserstein",
    "sinkhorn_wasserstein": "Sinkhorn Wasserstein",
}
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"


@dataclass(frozen=True)
class DriftTest:
    """One published test statistic and its report decision."""

    method: str
    statistic: float
    p_value: float

    @property
    def detected(self) -> bool:
        """Apply the producer report threshold to the published p-value."""
        return self.p_value < REPORT_THRESHOLD

    @property
    def finding(self) -> str:
        """Name the per-method report decision."""
        return "Detected" if self.detected else "Not detected"


@dataclass(frozen=True)
class EarlyStopping:
    """The effective permutation stopping policy saved by the producer."""

    enabled: bool
    threshold: float
    minimum: int

    def describe(self, budget: int) -> str:
        """Describe policy eligibility within the recorded budget."""
        if not self.enabled:
            return "Disabled"
        policy = f"After at least {max(1, self.minimum):,} permutations, stop if running p < {self.threshold:g}"
        if self.minimum > budget:
            return f"{policy} · minimum exceeds budget"
        return policy


@dataclass(frozen=True)
class DriftSummary:
    """Bind published tests and effective settings to the two proven cohorts."""

    cycle: int
    reference_sha256: str
    incoming_sha256: str
    reference_count: int
    incoming_count: int
    feature_names: tuple[str, ...]
    scaler: str | None
    preprocessing_sha256: str
    representation: str
    effective_gamma: float
    budget: int
    configured_seed: int | None
    effective_seed: int | None
    early_stopping: EarlyStopping
    wasserstein: str
    projections: int | None
    regularization: float | None
    tests: tuple[DriftTest, ...]

    @property
    def wasserstein_context(self) -> str:
        """Describe the configured variant and its applicable transport settings."""
        label = METHOD_LABELS[self.wasserstein]
        if len(self.feature_names) == 1:
            return f"{label} · shared 1D transport"
        if self.projections is not None:
            return f"{label} · {self.projections:,} projections"
        if self.regularization is not None:
            return f"{label} · regularization {self.regularization:g}"
        return f"{label} · squared Euclidean cost"


def parse_drift_report(report: str, wasserstein: str) -> tuple[DriftTest, ...]:
    """Require the three producer blocks and validate their published decisions."""
    if wasserstein not in (
        "wasserstein",
        "max_sliced_wasserstein",
        "sinkhorn_wasserstein",
    ):
        raise ValueError("Unsupported Wasserstein method.")
    lines = report.rstrip("\n").splitlines()
    if len(lines) != 9:
        raise ValueError("Drift report must contain exactly three test blocks.")
    tests = []
    for index, method in enumerate(("mmd", "energy", wasserstein)):
        statistic, p_value, finding = lines[index * 3 : index * 3 + 3]
        stat = re.fullmatch(rf"{method.upper()} Statistic: ({_NUMBER})", statistic)
        prob = re.fullmatch(rf"p-value: ({_NUMBER})", p_value)
        if stat is None or prob is None:
            raise ValueError(f"Missing or malformed drift report fields for {method}.")
        test = DriftTest(method, float(stat[1]), float(prob[1]))
        if not isfinite(test.statistic) or not 0 <= test.p_value <= 1:
            raise ValueError(f"Invalid drift statistic or p-value for {method}.")
        expected = (
            ("Domain shift" if test.detected else "No domain shift")
            + f" detected between reference and prediction datasets using {method} method."
        )
        if finding != expected:
            raise ValueError(
                f"Drift interpretation disagrees with the report cutoff for {method}."
            )
        tests.append(test)
    return tuple(tests)


def _finite_float(value: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError("Nonfinite drift metadata value.")
    return number


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate drift metadata field.")
    return fields


def read_drift_summary(
    metadata_path: Path,
    report_path: Path,
    *,
    expected: dict,
    reference_count: int,
    incoming_count: int,
) -> DriftSummary:
    """Check metadata against verified inputs, then parse the locked report."""
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_fields,
        parse_float=_finite_float,
        parse_constant=_finite_float,
    )
    if not isinstance(metadata, dict) or any(
        key not in metadata or not same_value(value, metadata[key])
        for key, value in expected.items()
    ):
        raise ValueError(
            "Drift metadata differs from the declared populations, model or settings."
        )

    settings = expected["configuration"]
    if settings["use_max_sliced_wasserstein"]:
        method = "max_sliced_wasserstein"
    elif settings["use_sinkhorn_wasserstein"]:
        method = "sinkhorn_wasserstein"
    else:
        method = "wasserstein"
    if metadata.get("wasserstein_method") != method:
        raise ValueError("Recorded Wasserstein method disagrees with configuration.")
    stopping = settings["early_stopping"] or {
        "enabled": False,
        "confidence_threshold": 0.01,
        "min_permutations": 100,
    }
    if not same_value(metadata.get("effective_early_stopping"), stopping):
        raise ValueError("Missing or inconsistent permutation early-stopping policy.")
    gamma = metadata.get("effective_gamma")
    configured_gamma = settings["gamma"]
    if (
        type(gamma) not in (float, int)
        or not isfinite(gamma)
        or (gamma <= 0 if configured_gamma is None else gamma != configured_gamma)
    ):
        raise ValueError("Missing or invalid effective drift gamma.")

    features = tuple(expected["feature_names"])
    seeds = expected["seed_provenance"]
    return DriftSummary(
        cycle=expected["model_cycle"],
        reference_sha256=expected["reference_dataset_sha256"],
        incoming_sha256=expected["incoming_dataset_sha256"],
        reference_count=reference_count,
        incoming_count=incoming_count,
        feature_names=features,
        scaler=expected["scaler_type"],
        preprocessing_sha256=expected["preprocessing_sha256"],
        representation=expected["representations"]["multivariate"],
        effective_gamma=gamma,
        budget=settings["n_permutations"],
        configured_seed=seeds["configured_seed"],
        effective_seed=seeds["effective_seed"],
        early_stopping=EarlyStopping(
            stopping["enabled"],
            stopping["confidence_threshold"],
            stopping["min_permutations"],
        ),
        wasserstein=method,
        projections=settings["n_projections"]
        if len(features) > 1 and method == "max_sliced_wasserstein"
        else None,
        regularization=settings["sinkhorn_wasserstein_reg"]
        if len(features) > 1 and method == "sinkhorn_wasserstein"
        else None,
        tests=parse_drift_report(report_path.read_text(encoding="utf-8"), method),
    )
