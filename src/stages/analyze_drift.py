"""Module for analyzing data drift between base and prediction datasets for a given cycle.

This script loads data, performs drift analysis using specified parameters, and saves results and plots.
"""

import argparse

from src.utilities.general_utils import (
    LogTime,
    load_params,
)


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",
        "prediction_data": f"results/data/prediction/cycle_{cycle_num}",
        "output": f"results/drift_analysis/cycle_{cycle_num}",
    }


def run_analyze_drift(
    paths: dict,
    params: dict,
):
    """Report that drift scoring is unavailable during the raw artifact cutover."""
    raise ValueError(
        "Drift analysis is temporarily unavailable during the raw-data cutover. "
        "M1.3b-iii will use the same fitted preprocessing for both multivariate inputs."
    )


def main():
    """Parse arguments and run drift analysis for a specified cycle.

    This function handles argument parsing, validation, parameter loading,
    and initiates drift analysis for the given cycle.
    """
    parser = argparse.ArgumentParser(description="Analyze drift for a given cycle.")
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")
    # Load parameters and paths
    params = load_params()
    paths = get_cycle_paths(cycle_num)
    # Run drift analysis
    with LogTime(task_name=f"\nDrift Analysis cycle{cycle_num}"):
        run_analyze_drift(paths, params)


if __name__ == "__main__":
    main()
