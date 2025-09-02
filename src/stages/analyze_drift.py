"""Module for analyzing data drift between base and prediction datasets for a given cycle.

This script loads data, performs drift analysis using specified parameters, and saves results and plots.
"""

import argparse
import os

import torch

from src.utilities.drift_detection_utils import AnalyzeDrift
from src.utilities.general_utils import LogTime, ensure_directory_exists, load_params


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
    """Run drift analysis between base data and prediction data for a given cycle.

    Parameters
    ----------
    paths : dict
        Dictionary containing paths for base data, prediction data, and output.
    params : dict
        Dictionary of parameters loaded from configuration.

    Raises
    ------
    FileNotFoundError
        If required data files are not found.

    """
    drift_params = params["analyze_drift"]
    output_path = ensure_directory_exists(paths["output"])
    # Load prediction data (current cycle)
    pred_data_path = os.path.join(paths["prediction_data"], "pred_data.pt")
    if not os.path.exists(pred_data_path):
        raise FileNotFoundError(f"Prediction data not found at {pred_data_path}")
    pred_data = torch.load(pred_data_path, weights_only=False)

    # Load base data (previous cycle)
    base_data_path = os.path.join(paths["base_data"], "base_data.pt")
    if not os.path.exists(base_data_path):
        raise FileNotFoundError(f"Base data not found at {base_data_path}")
    base_data = torch.load(base_data_path, weights_only=False)

    analyzer = AnalyzeDrift(
        base_data=base_data,
        pred_data=pred_data,
        save_path=output_path,
        gamma=drift_params["gamma"],
        n_permutations=drift_params["n_permutations"],
        n_projections=drift_params["n_projections"],
        use_max_sliced_wasserstein=drift_params["use_max_sliced_wasserstein"],
        use_sinkhorn_wasserstein=drift_params["use_sinkhorn_wasserstein"],
        early_stopping_config=drift_params.get("early_stopping", {}),
    )
    analyzer.export_drift_analysis_to_file()
    analyzer.export_drift_analysis_plots()
    print(f"✓ Drift analysis results saved to {output_path}.")


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
