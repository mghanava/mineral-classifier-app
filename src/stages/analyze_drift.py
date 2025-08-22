import argparse
import os
from pathlib import Path

import torch
import yaml

from src.utilities.drift_detection_utils import AnalyzeDrift
from src.utilities.logging_utils import LogTime


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",  # Previous cycle's data
        "prediction_data": f"results/data/prediction/cycle_{cycle_num}",
        "drift_analysis": f"results/drift_analysis/cycle_{cycle_num}",
    }


def main():
    """Analyze drift between previous base data and current prediction data."""
    parser = argparse.ArgumentParser(description="Analyze drift for a given cycle.")
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")

    # Get cycle-specific paths and ensure directories exist
    paths = get_cycle_paths(cycle_num)
    analysis_path = ensure_directory_exists(paths["drift_analysis"])

    # Load parameters
    params = load_params()
    drift_params = params["analyze_drift"]

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

    # Analyze drift
    print(
        f"\nAnalyzing drift between cycle {cycle_num - 1} base data and cycle {cycle_num} prediction data"
    )
    with LogTime(task_name="\nDrift Analysis"):
        analyzer = AnalyzeDrift(
            base_data=base_data,
            pred_data=pred_data,
            save_path=analysis_path,
            gamma=drift_params["gamma"],
            n_permutations=drift_params["n_permutations"],
            n_projections=drift_params["n_projections"],
            use_max_sliced_wasserstein=drift_params["use_max_sliced_wasserstein"],
            use_sinkhorn_wasserstein=drift_params["use_sinkhorn_wasserstein"],
            early_stopping_config=drift_params.get("early_stopping", {}),
        )
        analyzer.export_drift_analysis_to_file()
        analyzer.export_drift_analysis_plots()

    print(f"\n✓ Drift analysis results saved to {analysis_path}")


if __name__ == "__main__":
    main()
