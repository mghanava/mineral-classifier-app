import argparse
import os

import torch
import yaml

from src.utilities.drift_detection_utils import (
    AnalyzeDrift,
)
from src.utilities.logging_utils import LogTime


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    """Analyze drift between base and prediction datasets for a given cycle."""
    parser = argparse.ArgumentParser(
        description="Analyze drift for a given cycle."
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Current cycle number"
    )
    args = parser.parse_args()
    cycle_num = args.cycle

    params = load_params()

    N_PERMUTATIONS = params["analyze_drift"]["n_permutations"]
    GAMMA = params["analyze_drift"]["gamma"]

    base_data_path = f"results/data/base/cycle_{cycle_num}"
    pred_data_path = f"results/data/prediction/cycle_{cycle_num}"
    analyze_drift_path = f"results/drift_analysis/cycle_{cycle_num}"
    os.makedirs(analyze_drift_path, exist_ok=True)

    pred_data = torch.load(
        os.path.join(pred_data_path, "pred_data.pt"),
        weights_only=False,
    )
    base_data = torch.load(
        os.path.join(base_data_path, "base_data.pt"),
        weights_only=False,
    )

    with LogTime(task_name="\nAnalyzing drift"):
        print("Assessing domain shift between datasets ...\n")
        # Check your data characteristics first
        analyze_drift = AnalyzeDrift(
            base_data=base_data,
            pred_data=pred_data,
            save_path=analyze_drift_path,
            gamma=GAMMA,
            n_permutations=N_PERMUTATIONS,
        )
        analyze_drift.export_drift_analysis_to_file()
        analyze_drift.export_drift_analysis_plots()


if __name__ == "__main__":
    main()