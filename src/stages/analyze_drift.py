"""Module for analyzing drift between base and prediction datasets using permutation tests."""

import os

import torch
import yaml

from src.utilities.drift_detection_utils import (
    AnalyzeDrift,
)
from src.utilities.logging_utils import LogTime


def main():
    """Analyze drift between base and prediction datasets using permutation tests (MMD and energy)."""
    dataset_path = "results/data"
    analyze_drift_path = "results/drift_analysis"
    os.makedirs(analyze_drift_path, exist_ok=True)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    N_PERMUTATIONS = params["analyze_drift"]["n_permutations"]
    GAMMA = params["analyze_drift"]["gamma"]

    pred_data = torch.load(
        os.path.join(dataset_path, "pred_data.pt"), weights_only=False
    )
    base_data = torch.load(
        os.path.join(dataset_path, "base_data.pt"), weights_only=False
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
