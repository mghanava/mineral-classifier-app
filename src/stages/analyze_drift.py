"""Module for analyzing drift between base and prediction datasets using permutation tests."""

import os

import torch
import yaml

from src.utilities.drift_detection_utils import perform_permutation_test
from src.utilities.logging_utils import LogTime


def main():
    """Analyze drift between base and prediction datasets using permutation tests (MMD and energy)."""
    dataset_path = "results/data"
    analyze_drift_path = "results/drift_analysis"
    os.makedirs(analyze_drift_path, exist_ok=True)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    N_PERMUTATIONS = params["analyze_drift"]["n_permutations"]
    BANDWIDTH = params["analyze_drift"]["bandwidth"]

    pred_data = torch.load(
        os.path.join(dataset_path, "pred_data.pt"), weights_only=False
    )
    base_data = torch.load(
        os.path.join(dataset_path, "base_data.pt"), weights_only=False
    )
    with LogTime(task_name="\nAnalyzing drift"):
        print("Assessing domain shift between datasets ...\n")
        mmd_statistic, mmd_p_value = perform_permutation_test(
            base_data.unscaled_features,
            pred_data.unscaled_features,
            n_permutations=N_PERMUTATIONS,
            method="mmd",
            bandwidth=BANDWIDTH,
        )
        energy_stat, energy_p_val = perform_permutation_test(
            base_data.unscaled_features,
            pred_data.unscaled_features,
            n_permutations=N_PERMUTATIONS,
            method="energy",
        )

        # Save the MMD results
        with open(os.path.join(analyze_drift_path, "drift_results.txt"), "w") as f:
            f.write(f"MMD Statistic: {mmd_statistic}\n")
            f.write(f"p-value: {mmd_p_value}\n")
            if mmd_p_value < 0.05:
                f.write("Domain shift detected between base and prediction datasets.\n")
            else:
                f.write(
                    "No significant domain shift detected between base and prediction datasets.\n"
                )
            f.write(f"Energy Statistic: {energy_stat}\n")
            f.write(f"p-value: {energy_p_val}\n")
            if energy_p_val < 0.05:
                f.write("Domain shift detected between base and prediction datasets.\n")
            else:
                f.write(
                    "No significant domain shift detected between base and prediction datasets.\n"
                )


if __name__ == "__main__":
    main()
