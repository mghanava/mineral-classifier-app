import argparse
import os
from typing import Literal

import torch
import yaml

from src.utilities.data_utils import connect_graphs_preserve_weights

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


# Define the allowed similarity metrics
AllowedSimilarityMetrics = Literal["cosine", "euclidean", "dot"]


def prepare_combined_data(
    cycle_num: int,
    base_data_path: str,
    prev_pred_data_path: str,
    prev_combined_data_path: str,
    combined_output_path: str,
    similarity_metric: AllowedSimilarityMetrics,
    top_k: int,
    similarity_threshold: float,
):
    """Prepare combined data for a given cycle in a DVC pipeline.

    For cycle N:
    - If N=1: Only `base_data` is used.
    - If N>1: `combined_data` from N-1 is combined with `pred_data` from N-1.

    """
    # Validate similarity_metric
    if similarity_metric not in {"cosine", "euclidean", "dot"}:
        raise ValueError(
            f"Invalid similarity_metric: {similarity_metric}. "
            "Must be one of: 'cosine', 'euclidean', 'dot'"
        )
    # Ensure the output directory exists
    os.makedirs(combined_output_path, exist_ok=True)

    # Base case: cycle_num = 1 → only load base_data from the current cycle
    if cycle_num == 1:
        print(f"Cycle 1: Using base data from {base_data_path}")
        base_file = os.path.join(base_data_path, "base_data.pt")
        if not os.path.exists(base_file):
            raise FileNotFoundError(f"Base data not found: {base_file}")
        combined_data = torch.load(base_file, weights_only=False)
    else:
        # Load previous combined data (from cycle N-1)
        print(f"Cycle {cycle_num}: Combining data from previous cycle.")
        prev_combined_file = os.path.join(prev_combined_data_path, "combined_data.pt")
        if not os.path.exists(prev_combined_file):
            raise FileNotFoundError(
                f"Previous combined data not found: {prev_combined_file}"
            )
        combined_data = torch.load(prev_combined_file, weights_only=False)

        # Load new pred_data (from cycle N-1)
        new_data_file = os.path.join(prev_pred_data_path, "pred_data.pt")
        if not os.path.exists(new_data_file):
            raise FileNotFoundError(
                f"Prediction data for cycle {cycle_num - 1} not found: {new_data_file}"
            )
        new_data = torch.load(new_data_file, weights_only=False)

        # Combine incrementally
        combined_data = connect_graphs_preserve_weights(
            combined_data, new_data, similarity_metric, top_k, similarity_threshold
        )

    # Save the result to the current cycle's output directory
    output_file = os.path.join(combined_output_path, "combined_data.pt")
    torch.save(combined_data, output_file)
    print(f"Saved combined data for cycle {cycle_num} to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine data for a given cycle."
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Current cycle number"
    )
    args = parser.parse_args()
    cycle_num = args.cycle

    params = load_params()
    SIMILARITY_METRIC = params["combine_data"]["similarity_metric"]
    TOP_K = params["combine_data"]["top_k"]
    SIMILARITY_THRESHOLD = params["combine_data"]["similarity_threshold"]

    # Define DVC-managed paths
    base_data_path = f"results/data/base/cycle_{cycle_num}"
    combined_output_path = f"results/data/combined/cycle_{cycle_num}"
    # Define dependency paths from the *previous* cycle
    prev_pred_data_path = f"results/data/prediction/cycle_{cycle_num - 1}"
    prev_combined_data_path = f"results/data/combined/cycle_{cycle_num - 1}"

    prepare_combined_data(
        cycle_num,
        base_data_path,
        prev_pred_data_path,
        prev_combined_data_path,
        combined_output_path,
        SIMILARITY_METRIC,
        TOP_K,
        SIMILARITY_THRESHOLD,
    )


if __name__ == "__main__":
    main()
