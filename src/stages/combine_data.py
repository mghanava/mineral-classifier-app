import os
from typing import Literal

import torch
import yaml

from src.utilities.data_utils import connect_graphs_preserve_weights

dataset_path = "results/data/base"
prediction_data_path = "results/data/prediction"
combined_data_path = "results/data/combined"


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


# Define the allowed similarity metrics
AllowedSimilarityMetrics = Literal["cosine", "euclidean", "dot"]


def prepare_combined_data(
    cycle_num: int,
    dataset_path: str,
    prediction_data_path: str,
    combined_data_path: str,
    similarity_metric: AllowedSimilarityMetrics,
    top_k: int,
    similarity_threshold: float,
):
    """Prepare combined data for a given cycle in a DVC pipeline.

    For cycle N:
    - If N=1: Only `base_data` is used.
    - If N>1: `combined_data_cycle_{N-1}.pt` + `pred_data_cycle_{N-1}.pt` are combined.

    """
    # Validate similarity_metric (if coming from an untrusted source, like CLI/API)
    if similarity_metric not in {"cosine", "euclidean", "dot"}:
        raise ValueError(
            f"Invalid similarity_metric: {similarity_metric}. "
            "Must be one of: 'cosine', 'euclidean', 'dot'"
        )
    # Ensure directories exist (DVC might not create them automatically)
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(prediction_data_path, exist_ok=True)
    os.makedirs(combined_data_path, exist_ok=True)

    # Base case: cycle_num = 1 → only load base_data
    if cycle_num == 1:
        combined_data = torch.load(
            os.path.join(dataset_path, f"base_data_cycle_{cycle_num}.pt"),
            weights_only=False,
        )
    else:
        # Load previous combined data (from cycle N-1)
        prev_combined_file = os.path.join(
            combined_data_path, f"combined_data_cycle_{cycle_num - 1}.pt"
        )
        if not os.path.exists(prev_combined_file):
            raise FileNotFoundError(
                f"Previous combined data not found: {prev_combined_file}"
            )
        combined_data = torch.load(prev_combined_file, weights_only=False)

        # Load new pred_data (from cycle N-1)
        new_data_file = os.path.join(
            prediction_data_path, f"pred_data_cycle_{cycle_num - 1}.pt"
        )
        if not os.path.exists(new_data_file):
            raise FileNotFoundError(
                f"Prediction data for cycle {cycle_num - 1} not found: {new_data_file}"
            )
        new_data = torch.load(new_data_file, weights_only=False)

        # Combine incrementally
        combined_data = connect_graphs_preserve_weights(
            combined_data, new_data, similarity_metric, top_k, similarity_threshold
        )

    # Save for future cycles (DVC tracks this)
    torch.save(
        combined_data,
        os.path.join(combined_data_path, f"combined_data_cycle_{cycle_num}.pt"),
    )


def main():
    params = load_params()
    CYCLE_NUM = params["cycle"]
    SIMILARITY_METRIC = params["combine_data"]["similarity_metric"]
    TOP_K = params["combine_data"]["top_k"]
    SIMILARITY_THRESHOLD = params["combine_data"]["similarity_threshold"]

    prepare_combined_data(
        CYCLE_NUM,
        dataset_path,
        prediction_data_path,
        combined_data_path,
        SIMILARITY_METRIC,
        TOP_K,
        SIMILARITY_THRESHOLD,
    )


if __name__ == "__main__":
    main()
