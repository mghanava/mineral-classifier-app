import argparse
import os
from pathlib import Path

import torch
import yaml

from src.utilities.data_utils import connect_graphs_preserve_weights


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
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",  # Previous cycle's base data
        "previous_combined": f"results/data/combined/cycle_{cycle_num - 1}",  # Previous cycle's combined data
        "prediction": f"results/data/prediction/cycle_{cycle_num}",
        "output": f"results/data/combined/cycle_{cycle_num}",
    }


def prepare_combined_data_with_reservoir(
    cycle_num: int,
    paths: dict,
    params: dict,
):
    """Prepare combined data using reservoir sampling for scalability."""
    print(f"🔄 Preparing combined data for cycle {cycle_num} with reservoir sampling")

    combine_params = params.get("combine_data", {})
    reservoir_config = combine_params.get("reservoir", {})

    # Determine data source
    if cycle_num == 1:
        # First cycle: use base data
        base_data_path = os.path.join(paths["base_data"], "base_data.pt")
        if not os.path.exists(base_data_path):
            raise FileNotFoundError(f"Base data not found: {base_data_path}")
        print(f"📥 Loading base data from: {base_data_path}")
        base_data = torch.load(base_data_path, weights_only=False)
    else:
        # Subsequent cycles: use combined data from previous cycle
        combined_data_path = os.path.join(
            paths["previous_combined"], "combined_data.pt"
        )
        if os.path.exists(combined_data_path):
            print(f"📥 Loading previous combined data from: {combined_data_path}")
            base_data = torch.load(combined_data_path, weights_only=False)
        else:
            # Fallback to base data if combined data doesn't exist
            print("⚠️  Previous combined data not found, falling back to base data")
            base_data_path = os.path.join(paths["base_data"], "base_data.pt")
            base_data = torch.load(base_data_path, weights_only=False)

    # Load prediction data
    pred_data_path = os.path.join(paths["prediction"], "pred_data.pt")
    if not os.path.exists(pred_data_path):
        raise FileNotFoundError(f"Prediction data not found: {pred_data_path}")

    print(f"📥 Loading prediction data from: {pred_data_path}")
    pred_data = torch.load(pred_data_path, weights_only=False)

    print("📊 Data sizes before combination:")
    print(f"   Base/Previous: {base_data.x.shape[0]} samples")
    print(f"   Prediction: {pred_data.x.shape[0]} samples")
    print(f"   Total: {base_data.x.shape[0] + pred_data.x.shape[0]} samples")

    # First: Connect the original graphs properly using spatial relationships
    similarity_metric = combine_params.get("similarity_metric", "cosine")
    top_k = combine_params.get("top_k", 10)
    similarity_threshold = combine_params.get("similarity_threshold")

    print("🔗 Connecting graphs using spatial relationships...")
    print(f"   Similarity metric: {similarity_metric}")
    print(f"   Top-k connections: {top_k}")

    # Connect base_data and pred_data BEFORE sampling
    fully_connected_data = connect_graphs_preserve_weights(
        base_data,
        pred_data,
        similarity_metric=similarity_metric,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    print(
        f"📊 Fully connected graph: {fully_connected_data.x.shape[0]} nodes, {fully_connected_data.edge_index.shape[1]} edges"
    )

    # Then: Apply reservoir sampling to the connected graph
    print("🎯 Applying reservoir sampling to connected graph...")
    if reservoir_config.get("enabled", False):
        print(
            f"   Strategy: {reservoir_config.get('sampling_strategy', 'stratified_recent')}"
        )
        print(f"   Max size: {reservoir_config.get('max_combined_size', 3000)}")
        print(f"   Recent ratio: {reservoir_config.get('preserve_recent_ratio', 0.4)}")

        # Apply sampling to the fully connected graph
        from src.utilities.reservoir_sampling import ReservoirSampler

        sampler = ReservoirSampler(
            max_size=reservoir_config.get("max_combined_size", 3000),
            strategy=reservoir_config.get("sampling_strategy", "stratified_recent"),
            preserve_recent_ratio=reservoir_config.get("preserve_recent_ratio", 0.4),
            class_balance_weight=reservoir_config.get("class_balance_weight", 0.8),
            quality_threshold=reservoir_config.get("quality_threshold"),
        )

        # Create a dummy "prediction" dataset for the sampler API compatibility
        # Since we already connected the graphs, we'll sample from the combined graph
        n_base = base_data.x.shape[0]
        base_indices = list(range(n_base))
        pred_indices = list(range(n_base, fully_connected_data.x.shape[0]))

        final_combined_data = sampler._extract_subgraph_smart(
            fully_connected_data, base_indices, pred_indices, cycle_num
        )
        sampling_stats = sampler.get_sample_stats(final_combined_data)
    else:
        print("   Reservoir sampling disabled - using all connected data")
        final_combined_data = fully_connected_data
        from src.utilities.reservoir_sampling import ReservoirSampler

        sampler = ReservoirSampler()
        sampling_stats = sampler.get_sample_stats(fully_connected_data)

    # Display final statistics
    print("📈 Final combined data statistics:")
    print(f"   Samples: {sampling_stats['total_samples']}")
    print(f"   Classes: {sampling_stats['n_classes']}")
    print(f"   Class balance: {sampling_stats['class_balance']:.3f}")
    print(f"   Edges: {sampling_stats['n_edges']}")

    # Memory efficiency report
    original_size = base_data.x.shape[0] + pred_data.x.shape[0]
    final_size = sampling_stats["total_samples"]
    memory_reduction = (
        (1 - final_size / original_size) * 100 if original_size > 0 else 0
    )

    if memory_reduction > 0:
        print(f"💾 Memory efficiency: {memory_reduction:.1f}% reduction")
    else:
        print("💾 Memory usage: No reduction (all data retained)")

    return final_combined_data, sampling_stats


def main():
    """Combine data with reservoir sampling optimizations."""
    parser = argparse.ArgumentParser(
        description="Combine data with scalability optimizations."
    )
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")

    # Get cycle-specific paths and ensure directories exist
    paths = get_cycle_paths(cycle_num)
    output_path = ensure_directory_exists(paths["output"])

    # Load parameters
    params = load_params()

    print("=" * 60)
    print(f"🔄 OPTIMIZED DATA COMBINATION - Cycle {cycle_num}")
    print("=" * 60)
    print(f"📂 Output path: {output_path}")

    try:
        # Prepare combined data with reservoir sampling
        combined_data, stats = prepare_combined_data_with_reservoir(
            cycle_num=cycle_num, paths=paths, params=params
        )

        # Save combined data
        combined_data_path = os.path.join(output_path, "combined_data.pt")
        print(f"\n💾 Saving combined data to: {combined_data_path}")
        torch.save(combined_data, combined_data_path)

        # Save statistics
        stats_path = os.path.join(output_path, "sampling_stats.yaml")
        print(f"📊 Saving statistics to: {stats_path}")
        with open(stats_path, "w") as f:
            yaml.dump(
                {
                    "cycle": cycle_num,
                    "sampling_stats": stats,
                    "reservoir_config": params.get("combine_data", {}).get(
                        "reservoir", {}
                    ),
                },
                f,
            )

        print(f"\n✅ Data combination complete for cycle {cycle_num}!")
        print(f"📁 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during data combination: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
