"""Analyze and visualize cycle performance metrics.

This module loads cycle performance data and generates analysis plots.
"""

import argparse
from pathlib import Path

from src.utilities.cycles_performance_utils import (
    load_metrics_from_cycles,
    plot_metrics,
)
from src.utilities.general_utils import ensure_directory_exists, load_params


def main() -> None:
    """Load cycle performance data and generate analysis plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, help="Number of cycles to aggregate")
    args = parser.parse_args()
    cycles = args.cycles if args.cycles is not None else load_params()["cycles"]

    # Setup paths
    results_dir = Path("results")
    output_dir = results_dir / "cycles_performance_analysis"
    ensure_directory_exists(output_dir)

    # Load and plot metrics
    cycles_data = load_metrics_from_cycles(results_dir, range(1, cycles + 1))
    output_file = plot_metrics(cycles_data, output_dir)

    print(
        f"\nCycle performance analysis completed. Results saved as '{output_file}'.\n"
    )


if __name__ == "__main__":
    main()
