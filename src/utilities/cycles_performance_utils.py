import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def load_metrics_from_cycles(results_dir: Path):
    """Load metrics from evaluation and prediction directories."""
    cycles_data = {"evaluation": [], "prediction": []}

    for stage in ["evaluation", "prediction"]:
        stage_dir = results_dir / stage

        # Find all cycle directories
        cycle_dirs = sorted(
            [d for d in stage_dir.glob("cycle_*") if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1]),
        )

        for cycle_dir in cycle_dirs:
            cycle_num = int(cycle_dir.name.split("_")[1])
            metrics_file = cycle_dir / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = (
                        json.load(f)
                        if stage == "prediction"
                        else json.load(f)["calibrated"]
                    )

                cycles_data[stage].append(
                    {
                        "cycle": cycle_num,
                        "accuracy": metrics.get("acc", None),
                        "f1": metrics.get("f1", None),
                        "mcc": metrics.get("mcc", None),
                    }
                )

    return cycles_data


def plot_metrics(cycles_data, output_dir: Path):
    """Create subplot visualization for evaluation and prediction metrics."""
    try:
        sns.set_style("whitegrid")
    except ImportError:
        plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Define colors for consistent plotting
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for idx, (stage, ax) in enumerate([("evaluation", ax1), ("prediction", ax2)]):
        df = pd.DataFrame(cycles_data[stage])
        if not df.empty:
            for i, column in enumerate(["accuracy", "f1", "mcc"]):
                df.plot(
                    x="cycle",
                    y=column,
                    marker="o",
                    ax=ax,
                    linewidth=2,
                    color=colors[i],
                    label=column.upper(),
                )
            ax.set_title(f"{stage.capitalize()} Metrics Across Cycles")
            ax.set_xlabel("Cycle Number")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)
            # Set x-axis to show integer ticks
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # Ensure x-axis limits are slightly padded
            x_min, x_max = df["cycle"].min(), df["cycle"].max()
            ax.set_xlim(x_min - 0.2, x_max + 0.2)

    plt.tight_layout()
    output_file = output_dir / "cycles_performance.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()

    return output_file
