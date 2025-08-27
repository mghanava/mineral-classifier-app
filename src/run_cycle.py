import argparse
import subprocess
import sys
from pathlib import Path

from src.utilities.logging_utils import LogTime


def run_stage(stage_name, cycle, model=None):
    """Run a stage with proper arguments."""
    cmd = ["python", f"src/stages/{stage_name}.py", "--cycle", str(cycle)]
    if model:
        cmd.extend(["--model", model])

    print(f"Running {stage_name} for cycle {cycle}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error in {stage_name} for cycle {cycle}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    # Verify previous cycle data exists (except for cycle 1)
    if args.cycle > 1:
        prev_cycle_dir = Path(f"results/data/base/cycle_{args.cycle - 1}")
        if not prev_cycle_dir.exists():
            print(f"Error: Previous cycle data not found at {prev_cycle_dir}")
            sys.exit(1)

    # Run all stages in sequence
    stages = [
        ("train", True),  # stage_name, needs_model
        ("evaluate", True),
        ("generate_pred_data", False),
        ("predict", True),
        ("analyze_drift", False),
        ("prepare_next_cycle_data", False),
    ]
    for stage_name, needs_model in stages:
        run_stage(stage_name, args.cycle, args.model if needs_model else None)


if __name__ == "__main__":
    with LogTime(task_name="\nFull cycles"):
        main()
