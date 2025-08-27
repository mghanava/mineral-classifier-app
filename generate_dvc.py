import yaml


def parse_range(range_str):
    start, end = map(int, range_str.split("-"))
    return list(range(start, end + 1))


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def generate_performance_analysis_stage(cycles):
    """Generate the performance analysis stage configuration"""
    metrics_files = []
    for cycle in cycles:
        metrics_files.extend(
            [
                f"results/evaluation/cycle_{cycle}/metrics.json",
                f"results/prediction/cycle_{cycle}/metrics.json",
            ]
        )

    return {
        "analyze_performance": {
            "cmd": "python src/stages/analyze_cycle_performance.py",
            "deps": ["src/stages/analyze_cycle_performance.py", *metrics_files],
            "outs": ["results/performance_analysis/cycle_performance.png"],
        }
    }


def generate_dvc_yaml():
    params = load_params()
    cycles = parse_range(params["cycles"])

    stages = {
        "stages": {
            "bootstrap": {
                "cmd": "python src/stages/generate_base_data.py --cycle 0",
                "deps": ["src/stages/generate_base_data.py"],
                "params": ["data.base"],
                "outs": ["results/data/base/cycle_0"],
            }
        }
    }

    # Generate cycle stages
    for cycle in cycles:
        stages["stages"][f"cycle_{cycle}"] = {
            "cmd": f"python src/run_cycle.py --cycle {cycle} --model {params['default_model']}",
            "deps": [
                "src/run_cycle.py",
                "src/stages/",
                "src/models/",
                "src/utilities/",
                f"results/data/base/cycle_{cycle - 1}"
                if cycle > 1
                else "results/data/base/cycle_0",
            ],
            "params": [
                f"models.{params['default_model']}",
                "data",
                "train",
                "evaluate",
                "analyze_drift",
            ],
            "outs": [
                f"results/data/base/cycle_{cycle}",
                f"results/data/prediction/cycle_{cycle}",
                f"results/trained/cycle_{cycle}",
                f"results/evaluation/cycle_{cycle}",
                f"results/drift_analysis/cycle_{cycle}",
            ],
        }

    # Add performance analysis stage
    stages["stages"].update(generate_performance_analysis_stage(cycles))

    with open("dvc.yaml", "w") as f:
        yaml.dump(stages, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    generate_dvc_yaml()
