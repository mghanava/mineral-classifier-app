import yaml


def parse_range(range_str):
    start, end = map(int, range_str.split("-"))
    return list(range(start, end + 1))


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def generate_dvc_yaml():
    params = load_params()

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

    for cycle in parse_range(params["cycles"]):
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
            ],
            "outs": [
                f"results/data/base/cycle_{cycle}",
                f"results/data/prediction/cycle_{cycle}",
                f"results/trained/cycle_{cycle}",
                f"results/evaluation/cycle_{cycle}",
                f"results/drift_analysis/cycle_{cycle}",
            ],
        }

    with open("dvc.yaml", "w") as f:
        yaml.dump(stages, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    generate_dvc_yaml()
