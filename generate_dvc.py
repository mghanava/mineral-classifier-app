import yaml


def parse_range(range_str):
    start, end = map(int, range_str.split("-"))
    return list(range(start, end + 1))


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def generate_performance_analysis_stage(cycles):
    """Generate the performance analysis stage configuration."""
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


def generate_fine_grained_dvc_yaml():
    """Generate fine-grained DVC pipeline with individual stages."""
    params = load_params()
    cycles = parse_range(params["cycles"])
    model = params["default_model"]

    stages = {
        "stages": {
            "bootstrap": {
                "cmd": "python src/stages/generate_base_data.py --cycle 0",
                "deps": [
                    "src/stages/generate_base_data.py",
                    "src/utilities/data_utils.py",
                ],
                "params": ["data.base"],
                "outs": [
                    "results/data/base/cycle_0/base_data.pt",
                    "results/data/base/cycle_0/fold_data.pt",
                    "results/data/base/cycle_0/test_data.pt",
                ],
            }
        }
    }

    # Generate cycle stages
    for cycle in cycles:
        stages["stages"][f"train_cycle_{cycle}"] = {
            "cmd": f"python src/stages/train.py --cycle {cycle} --model {model}",
            "deps": [
                "src/stages/train.py",
                "src/utilities/train_utils.py",
                "src/utilities/logging_utils.py",
                f"results/data/base/cycle_{cycle - 1}/fold_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/fold_data.pt",
                "src/models/",
            ],
            "params": [
                f"models.{model}",
                "train",
            ],
            "outs": [
                f"results/trained/cycle_{cycle}",
            ],
        }
        stages["stages"][f"evaluate_cycle_{cycle}"] = {
            "cmd": f"python src/stages/evaluate.py --cycle {cycle} --model {model}",
            "deps": [
                "src/stages/evaluate.py",
                "src/utilities/eval_utils.py",
                "src/utilities/logging_utils.py",
                f"results/data/base/cycle_{cycle - 1}/test_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/test_data.pt",
                "src/models/",
            ],
            "params": [
                f"models.{model}",
                "evaluate",
            ],
            "outs": [
                f"results/evaluation/cycle_{cycle}",
            ],
        }
        stages["stages"][f"generate_pred_data_cycle_{cycle}"] = {
            "cmd": f"python src/stages/generate_pred_data.py --cycle {cycle}",
            "deps": [
                "src/stages/generate_pred_data.py",
                "src/utilities/pred_utils.py",
                "src/utilities/data_utils.py",
                f"results/data/base/cycle_{cycle - 1}/base_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/base_data.pt",  # To avoid overlap
                "src/models/",
            ],
            "params": [
                "data.pred",
            ],
            "outs": [
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
            ],
        }
        stages["stages"][f"predict_cycle_{cycle}"] = {
            "cmd": f"python src/stages/predict.py --cycle {cycle} --model {model}",
            "deps": [
                "src/stages/predict.py",
                "src/utilities/pred_utils.py",
                "src/utilities/logging_utils.py",
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
                f"results/trained/cycle_{cycle}/{model}.pt",
                f"results/evaluation/cycle_{cycle}/calibrator.pt",
                "src/models/",
            ],
            "params": [
                f"models.{model}",
            ],
            "outs": [
                f"results/prediction/cycle_{cycle}",
            ],
        }
        stages["stages"][f"analyze_drift_cycle_{cycle}"] = {
            "cmd": f"python src/stages/analyze_drift.py --cycle {cycle}",
            "deps": [
                "src/stages/analyze_drift.py",
                "src/utilities/drift_detection_utils.py",
                "src/utilities/logging_utils.py",
                f"results/data/base/cycle_{cycle - 1}/base_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/base_data.pt",
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
            ],
            "params": [
                "analyze_drift",
            ],
            "outs": [
                f"results/drift_analysis/cycle_{cycle}",
            ],
        }
        stages["stages"][f"prepare_next_cycle_data_{cycle}"] = {
            "cmd": f"python src/stages/prepare_next_cycle_data.py --cycle {cycle}",
            "deps": [
                "src/stages/prepare_next_cycle_data.py",
                "src/utilities/data_utils.py",
                f"results/data/base/cycle_{cycle - 1}/base_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/base_data.pt",
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
            ],
            "params": [
                "data.base",
            ],
            "outs": [
                f"results/data/base/cycle_{cycle}/base_data.pt",
                f"results/data/base/cycle_{cycle}/fold_data.pt",
                f"results/data/base/cycle_{cycle}/test_data.pt",
            ],
        }
    # Add performance analysis stage
    stages["stages"].update(generate_performance_analysis_stage(cycles))

    # Write to dvc.yaml
    with open("dvc.yaml", "w") as f:
        yaml.dump(stages, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    generate_fine_grained_dvc_yaml()
