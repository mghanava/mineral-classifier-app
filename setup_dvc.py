"""Generate a dvc.yaml file with fine-grained stages for a machine learning pipeline.

It includes stages for data generation, training, evaluation, prediction, drift analysis, and performance analysis.
The pipeline is designed to support iterative training cycles and drift detection.
"""

import yaml

from src.utilities.general_utils import load_params


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
        "analyze_cycles_performance": {
            "cmd": "python src/stages/analyze_cycles_performance.py",
            "deps": [
                "src/stages/analyze_cycles_performance.py",
                "src/utilities/cycles_performance_utils.py",
                *metrics_files,
            ],
            "params": ["cycles"],
            "outs": ["results/cycles_performance_analysis/cycles_performance.png"],
        }
    }


def generate_fine_grained_dvc_yaml():
    """Generate fine-grained DVC pipeline with individual stages."""
    params = load_params()
    cycles = range(params["cycles"] + 1)[1:]  # Skip cycle 0 (bootstrap)
    model = params["default_model"]

    # Add cleanup command to bootstrap stage
    cleanup_cmd = " && ".join(
        [
            "rm -rf results/*",  # Clear all results first
            f"find results -type d -name 'cycle_*' | grep -E 'cycle_[0-9]+$' | awk -F'cycle_' '{{if ($2 > {params['cycles']}) system(\"rm -rf \"$0)}}'",
            "python src/stages/generate_base_data.py --cycle 0",  # Original bootstrap command
        ]
    )

    stages = {
        "stages": {
            "bootstrap": {
                "cmd": cleanup_cmd,
                "deps": [
                    "src/stages/generate_base_data.py",
                    "src/utilities/data_utils.py",
                    "src/utilities/general_utils.py",
                ],
                "params": ["data.base", "add_self_loops"],
                "outs": [
                    "results/data/base/cycle_0/base_data.pt",
                    "results/data/base/cycle_0/fold_data.pt",
                    "results/data/base/cycle_0/test_data.pt",
                    "results/data/base/cycle_0/calib_data.pt",
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
                "src/utilities/general_utils.py",
                f"results/data/base/cycle_{cycle - 1}/fold_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/fold_data.pt",
                "src/models/",
            ],
            "params": [
                f"models.{model}",
                "add_self_loops",
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
                "src/utilities/general_utils.py",
                f"results/data/base/cycle_{cycle - 1}/test_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/test_data.pt",
                "src/models/",
                f"results/trained/cycle_{cycle}/model.pt",
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
                "src/utilities/general_utils.py",
                f"results/data/base/cycle_{cycle - 1}/base_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/base_data.pt",
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
                "src/utilities/general_utils.py",
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
                f"results/trained/cycle_{cycle}/model.pt",
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
                "src/utilities/general_utils.py",
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
                "src/utilities/general_utils.py",
                f"results/data/base/cycle_{cycle - 1}/base_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/base_data.pt",
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
            ],
            "params": ["data.base", "add_self_loops"],
            "outs": [
                f"results/data/base/cycle_{cycle}/base_data.pt",
                f"results/data/base/cycle_{cycle}/fold_data.pt",
                f"results/data/base/cycle_{cycle}/test_data.pt",
                f"results/data/base/cycle_{cycle}/calib_data.pt",
            ],
        }
    # Add performance analysis stage
    if len(cycles) > 1:
        stages["stages"].update(generate_performance_analysis_stage(cycles))

    # Write to dvc.yaml
    with open("dvc.yaml", "w") as f:
        yaml.dump(stages, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    generate_fine_grained_dvc_yaml()
