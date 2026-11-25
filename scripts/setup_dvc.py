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

    # Bootstrap stage is pure data generation. No cleanup/reset embedded in the
    # command: doing so would invalidate bootstrap's DVC cache whenever the
    # number of cycles changes. Use `make reset` to deliberately wipe results.
    bootstrap_cmd = "python src/stages/generate_base_data.py --cycle 0"

    raw_helpers = ["src/utilities/raw_graphs.py", "src/utilities/preprocessing.py"]
    bundle_helpers = ["src/utilities/model_bundle.py", *raw_helpers]
    model_schema = [
        "data.base.n_features",
        "data.base.n_classes",
        "data.base.class_names",
    ]
    edge_settings = [
        "add_self_loops",
        "data.make_edge_weight",
        "data.make_edge_weight_method",
    ]
    stages = {
        "stages": {
            "bootstrap": {
                "cmd": bootstrap_cmd,
                "deps": [
                    "src/stages/generate_base_data.py",
                    "src/utilities/data_utils.py",
                    "src/utilities/general_utils.py",
                    *raw_helpers,
                ],
                "params": ["data.base", *edge_settings],
                "outs": [
                    "results/data/base/cycle_0/base_data.pt",
                    "results/data/base/cycle_0/fold_data.pt",
                    "results/data/base/cycle_0/test_data.pt",
                    "results/data/base/cycle_0/calib_data.pt",
                    "results/data/base/cycle_0/hotspots.npy",
                    "results/data/base/cycle_0/prototypes.npy",
                ],
            }
        }
    }
    # Generate cycle stages
    for cycle in cycles:
        base = f"results/data/base/cycle_{cycle - 1}"
        trained = f"results/trained/cycle_{cycle}"
        evaluation = f"results/evaluation/cycle_{cycle}"
        cycle_partitions = [
            f"{base}/{name}.pt"
            for name in ("base_data", "fold_data", "test_data", "calib_data")
        ]
        model_bundle = [f"{trained}/model.pt", f"{trained}/preprocessing.json"]
        stages["stages"][f"train_cycle_{cycle}"] = {
            "cmd": f"python src/stages/train.py --cycle {cycle} --model {model}",
            "deps": [
                "src/stages/train.py",
                "src/utilities/train_utils.py",
                "src/utilities/general_utils.py",
                *bundle_helpers,
                *cycle_partitions,
                "src/models/",
            ],
            "params": [
                f"models.{model}",
                "add_self_loops",
                "train",
                *model_schema,
                "data.scaler_type",
                "data.scaler_params",
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
                "src/utilities/calibration_utils.py",
                *bundle_helpers,
                *cycle_partitions,
                "src/models/",
                *model_bundle,
            ],
            "params": [
                f"models.{model}",
                "evaluate",
                "add_self_loops",
                *model_schema,
            ],
            "outs": [
                f"results/evaluation/cycle_{cycle}",
            ],
        }
        stages["stages"][f"generate_pred_data_cycle_{cycle}"] = {
            "cmd": f"python src/stages/generate_pred_data.py --cycle {cycle}",
            "deps": [
                "src/stages/generate_pred_data.py",
                "src/utilities/data_utils.py",
                "src/utilities/general_utils.py",
                *raw_helpers,
                f"{base}/base_data.pt",
                "results/data/base/cycle_0/hotspots.npy",
                "results/data/base/cycle_0/prototypes.npy",
            ],
            "params": [
                "data.pred",
                "data.base",
                *edge_settings,
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
                "src/utilities/calibration_utils.py",
                "src/utilities/eval_utils.py",
                *bundle_helpers,
                *model_bundle,
                f"{evaluation}/calibrator.pt",
                f"{evaluation}/calibration_binding.json",
                "src/models/",
            ],
            "params": [
                f"models.{model}",
                "add_self_loops",
                *model_schema,
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
                *bundle_helpers,
                *model_bundle,
                f"{base}/base_data.pt",
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
            ],
            "params": [
                "analyze_drift",
                *model_schema,
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
                *raw_helpers,
                f"results/data/base/cycle_{cycle - 1}/base_data.pt"
                if cycle > 1
                else "results/data/base/cycle_0/base_data.pt",
                f"results/data/prediction/cycle_{cycle}/pred_data.pt",
            ],
            "params": ["data.base", *edge_settings],
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
