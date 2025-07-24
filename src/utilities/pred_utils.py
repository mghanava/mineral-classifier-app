import os

import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight
from torch_geometric.data import Data

from src.utilities.calibration_utils import CalibrationPipeline
from src.utilities.eval_utils import plot_confusion_matrix


def prediction(
    pred_data: Data,
    model: torch.nn.Module,
    calibrator_path: str,
    class_names: list,
    cycle_num: int,
    save_path: str | None = None,
    device: torch.device = torch.device("cpu"),
):
    # hide the labels from model
    hidden_labels = pred_data.y
    pred_data.y = None
    calibrated_probs = CalibrationPipeline.load(
        filepath=calibrator_path,
        base_model=model,
        device=device,
    ).predict(pred_data)
    model.eval()
    with torch.no_grad():
        model.to(device)
        data = pred_data.to(str(device))
        logits = model(data)
        uncalibrated_probs = F.softmax(logits, dim=1)
    # Create a DataFrame with one column per class probability
    calib_prob_array = calibrated_probs.cpu().numpy()
    num_classes = calib_prob_array.shape[1]
    calib_prob_columns = {
        f"calib_prob_class_{i}": calib_prob_array[:, i] for i in range(num_classes)
    }
    uncalib_prob_array = uncalibrated_probs.cpu().numpy()
    uncalib_prob_columns = {
        f"uncalib_prob_class_{i}": uncalib_prob_array[:, i] for i in range(num_classes)
    }
    prob_columns = {**calib_prob_columns, **uncalib_prob_columns}
    calib_entropies = -torch.sum(
        calibrated_probs * torch.log(calibrated_probs + 1e-8), dim=1
    )
    uncalib_entropies = -torch.sum(
        uncalibrated_probs * torch.log(uncalibrated_probs + 1e-8), dim=1
    )
    # Create a DataFrame with the results
    true_label = hidden_labels.cpu().numpy()
    cal_pred_labels = calib_prob_array.argmax(axis=1)

    result_df = pd.DataFrame(
        {
            **prob_columns,
            "predicted_label": cal_pred_labels,
            "true_label": true_label
            if isinstance(hidden_labels, torch.Tensor)
            else int(hidden_labels)
            if hidden_labels is not None
            else None,
            "calibrated_entropy": calib_entropies.cpu().numpy(),
            "uncalibrated_entropy": uncalib_entropies.cpu().numpy(),
        }
    )
    fig = result_df.hist(figsize=(20, 10))
    if save_path is not None:
        result_df.to_csv(
            os.path.join(save_path, f"predictions_cycle_{cycle_num}.csv"), index=False
        )
        pyplot.savefig(os.path.join(save_path, "histograms_cycle_{cycle_num}.png"))
    pyplot.close()
    sample_weights = (
        compute_sample_weight("balanced", true_label)
        if true_label is not None
        else None
    )
    mcc = matthews_corrcoef(true_label, cal_pred_labels, sample_weight=sample_weights)
    if save_path is not None:
        plot_confusion_matrix(
            true_label,
            cal_pred_labels,
            class_names,
            title=f"Matthews correlation coefficient {mcc:.3f}",
            save_path=os.path.join(
                save_path, f"confussion_matrix_cycle_{cycle_num}.png"
            ),
        )

    return fig
