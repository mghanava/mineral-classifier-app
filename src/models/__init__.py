"""Models package for mineral deposit prediction.

This package provides model factory functions and imports GAT and GCN model classes.
"""

import yaml

from .gat import MineralDepositGAT
from .gcn import MineralDepositGCN


def get_model(model_name: str, model_params):
    """Create a mineral deposit prediction model instance.

    Parameters
    ----------
    model_name : str
        The name of the model type ('gcn' or 'gat').
    model_params : dict
        Dictionary of model parameters.

    Returns
    -------
    MineralDepositGCN or MineralDepositGAT
        An instance of the specified model.

    Raises
    ------
    ValueError
        If the model_name is not recognized.

    """
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # dynamic data depending on data structure
    data_info = {
        "n_classes": params["data"]["base"]["n_classes"],
        "in_channels": params["data"]["base"]["n_features"],
    }
    if model_name == "gcn":
        return MineralDepositGCN(**model_params, **data_info)
    elif model_name == "gat":
        model_params = {**model_params, **data_info}
        return MineralDepositGAT(**model_params)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
