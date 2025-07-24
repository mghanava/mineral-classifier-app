import yaml

from .gat import MineralDepositGAT
from .gcn import MineralDepositGCN


def get_model(model_name: str, model_params):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # dynamic data depending on data structure
    data_info = {
        "n_classes": params["data"]["base"]["n_classes"],
        "in_channels": params["data"]["base"]["n_features"],
    }
    """Factory function to create a model instance based on type."""
    if model_name == "gcn":
        return MineralDepositGCN(**model_params, **data_info)
    elif model_name == "gat":
        model_params = {**model_params, **data_info}
        return MineralDepositGAT(**model_params)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
