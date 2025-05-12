import yaml

from .gat import MineralDepositGAT
from .gcn import MineralDepositGCN


def get_model(model_name: str, model_params):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    """Factory function to create a model instance based on type."""
    if model_name == "gcn":
        data_info = {
            "n_classes": params["data"]["n_classes"],
            "in_channels": params["data"]["n_features"],
        }
        return MineralDepositGCN(**model_params, **data_info)
    elif model_name == "gat":
        data_info = {
            "n_classes": params["data"]["n_classes"],
            "in_channels": params["data"]["n_features"],
            "edge_dim": params["data"]["edge_dim"],
        }
        model_params = {**model_params, **data_info}
        return MineralDepositGAT(**model_params)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
