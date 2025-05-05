from yaml import safe_load

from .gat import MineralDepositGAT
from .gnn import MineralDepositGCN


def get_model(model_name: str, model_params):
    with open("params.yaml") as f:
        params = safe_load(f)
    data_info = {
        "n_classes": params["data"]["n_classes"],
        "in_channels": params["data"]["n_features"],
        "edge_dim": params["data"]["edge_dim"],
    }
    """Factory function to create a model instance based on type."""
    if model_name == "gcn":
        return MineralDepositGCN(**model_params)
    elif model_name == "gat":
        model_params = {**model_params, **data_info}
        return MineralDepositGAT(**model_params)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
