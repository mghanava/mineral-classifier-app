"""Resolve the compute device for model stages."""

import re

import torch


def select_device(requested: object = "auto") -> torch.device:
    """Resolve auto, cpu, cuda or cuda:N against visible CUDA devices."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    if isinstance(requested, str) and re.fullmatch(r"cuda(?::[0-9]+)?", requested):
        index = int(requested.split(":")[-1]) if ":" in requested else 0
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return torch.device("cuda", index)
        raise ValueError(
            f"Requested device {requested!r} is unavailable. Check CUDA drivers and "
            "visible devices, or set device: cpu (or auto) in params.yaml."
        )
    raise ValueError("Set device in params.yaml to auto, cpu, cuda or cuda:N.")
