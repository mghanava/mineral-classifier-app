"""Utility functions for data handling and execution time measurement.

This module provides:
- LogTime: Context manager for measuring code execution time
- load_params: Load parameters from YAML files
- load_data: Load data from files with error handling
- save_data: Save data to files with optional NumPy format
- ensure_directory_exists: Create directories if they don't exist
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml


class LogTime:
    """A context manager for measuring execution time of code blocks and functions.

    Usage as a context manager:
    ```
    with ExecutionTimer(name="My Task"):
        # code to measure
        time.sleep(1)
    ```

    """

    def __init__(self, task_name: str):
        """Initialize the LogTime context manager with a task name."""
        self.start_time = time.time()
        self.task_name = task_name

    def __enter__(self):
        """Start timing the execution of the code block."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """End timing and print the execution time of the code block."""
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print(f"{self.task_name} executed in {self.execution_time:.3f} seconds.\n")
        return False  # Propagate any exceptions


def load_params(path: str = "params.yaml"):
    """Load parameters from params.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path: str, data_type: str, load_numpy: bool = False):
    """Load data from a given path.

    Args:
        path (str): Path to the data file.
        data_type (str): Description of the data type for logging.
        load_numpy (bool): Whether to load a NumPy file.

    Returns:
        Loaded data object.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{data_type} data not found at {path}!")
    if not load_numpy:
        data = torch.load(path, weights_only=False)
    else:
        with open(path, "rb") as f:
            data = np.load(f)
    print(f"📥 {data_type} data from {path} loaded successfully!")
    return data


def save_data(data, path: str, data_type: str, save_numpy: bool = False):
    """Save data to a given path.

    Args:
        data: Data object to save.
        path (str): Path to save the data file.
        data_type (str): Description of the data type for logging.
        save_numpy (bool): Whether to save a NumPy file.

    """
    if not save_numpy:
        torch.save(data, path)
    else:
        with open(path, "wb") as f:
            np.save(f, data)
    print(f"💾 {data_type} data saved at {path} successfully!")


def load_model_weights(model, path):
    """Load model weights from a given path.

    Args:
        model: The model instance to load weights into.
        path (str): Path to the model weights file.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}!")
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"📥 Model weights from {path} loaded successfully!")


def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
