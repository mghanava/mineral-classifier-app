import time
from pathlib import Path

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


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

