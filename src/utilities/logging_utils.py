"""Utilities for logging and measuring execution time.

This module provides:
- LogTime: A context manager for measuring and logging execution time of code blocks.
"""

import time


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
