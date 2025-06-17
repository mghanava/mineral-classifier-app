"""Utility module providing early stopping functionality for training loops.

This module provides:
- EarlyStopping: A class that implements early stopping to prevent overfitting
  by monitoring a metric and stopping training when it stops improving.
"""


class EarlyStopping:
    """Implements early stopping to terminate training when a monitored metric stops improving.

    Attributes
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    best_loss : float
        Best score seen so far.
    epochs_without_improvement : int
        Number of epochs since last improvement.
    early_stop : bool
        Whether early stopping has been triggered.
    decreasing_score : bool
        Whether a lower score is considered better.

    Methods
    -------
    __call__(loss_val)
        Call to update early stopping state with new loss value.
    reset_counter()
        Reset the counter for epochs without improvement.
    best_score
        Property to get the best score seen so far.

    """

    def __init__(self, patience=50, min_delta=0.001, decreasing_score: bool = True):
        """Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as an improvement.
            decreasing_score (bool): Whether a lower score is considered better.

        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf") if decreasing_score else -float("inf")
        self.epochs_without_improvement = 0
        self.early_stop = False
        self.decreasing_score = decreasing_score

    def __call__(self, loss_val):
        """Update early stopping state with new loss value.

        Args:
            loss_val (float): The current loss value to evaluate for early stopping.

        """
        stop_check_condition = (
            loss_val < self.best_loss - self.min_delta
            if self.decreasing_score
            else loss_val > self.best_loss + self.min_delta
        )
        if stop_check_condition:
            self.best_loss = loss_val
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.early_stop = True

    def reset_counter(self):
        """Reset the counter when learning rate changes."""
        self.epochs_without_improvement = 0

    @property
    def best_score(self):
        """Return the best score seen so far."""
        return self.best_loss
