"""
Helper module for exercise_1_1.py circular import demonstration.

This simulates what would happen if model.py imported metrics.py
and metrics.py imported model.py.
"""

# In a real circular import scenario, this line would cause the problem:
# from exercise_1_1 import MetricsCalculator


class ModelPredictor:
    """A simple model predictor for demonstration."""

    def __init__(self):
        self.is_trained = False

    def predict(self, X: list) -> list[int]:
        """Return dummy predictions."""
        return [0] * len(X)
