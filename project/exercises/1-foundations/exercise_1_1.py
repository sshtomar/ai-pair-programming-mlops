"""
Exercise 1.1: Welcome & Setup
Difficulty: ★☆☆
Topic: Environment verification and Python imports

Instructions:
This exercise has two parts:

PART A - Guided Stub (check_environment):
Write a function that verifies your ML development environment is properly set up.
The function should:
1. Check that Python version is >= 3.10
2. Verify required packages are installed
3. Return a dict mapping package names to their installed versions

PART B - Debug Exercise (fix_circular_import):
The module below has a circular import issue. Identify and fix it.

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import sys
from typing import Any


# =============================================================================
# PART A: Guided Stub - Environment Checker
# =============================================================================

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "pytest",
]


def check_environment() -> dict[str, str]:
    """
    Verify the ML development environment is properly configured.

    Returns:
        dict mapping package name to version string.

    Raises:
        RuntimeError: If Python version is < 3.10
        ImportError: If a required package is not installed

    Example:
        >>> result = check_environment()
        >>> "numpy" in result
        True
        >>> result["numpy"]  # e.g., "1.24.0"
    """
    # TODO: Step 1 - Check Python version
    # Get the current Python version using sys.version_info
    # Raise RuntimeError if major < 3 or (major == 3 and minor < 10)

    # TODO: Step 2 - Build package version dictionary
    # For each package in REQUIRED_PACKAGES:
    #   - Try to import the package using importlib
    #   - Get its __version__ attribute
    #   - Add to result dict
    # Raise ImportError if package not found

    # TODO: Step 3 - Return the dictionary

    pass  # Remove this when you implement


# =============================================================================
# PART B: Debug Exercise - Circular Import
# =============================================================================

# The code below simulates a circular import problem.
# In real projects, this would be split across files, but we simulate it here.
#
# SCENARIO: We have a "metrics" module that imports "model" to get predictions,
# and a "model" module that imports "metrics" to evaluate itself.
#
# BUG: The current structure causes an ImportError at module load time.
# FIX: Restructure to break the circular dependency.

# --- Simulated metrics.py ---
class MetricsCalculator:
    """Calculates model performance metrics."""

    def __init__(self):
        # BUG: This import happens at class instantiation, but the real
        # problem is that in a real scenario, this would be a top-level import
        # that creates a cycle: metrics -> model -> metrics
        from exercise_1_1_model import ModelPredictor  # noqa: F401
        self.predictor = None  # Would be ModelPredictor()

    def calculate_accuracy(self, y_true: list[int], y_pred: list[int]) -> float:
        """Calculate accuracy score."""
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between y_true and y_pred")
        if len(y_true) == 0:
            raise ValueError("Empty input lists")
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)


def fix_circular_import() -> type:
    """
    Return a fixed version of MetricsCalculator that doesn't have
    circular import issues.

    The fix should:
    1. Not import ModelPredictor at class instantiation
    2. Use lazy loading or dependency injection instead

    Returns:
        A class that calculates metrics without circular imports
    """
    # TODO: Create a fixed version of MetricsCalculator
    # Options to consider:
    # - Lazy import (import inside method that needs it)
    # - Dependency injection (pass predictor as parameter)
    # - Remove the dependency entirely if not needed

    class FixedMetricsCalculator:
        """Fixed version without circular import issues."""

        def __init__(self, predictor: Any = None):
            # TODO: Implement a clean initialization
            pass

        def calculate_accuracy(self, y_true: list[int], y_pred: list[int]) -> float:
            # TODO: Copy the working logic from MetricsCalculator
            pass

    return FixedMetricsCalculator


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
/hint 1 - check_environment:
Use importlib.import_module() to dynamically import packages.
Example: module = importlib.import_module("numpy")

/hint 2 - check_environment:
Most packages store their version in __version__ attribute.
Some use VERSION or version. Try __version__ first.
sys.version_info gives you (major, minor, micro, ...) tuple.

/hint 3 - check_environment:
Complete solution structure:
    import importlib

    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, got {sys.version}")

    versions = {}
    for pkg in REQUIRED_PACKAGES:
        module = importlib.import_module(pkg)
        versions[pkg] = getattr(module, "__version__", "unknown")
    return versions

/hint 1 - fix_circular_import:
The problem is importing at module/class level. Move imports to where
they're actually needed (inside methods).

/hint 2 - fix_circular_import:
Better yet: use dependency injection. Pass the predictor into __init__
instead of creating it inside the class.

/hint 3 - fix_circular_import:
class FixedMetricsCalculator:
    def __init__(self, predictor=None):
        self.predictor = predictor  # Injected, not imported

    def calculate_accuracy(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch")
        if len(y_true) == 0:
            raise ValueError("Empty input")
        return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
"""
