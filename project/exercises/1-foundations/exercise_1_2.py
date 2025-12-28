"""
Exercise 1.2: MLOps Principles
Difficulty: ★★☆
Topic: Business metrics and configuration management

Instructions:
This exercise has two parts:

PART A - Guided Stub (calculate_business_metric):
In production ML, accuracy isn't everything. A false negative (missing fraud)
might cost $10,000, while a false positive (blocking legitimate transaction)
might cost $50 in customer service time.

Write a function that calculates the total business cost of model errors.

PART B - Debug Exercise (load_config):
The config loading function has hardcoded paths that break in different
environments. Fix it to be environment-agnostic.

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import json
import os
from pathlib import Path


# =============================================================================
# PART A: Guided Stub - Business Metrics Calculator
# =============================================================================

def calculate_business_metric(
    predictions: list[int],
    actuals: list[int],
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float]:
    """
    Calculate the business cost of model predictions.

    In a binary classification scenario (0 = negative, 1 = positive):
    - False Negative (FN): Predicted 0, Actual 1 (missed positive)
    - False Positive (FP): Predicted 1, Actual 0 (false alarm)

    Args:
        predictions: List of predicted labels (0 or 1)
        actuals: List of actual labels (0 or 1)
        false_negative_cost: Dollar cost per false negative
        false_positive_cost: Dollar cost per false positive

    Returns:
        Dictionary with keys:
        - "false_negatives": count of FN
        - "false_positives": count of FP
        - "true_positives": count of TP
        - "true_negatives": count of TN
        - "total_cost": total dollar cost (FN * fn_cost + FP * fp_cost)
        - "accuracy": traditional accuracy metric
        - "cost_per_prediction": average cost per prediction

    Raises:
        ValueError: If predictions and actuals have different lengths
        ValueError: If lists are empty
        ValueError: If values are not 0 or 1

    Example:
        >>> preds = [0, 1, 0, 1]
        >>> actual = [0, 0, 1, 1]  # FN at index 2, FP at index 1
        >>> result = calculate_business_metric(preds, actual, 100, 10)
        >>> result["false_negatives"]
        1
        >>> result["false_positives"]
        1
        >>> result["total_cost"]
        110.0
    """
    # TODO: Step 1 - Input validation
    # Check that predictions and actuals have the same length
    # Check that neither list is empty
    # Check that all values are 0 or 1

    # TODO: Step 2 - Count confusion matrix components
    # Initialize counters: tp, tn, fp, fn = 0, 0, 0, 0
    # Loop through predictions and actuals together
    # Classify each pair into one of the four categories

    # TODO: Step 3 - Calculate metrics
    # total_cost = fn * false_negative_cost + fp * false_positive_cost
    # accuracy = (tp + tn) / total
    # cost_per_prediction = total_cost / total

    # TODO: Step 4 - Return the results dictionary

    pass  # Remove this when you implement


# =============================================================================
# PART B: Debug Exercise - Config Loading with Hardcoded Paths
# =============================================================================

def load_config_broken(config_name: str) -> dict:
    """
    BUG: This function has hardcoded paths that only work on the original
    developer's machine. It will fail in:
    - Different user home directories
    - Docker containers
    - CI/CD pipelines
    - Windows vs Unix systems

    Find and fix all the hardcoded path issues.
    """
    # BUG 1: Hardcoded absolute path
    config_dir = "/Users/john/projects/mlops/configs"

    # BUG 2: Hardcoded path separator (fails on Windows)
    config_path = config_dir + "/" + config_name + ".json"

    # BUG 3: No fallback for missing config
    with open(config_path) as f:
        config = json.load(f)

    # BUG 4: Hardcoded model path in the config override
    config["model_path"] = "/Users/john/models/sentiment_model.pkl"

    return config


def load_config(config_name: str, config_dir: Path | None = None) -> dict:
    """
    Load a JSON configuration file in an environment-agnostic way.

    Args:
        config_name: Name of the config file (without .json extension)
        config_dir: Optional directory containing configs. If None, uses
                   the 'configs' directory relative to this file's location.

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON

    Example:
        >>> config = load_config("development")
        >>> "model_path" in config or True  # May or may not have model_path
        True
    """
    # TODO: Step 1 - Determine config directory
    # If config_dir is None, use Path(__file__).parent / "configs"
    # Make sure to resolve the path to handle symlinks

    # TODO: Step 2 - Build the full path using pathlib
    # Use the / operator for cross-platform path joining
    # Add the .json extension

    # TODO: Step 3 - Load and return the config
    # Open the file and parse JSON
    # Let FileNotFoundError and JSONDecodeError propagate naturally

    pass  # Remove this when you implement


# =============================================================================
# Conceptual Question (10% of exercise)
# =============================================================================

"""
CONCEPTUAL QUESTION:

A model has 95% accuracy but costs the business $1M per month in false negatives.
A different model has 90% accuracy but only costs $200K per month.

1. Which model should you deploy and why?

2. What additional information would help you make this decision?

3. How would you monitor this in production?

Write your answers as comments below:
"""

# YOUR ANSWER 1:
#

# YOUR ANSWER 2:
#

# YOUR ANSWER 3:
#


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
/hint 1 - calculate_business_metric:
For confusion matrix counting, think through each case:
- pred=1, actual=1 -> True Positive (correctly identified positive)
- pred=0, actual=0 -> True Negative (correctly identified negative)
- pred=1, actual=0 -> False Positive (false alarm)
- pred=0, actual=1 -> False Negative (missed positive)

/hint 2 - calculate_business_metric:
Use zip() to iterate through both lists together:
    for pred, actual in zip(predictions, actuals):
        if pred == 1 and actual == 1:
            tp += 1
        elif ...

/hint 3 - calculate_business_metric:
Complete validation:
    if len(predictions) != len(actuals):
        raise ValueError("Length mismatch")
    if len(predictions) == 0:
        raise ValueError("Empty input")
    valid_values = {0, 1}
    if not all(p in valid_values for p in predictions):
        raise ValueError("Predictions must be 0 or 1")
    if not all(a in valid_values for a in actuals):
        raise ValueError("Actuals must be 0 or 1")

/hint 1 - load_config:
Use pathlib.Path for all path operations:
    from pathlib import Path
    config_dir = Path(__file__).parent / "configs"

/hint 2 - load_config:
Environment variables can provide flexible configuration:
    config_dir = Path(os.environ.get("CONFIG_DIR", default_dir))

/hint 3 - load_config:
Complete solution:
    def load_config(config_name, config_dir=None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        config_dir = Path(config_dir).resolve()

        config_path = config_dir / f"{config_name}.json"

        with open(config_path) as f:
            return json.load(f)
"""
