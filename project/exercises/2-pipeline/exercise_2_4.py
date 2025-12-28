"""
Exercise 2.4: Testing ML Code
Difficulty: ★★★
Topic: ML-specific testing patterns and property-based testing

Learning Objectives:
- Validate ML model outputs systematically
- Fix common sources of test flakiness in ML
- Apply property-based testing to ML code

Instructions:

PART A - Write & Verify (50%):
Implement `validate_model_output()` that checks prediction quality:
- Correct types and shapes
- No null values
- Probability distributions sum to 1
- Predictions within expected ranges

PART B - Debug (30%):
Fix `flaky_test_model_training()` which fails randomly due to
model randomness. Make it deterministic.

PART C - Extend (20%):
Add property-based testing using hypothesis to test model
invariants across random inputs.

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


# Optional imports for testing
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    from hypothesis import given, settings, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


class ValidationLevel(Enum):
    """How strict should validation be?"""
    STRICT = "strict"      # Any issue is a failure
    WARNING = "warning"    # Log warnings but don't fail
    LENIENT = "lenient"    # Only fail on critical issues


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    level: str  # "error", "warning", "info"
    check_name: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of model output validation."""
    is_valid: bool
    issues: list[ValidationIssue]
    statistics: dict[str, Any]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]


# =============================================================================
# PART A: Write & Verify - Model Output Validation
# =============================================================================

def validate_model_output(
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
    expected_classes: list[int] | None = None,
    n_samples: int | None = None,
    level: ValidationLevel = ValidationLevel.STRICT
) -> ValidationResult:
    """Validate ML model predictions for correctness.

    Performs comprehensive checks on model output:
    1. Type checking: predictions are correct numpy types
    2. Shape checking: dimensions match expected
    3. Null checking: no NaN or None values
    4. Range checking: predictions are valid class labels
    5. Probability checking: probabilities sum to 1 (if provided)

    Args:
        predictions: Array of predicted class labels.
        probabilities: Optional array of class probabilities (n_samples, n_classes).
        expected_classes: List of valid class labels (e.g., [0, 1] for binary).
        n_samples: Expected number of samples.
        level: How strict to be about issues.

    Returns:
        ValidationResult with validity status and any issues found.

    Example:
        >>> preds = np.array([0, 1, 1, 0, 2])
        >>> probs = np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6]])
        >>> result = validate_model_output(preds, probs, expected_classes=[0, 1])
        >>> if not result.is_valid:
        ...     print(f"Issues: {result.issues}")
    """
    # TODO: Implement comprehensive validation
    #
    # Steps:
    # 1. Initialize empty issues list
    # 2. Check predictions is numpy array
    # 3. Check for NaN/null values
    # 4. Check predictions are in expected_classes (if provided)
    # 5. Check shape matches n_samples (if provided)
    # 6. If probabilities provided:
    #    - Check same number of samples
    #    - Check probabilities sum to 1 (with tolerance)
    #    - Check all probabilities in [0, 1]
    # 7. Calculate statistics (mean, std, class distribution)
    # 8. Determine is_valid based on level and issues

    raise NotImplementedError("Implement validate_model_output()")


def check_predictions_type(predictions: Any) -> list[ValidationIssue]:
    """Check that predictions are the correct type.

    Args:
        predictions: Model predictions to check.

    Returns:
        List of issues found (empty if valid).
    """
    # TODO: Implement type checking
    # Check: is numpy array, is numeric dtype

    raise NotImplementedError("Implement check_predictions_type()")


def check_null_values(arr: np.ndarray, name: str = "array") -> list[ValidationIssue]:
    """Check for null/NaN values in array.

    Args:
        arr: Array to check.
        name: Name for error messages.

    Returns:
        List of issues found.
    """
    # TODO: Implement null checking
    # Use np.isnan() for float arrays, handle int arrays separately

    raise NotImplementedError("Implement check_null_values()")


def check_probability_distribution(
    probabilities: np.ndarray,
    tolerance: float = 1e-6
) -> list[ValidationIssue]:
    """Check that probability distributions are valid.

    Args:
        probabilities: Array of shape (n_samples, n_classes).
        tolerance: Allowed deviation from sum=1.

    Returns:
        List of issues found.
    """
    # TODO: Implement probability checking
    # Checks:
    # - All values in [0, 1]
    # - Each row sums to 1 (within tolerance)
    # - Shape is 2D

    raise NotImplementedError("Implement check_probability_distribution()")


def check_class_distribution(
    predictions: np.ndarray,
    expected_classes: list[int],
    min_class_fraction: float = 0.0
) -> list[ValidationIssue]:
    """Check class distribution in predictions.

    Args:
        predictions: Predicted class labels.
        expected_classes: Valid class labels.
        min_class_fraction: Minimum fraction of each class (0 to disable).

    Returns:
        List of issues found.
    """
    # TODO: Implement class distribution checking
    # Useful for detecting degenerate models that always predict same class

    raise NotImplementedError("Implement check_class_distribution()")


# =============================================================================
# PART B: Debug - Fix Flaky Test
# =============================================================================

def buggy_test_model_training():
    """BUGGY: This test fails randomly about 20% of the time!

    Bugs:
    1. No random seed - model training is non-deterministic
    2. Exact equality check on floating point metrics
    3. No tolerance in accuracy comparison
    4. Single sample size makes metrics unstable
    """
    # Bug 1: No seed
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    # Bug 4: Small sample size
    X, y = make_classification(n_samples=20, n_features=5, random_state=None)

    model = LogisticRegression()  # Bug 1: No random_state
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()

    # Bug 2 & 3: Exact equality on float
    assert accuracy == 0.95  # This will almost never pass!


def fixed_test_model_training():
    """FIXED: Deterministic test with proper assertions.

    Fixes:
    1. Set random seeds for all random operations
    2. Use approximate comparison for floats
    3. Use reasonable tolerance based on sample size
    4. Use larger sample size for stable metrics
    """
    # TODO: Fix all four bugs
    #
    # Fix 1: Set random seeds
    #   - np.random.seed(42)
    #   - random_state=42 in make_classification
    #   - random_state=42 in LogisticRegression
    #
    # Fix 2 & 3: Use approximate comparison
    #   - assert accuracy >= 0.8  (lower bound instead of exact)
    #   - or use pytest.approx(0.95, abs=0.1)
    #   - or use np.isclose(accuracy, 0.95, atol=0.1)
    #
    # Fix 4: Use more samples
    #   - n_samples=100 or more

    raise NotImplementedError("Implement fixed_test_model_training()")


def deterministic_model_training(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> tuple[Any, float]:
    """Train a model deterministically.

    This wrapper ensures reproducible training.

    Args:
        X: Feature matrix.
        y: Labels.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (trained model, accuracy).
    """
    # TODO: Implement deterministic training
    # Set all relevant random states

    raise NotImplementedError("Implement deterministic_model_training()")


def create_reproducible_test_data(
    n_samples: int = 100,
    n_features: int = 10,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible test data for ML tests.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        random_state: Random seed.

    Returns:
        Tuple of (X, y) arrays.
    """
    # TODO: Implement reproducible data generation

    raise NotImplementedError("Implement create_reproducible_test_data()")


# =============================================================================
# PART C: Extend - Property-Based Testing
# =============================================================================

# Property-based testing generates many random inputs and checks
# that certain properties always hold.

def property_predictions_correct_shape(model, n_features: int):
    """Property: predictions should always have correct shape.

    For any valid input shape, output should have matching first dimension.
    """
    # TODO: Implement this property test
    #
    # Given: random n_samples between 1 and 1000
    # When: model.predict(X) where X has shape (n_samples, n_features)
    # Then: predictions.shape == (n_samples,)

    raise NotImplementedError("Implement property_predictions_correct_shape()")


def property_probabilities_valid(model, n_features: int):
    """Property: probabilities should always be valid distributions.

    For any input, predict_proba output should:
    - Have values in [0, 1]
    - Have rows that sum to 1
    """
    # TODO: Implement this property test

    raise NotImplementedError("Implement property_probabilities_valid()")


def property_predictions_deterministic(model, X: np.ndarray):
    """Property: same input should give same output.

    Model predictions should be deterministic (for most model types).
    """
    # TODO: Implement this property test
    # Call predict twice, assert outputs are equal

    raise NotImplementedError("Implement property_predictions_deterministic()")


# Hypothesis strategies for ML testing
if HYPOTHESIS_AVAILABLE:
    # Strategy for generating valid feature matrices
    def feature_matrix_strategy(
        min_samples: int = 1,
        max_samples: int = 100,
        n_features: int = 10
    ):
        """Generate random valid feature matrices."""
        return st.integers(min_value=min_samples, max_value=max_samples).flatmap(
            lambda n: st.just(np.random.randn(n, n_features))
        )

    # Strategy for generating valid labels
    def label_strategy(n_samples: int, n_classes: int = 2):
        """Generate random valid labels."""
        return st.just(np.random.randint(0, n_classes, size=n_samples))


def create_property_test_suite(model, n_features: int):
    """Create a suite of property tests for a model.

    Args:
        model: Trained sklearn model.
        n_features: Number of features model expects.

    Returns:
        List of test functions to run.

    Example:
        >>> model = LogisticRegression().fit(X, y)
        >>> tests = create_property_test_suite(model, n_features=10)
        >>> for test in tests:
        ...     test()  # Runs the property test
    """
    # TODO: Create test suite using hypothesis if available

    raise NotImplementedError("Implement create_property_test_suite()")


# =============================================================================
# Utility functions
# =============================================================================

def calculate_prediction_statistics(predictions: np.ndarray) -> dict:
    """Calculate statistics about predictions.

    Args:
        predictions: Array of predictions.

    Returns:
        Dictionary with mean, std, unique values, class counts, etc.
    """
    unique, counts = np.unique(predictions, return_counts=True)
    return {
        "n_samples": len(predictions),
        "n_unique": len(unique),
        "unique_values": unique.tolist(),
        "class_counts": dict(zip(unique.tolist(), counts.tolist())),
        "most_common_class": unique[np.argmax(counts)],
        "most_common_fraction": counts.max() / len(predictions)
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    print("Testing validation...")

    # Create sample predictions
    np.random.seed(42)
    preds = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    probs = np.random.dirichlet([1, 1], size=10)

    print(f"Predictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Prob sums: {probs.sum(axis=1)}")

    # Test validation
    # result = validate_model_output(preds, probs, expected_classes=[0, 1])
    # print(f"Valid: {result.is_valid}")
    # print(f"Issues: {result.issues}")

    print("\nTesting deterministic training...")
    # X, y = create_reproducible_test_data()
    # model, acc = deterministic_model_training(X, y)
    # print(f"Accuracy: {acc}")


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Conceptual) ===
For validate_model_output:
- np.isnan(arr).any() checks for NaN values
- np.isin(predictions, expected_classes).all() checks valid classes
- probabilities.sum(axis=1) should be all 1s
- np.allclose(sums, 1.0, atol=tolerance) for approximate comparison

For fixed_test_model_training:
- Set np.random.seed(42) at start
- Use random_state=42 in all sklearn functions
- Use assert accuracy >= 0.8 or np.isclose()

=== HINT 2 (More specific) ===
For check_probability_distribution:
```python
issues = []
if probabilities.ndim != 2:
    issues.append(ValidationIssue("error", "shape", "Expected 2D array"))
    return issues

if (probabilities < 0).any() or (probabilities > 1).any():
    issues.append(ValidationIssue("error", "range", "Values outside [0,1]"))

row_sums = probabilities.sum(axis=1)
if not np.allclose(row_sums, 1.0, atol=tolerance):
    bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=tolerance))[0]
    issues.append(ValidationIssue("error", "sum", f"Rows don't sum to 1: {bad_rows}"))

return issues
```

For deterministic_model_training:
```python
np.random.seed(random_state)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=random_state, solver='lbfgs')
model.fit(X, y)
accuracy = (model.predict(X) == y).mean()
return model, accuracy
```

=== HINT 3 (Nearly complete solution) ===
def validate_model_output(...) -> ValidationResult:
    issues = []
    statistics = {}

    # Type check
    if not isinstance(predictions, np.ndarray):
        issues.append(ValidationIssue("error", "type", "predictions must be numpy array"))
        return ValidationResult(is_valid=False, issues=issues, statistics={})

    # Null check
    if np.issubdtype(predictions.dtype, np.floating):
        if np.isnan(predictions).any():
            n_null = np.isnan(predictions).sum()
            issues.append(ValidationIssue("error", "null", f"Found {n_null} NaN values"))

    # Shape check
    if n_samples is not None and len(predictions) != n_samples:
        issues.append(ValidationIssue("error", "shape",
            f"Expected {n_samples} samples, got {len(predictions)}"))

    # Class check
    if expected_classes is not None:
        invalid = ~np.isin(predictions, expected_classes)
        if invalid.any():
            bad_values = np.unique(predictions[invalid])
            issues.append(ValidationIssue("error", "class",
                f"Invalid classes found: {bad_values}"))

    # Probability check
    if probabilities is not None:
        if probabilities.shape[0] != len(predictions):
            issues.append(ValidationIssue("error", "prob_shape", "Shape mismatch"))
        else:
            row_sums = probabilities.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                issues.append(ValidationIssue("error", "prob_sum", "Probabilities don't sum to 1"))
            if (probabilities < 0).any() or (probabilities > 1).any():
                issues.append(ValidationIssue("error", "prob_range", "Probabilities outside [0,1]"))

    # Calculate statistics
    statistics = calculate_prediction_statistics(predictions)

    # Determine validity
    has_errors = any(i.level == "error" for i in issues)
    is_valid = not has_errors if level == ValidationLevel.STRICT else True

    return ValidationResult(is_valid=is_valid, issues=issues, statistics=statistics)

def fixed_test_model_training():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    # Fix 1 & 4: Seed and larger sample
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    # Fix 1: Seed in model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()

    # Fix 2 & 3: Threshold instead of exact
    assert accuracy >= 0.8, f"Accuracy {accuracy} below threshold"
"""
