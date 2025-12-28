"""
Exercise 4.1: CI/CD for Machine Learning
Difficulty: *** (Production-level debugging)
Topic: GitHub Actions, dependency caching, CI/CD pipelines, deployment gates

Instructions:
This exercise simulates real production CI/CD bugs you'll encounter. Each section presents
a broken configuration or code that you need to debug and fix.

Scenario: You've joined a team with an existing ML pipeline. The CI/CD is "mostly working"
but has several issues that are causing slow builds, flaky tests, and occasional bad deploys.

Part 1: Fix the GitHub Actions caching (DEBUG)
Part 2: Fix the CI environment mismatch (DEBUG)
Part 3: Implement a model validation gate (WRITE & VERIFY)

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

from pathlib import Path
from typing import TypedDict
import json
import time

# =============================================================================
# PART 1: DEBUG - Fix GitHub Actions Caching
# =============================================================================
#
# The following YAML was extracted from a team's workflow. Despite having caching
# configured, every build reinstalls dependencies from scratch (taking 3+ minutes).
# The team is confused because "the cache action is right there!"
#
# BUG REPORT FROM TEAM:
# """
# CI Build Times:
# - Expected: ~45 seconds (with cache hit)
# - Actual: 3-4 minutes (cache always misses)
#
# We added caching last week but it doesn't seem to work.
# The logs show "Cache not found for key: ..." every time.
# """
#
# Your task: Examine the workflow YAML below and identify the bug.
# Then implement the fixed version in `fix_github_actions_cache()`.

BROKEN_WORKFLOW_YAML = """
name: ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Run tests
        run: pytest tests/ -v
"""

def fix_github_actions_cache() -> str:
    """
    Return the FIXED workflow YAML that properly caches dependencies.

    The original has a common mistake that causes cache misses.
    Your fix should result in cache hits on subsequent runs.

    Returns:
        str: The corrected YAML workflow content
    """
    # TODO: Return the fixed YAML
    # HINT: Think about the ORDER of steps in the workflow
    # HINT: When does the cache action restore vs save?

    fixed_yaml = """
# TODO: Write the corrected workflow YAML here
# The fix involves reordering steps and possibly adjusting the cache key
"""
    return fixed_yaml


# =============================================================================
# PART 2: DEBUG - Fix CI Environment Mismatch
# =============================================================================
#
# A developer reports: "Tests pass locally but fail in CI with import errors"
#
# ERROR LOG FROM CI:
# """
# ============================= test session starts ==============================
# platform linux -- Python 3.11.0, pytest-7.4.0
# collected 15 items
#
# tests/test_model.py F
#
# FAILED tests/test_model.py::test_model_prediction - ModuleNotFoundError: No module named 'sklearn'
# FAILED tests/test_model.py::test_feature_extraction - ModuleNotFoundError: No module named 'nltk'
#
# ============================= 2 failed, 13 passed ===============================
# """
#
# LOCAL ENVIRONMENT:
# """
# $ pytest tests/ -v
# ============================= test session starts ==============================
# collected 15 items
#
# tests/test_model.py::test_model_prediction PASSED
# tests/test_model.py::test_feature_extraction PASSED
# ...
# ============================= 15 passed =========================================
# """

# Here are the project's requirement files:

REQUIREMENTS_TXT = """
# requirements.txt - Production dependencies
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
joblib>=1.3.0
numpy>=1.24.0
"""

REQUIREMENTS_DEV_TXT = """
# requirements-dev.txt - Development dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
"""

# The CI workflow installs both files:
CI_INSTALL_STEP = """
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
"""


def diagnose_ci_environment_bug() -> dict:
    """
    Analyze the requirements files and CI configuration to identify the bug.

    Returns:
        dict with keys:
        - 'root_cause': str explaining why tests fail in CI but pass locally
        - 'missing_packages': list of packages that should be added
        - 'file_to_modify': which requirements file needs the fix
        - 'fixed_requirements': the corrected file content
    """
    # TODO: Implement your diagnosis
    # HINT: Compare what packages the tests need vs what's installed in CI
    # HINT: Why would it work locally but not in CI?

    diagnosis = {
        'root_cause': "",  # TODO: Explain the bug
        'missing_packages': [],  # TODO: List missing packages
        'file_to_modify': "",  # TODO: Which file?
        'fixed_requirements': "",  # TODO: Provide fixed content
    }
    return diagnosis


# =============================================================================
# PART 3: WRITE & VERIFY - Model Validation Gate
# =============================================================================
#
# Before deploying a model, you need to validate it meets minimum quality standards.
# This is a "gate" that blocks bad models from reaching production.
#
# Implement a validation function that checks:
# 1. Model file exists and can be loaded
# 2. Model accuracy on test data meets threshold
# 3. Model inference latency is acceptable
# 4. Model makes valid predictions (not all same class)


class ValidationResult(TypedDict):
    """Result of model validation."""
    passed: bool
    accuracy: float
    accuracy_threshold: float
    latency_p99_ms: float
    latency_threshold_ms: float
    prediction_distribution: dict[str, float]
    checks: dict[str, bool]
    errors: list[str]


def validate_model_before_deploy(
    model_path: str | Path,
    test_data_path: str | Path,
    accuracy_threshold: float = 0.85,
    latency_threshold_ms: float = 100.0,
    min_class_ratio: float = 0.1,
) -> ValidationResult:
    """
    Validate a model meets deployment requirements.

    This function is called by CI/CD before deploying a new model.
    If it returns passed=False, the deployment is blocked.

    Args:
        model_path: Path to the trained model file (.pkl or .joblib)
        test_data_path: Path to test CSV with 'text' and 'label' columns
        accuracy_threshold: Minimum required accuracy (0-1)
        latency_threshold_ms: Maximum p99 latency in milliseconds
        min_class_ratio: Minimum ratio for any predicted class (catches degenerate models)

    Returns:
        ValidationResult with all validation details

    Example:
        >>> result = validate_model_before_deploy(
        ...     "models/sentiment.pkl",
        ...     "data/test.csv",
        ...     accuracy_threshold=0.85
        ... )
        >>> if not result['passed']:
        ...     print("Blocking deployment:", result['errors'])
    """
    # TODO: Implement the validation logic
    #
    # Steps:
    # 1. Try to load the model (handle FileNotFoundError)
    # 2. Load test data (handle FileNotFoundError, validate columns exist)
    # 3. Run predictions and calculate accuracy
    # 4. Measure latency (run multiple predictions, get p99)
    # 5. Check prediction distribution (no class should be < min_class_ratio)
    # 6. Compile results into ValidationResult
    #
    # Remember: Follow the coding best practices from CLAUDE.md
    # - Use specific exceptions (FileNotFoundError, not Exception)
    # - No silent failures
    # - Honest type signatures

    errors: list[str] = []
    checks: dict[str, bool] = {}

    # TODO: Your implementation here

    # Placeholder return - replace with your implementation
    return ValidationResult(
        passed=False,
        accuracy=0.0,
        accuracy_threshold=accuracy_threshold,
        latency_p99_ms=0.0,
        latency_threshold_ms=latency_threshold_ms,
        prediction_distribution={},
        checks=checks,
        errors=["Not implemented"],
    )


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Part 1 - Caching) ===

Look at the ORDER of steps in the workflow:
1. Checkout
2. Setup Python
3. Install dependencies  <-- pip install happens HERE
4. Cache dependencies    <-- Cache restore happens HERE (TOO LATE!)
5. Run tests

The cache action does TWO things:
- At the START of the step: tries to RESTORE from cache
- At the END of the job: SAVES to cache

If you install before caching, the restore happens AFTER install!

The fix: Move the cache step BEFORE the install step.


=== HINT 2 (Part 2 - CI Environment) ===

Look at what packages are in requirements.txt vs requirements-dev.txt:
- requirements.txt: fastapi, uvicorn, pydantic, joblib, numpy
- requirements-dev.txt: pytest, black, flake8, mypy

The test errors mention: sklearn, nltk

Neither file includes sklearn or nltk! But tests pass locally...

Why? The developer probably installed these manually (`pip install sklearn`)
in their local environment but forgot to add them to requirements files.

The fix: Add the ML packages (scikit-learn, nltk) to requirements.txt


=== HINT 3 (Part 3 - Validation Gate) ===

Here's a structure for the implementation:

```python
def validate_model_before_deploy(...) -> ValidationResult:
    errors = []
    checks = {}

    # Step 1: Load model
    model_path = Path(model_path)
    try:
        import joblib
        model = joblib.load(model_path)
        checks['model_loads'] = True
    except FileNotFoundError:
        errors.append(f"Model not found: {model_path}")
        checks['model_loads'] = False
        # Return early - can't continue without model
        return ValidationResult(passed=False, ...)

    # Step 2: Load test data
    try:
        import pandas as pd
        test_df = pd.read_csv(test_data_path)
        # Validate required columns exist
        if 'text' not in test_df.columns or 'label' not in test_df.columns:
            raise ValueError("Missing required columns")
        checks['data_loads'] = True
    except FileNotFoundError:
        ...

    # Step 3: Calculate accuracy
    predictions = model.predict(test_df['text'].tolist())
    accuracy = (predictions == test_df['label']).mean()
    checks['accuracy_threshold'] = accuracy >= accuracy_threshold

    # Step 4: Measure latency
    import time
    latencies = []
    for text in test_df['text'].head(100):
        start = time.perf_counter()
        model.predict([text])
        latencies.append((time.perf_counter() - start) * 1000)

    import numpy as np
    latency_p99 = np.percentile(latencies, 99)
    checks['latency_threshold'] = latency_p99 <= latency_threshold_ms

    # Step 5: Check distribution
    from collections import Counter
    pred_counts = Counter(predictions)
    total = len(predictions)
    distribution = {k: v/total for k, v in pred_counts.items()}

    min_ratio = min(distribution.values())
    checks['distribution_balanced'] = min_ratio >= min_class_ratio

    # Step 6: Compile result
    passed = all(checks.values())

    return ValidationResult(
        passed=passed,
        accuracy=accuracy,
        accuracy_threshold=accuracy_threshold,
        latency_p99_ms=latency_p99,
        latency_threshold_ms=latency_threshold_ms,
        prediction_distribution=distribution,
        checks=checks,
        errors=errors,
    )
```

The key production concerns:
- Handle all failure modes explicitly
- Return detailed diagnostics (which checks failed, why)
- Don't let a bad model slip through
"""
