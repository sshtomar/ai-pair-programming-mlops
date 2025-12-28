"""
Exercise 2.2: Experiment Tracking
Difficulty: ★★☆
Topic: MLflow experiment logging and run management

Learning Objectives:
- Create structured MLflow logging wrappers
- Debug common MLflow issues (duplicate runs, nested contexts)
- Add automatic metadata capture (git info, timestamps)

Instructions:

PART A - Write & Verify (50%):
Implement `log_experiment()` that wraps MLflow with proper run naming,
parameter logging, metric logging, and artifact handling.

PART B - Debug (30%):
The `buggy_train_with_logging()` function creates duplicate runs.
Fix it in `fixed_train_with_logging()`.

PART C - Extend (20%):
Add automatic git commit hash logging to capture code version.

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


# Note: MLflow import is optional for the exercise structure
# Students should have MLflow installed: pip install mlflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Install with: pip install mlflow")


@dataclass
class ExperimentResult:
    """Result of logging an experiment."""
    run_id: str
    run_name: str
    experiment_id: str
    artifact_uri: str
    params: dict[str, Any]
    metrics: dict[str, float]


# =============================================================================
# PART A: Write & Verify - MLflow Logging Wrapper
# =============================================================================

def log_experiment(
    experiment_name: str,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: dict[str, str | Path] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None
) -> ExperimentResult:
    """Log an ML experiment to MLflow with proper structure.

    This wrapper ensures consistent logging practices:
    - Automatic run naming with timestamps
    - Nested parameter logging (flattens dicts)
    - Metric logging with validation
    - Artifact logging with proper paths

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Optional run name. If None, generates timestamp-based name.
        params: Dictionary of parameters to log.
        metrics: Dictionary of metrics to log.
        artifacts: Dictionary mapping artifact names to file paths.
        tags: Additional tags to add to the run.
        tracking_uri: MLflow tracking server URI. If None, uses local.

    Returns:
        ExperimentResult with run details.

    Raises:
        ValueError: If metrics contain non-numeric values.

    Example:
        >>> result = log_experiment(
        ...     experiment_name="sentiment-classifier",
        ...     params={"learning_rate": 0.01, "epochs": 10},
        ...     metrics={"accuracy": 0.95, "f1": 0.93},
        ...     artifacts={"model": "models/model.pkl"}
        ... )
        >>> print(f"Logged run: {result.run_id}")
    """
    # TODO: Implement this function
    #
    # Steps:
    # 1. Set tracking URI if provided
    # 2. Set or create experiment
    # 3. Generate run name if not provided (use timestamp)
    # 4. Start MLflow run
    # 5. Log parameters (flatten nested dicts)
    # 6. Validate and log metrics
    # 7. Log artifacts
    # 8. Log tags
    # 9. Return ExperimentResult

    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow is required. Install with: pip install mlflow")

    raise NotImplementedError("Implement log_experiment()")


def generate_run_name(prefix: str = "run") -> str:
    """Generate a unique run name with timestamp.

    Args:
        prefix: Prefix for the run name.

    Returns:
        Run name like "run_2024-01-15_14-30-45"
    """
    # TODO: Implement this helper
    # Use datetime.now().strftime() to create a timestamp

    raise NotImplementedError("Implement generate_run_name()")


def flatten_params(params: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten nested parameter dictionaries for MLflow logging.

    MLflow doesn't handle nested dicts well, so we flatten them.

    Args:
        params: Possibly nested parameter dictionary.
        parent_key: Prefix for nested keys.

    Returns:
        Flattened dictionary with dot-notation keys.

    Example:
        >>> flatten_params({"model": {"lr": 0.01, "layers": 3}})
        {"model.lr": 0.01, "model.layers": 3}
    """
    # TODO: Implement recursive flattening
    #
    # For each key-value pair:
    # - If value is a dict, recursively flatten with updated parent_key
    # - Otherwise, add to result with full key path

    raise NotImplementedError("Implement flatten_params()")


def validate_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Validate that all metrics are numeric.

    Args:
        metrics: Dictionary of metrics to validate.

    Returns:
        Dictionary with validated float metrics.

    Raises:
        ValueError: If any metric value is not numeric.
    """
    # TODO: Implement validation
    # Check each value is int or float, convert to float

    raise NotImplementedError("Implement validate_metrics()")


# =============================================================================
# PART B: Debug - Fix Duplicate Runs
# =============================================================================

def buggy_train_with_logging(data_path: str, params: dict) -> str:
    """BUGGY: This function creates duplicate MLflow runs!

    Can you spot the bugs?

    Bug 1: Starts a run but also uses autolog which starts another run
    Bug 2: Doesn't properly end the run on exceptions
    Bug 3: Creates new experiment each time instead of reusing
    """
    if not MLFLOW_AVAILABLE:
        return "mlflow_not_available"

    # Bug 1: autolog starts its own run
    mlflow.autolog()

    # Bug 3: This creates a new experiment each call if it exists
    experiment_id = mlflow.create_experiment(f"training_{datetime.now()}")
    mlflow.set_experiment(experiment_id=experiment_id)

    # Bug 2: No exception handling - run stays open on failure
    run = mlflow.start_run(run_name="training")

    mlflow.log_params(params)

    # Simulate training
    accuracy = 0.95

    mlflow.log_metric("accuracy", accuracy)

    # Bug 2: If exception occurs above, this never runs
    mlflow.end_run()

    return run.info.run_id


def fixed_train_with_logging(data_path: str, params: dict) -> str:
    """FIXED version of the buggy training function.

    Fixes:
    1. Disable autolog or configure it properly
    2. Use context manager for proper run cleanup
    3. Use set_experiment to reuse existing experiments

    Args:
        data_path: Path to training data.
        params: Training parameters.

    Returns:
        MLflow run ID.
    """
    # TODO: Fix all three bugs
    #
    # Fix 1: Either disable autolog or use it exclusively (not both)
    # Fix 2: Use `with mlflow.start_run() as run:` context manager
    # Fix 3: Use mlflow.set_experiment("name") which creates OR reuses

    if not MLFLOW_AVAILABLE:
        return "mlflow_not_available"

    raise NotImplementedError("Implement fixed_train_with_logging()")


# =============================================================================
# PART C: Extend - Git Commit Logging
# =============================================================================

def get_git_info() -> dict[str, str]:
    """Get current git information for experiment tracking.

    Returns:
        Dictionary with git commit, branch, and dirty status.

    Example:
        >>> info = get_git_info()
        >>> print(info)
        {
            "git_commit": "abc123def456...",
            "git_branch": "main",
            "git_dirty": "false",
            "git_remote": "origin/main"
        }
    """
    # TODO: Implement git info extraction
    #
    # Use subprocess to run:
    # - git rev-parse HEAD (get commit hash)
    # - git rev-parse --abbrev-ref HEAD (get branch name)
    # - git status --porcelain (check if dirty)
    #
    # Handle case where git is not available or not a git repo

    raise NotImplementedError("Implement get_git_info()")


def log_experiment_with_git(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    **kwargs
) -> ExperimentResult:
    """Log experiment with automatic git commit tracking.

    This extends log_experiment() to automatically capture:
    - Git commit hash
    - Branch name
    - Whether working directory is dirty

    Why this matters:
    - Reproducibility: Know exactly which code produced results
    - Debugging: Track down when bugs were introduced
    - Auditing: Prove which code version was used in production

    Args:
        experiment_name: Name of the experiment.
        params: Training parameters.
        metrics: Training metrics.
        **kwargs: Additional arguments passed to log_experiment.

    Returns:
        ExperimentResult with git info in tags.
    """
    # TODO: Implement this function
    #
    # Steps:
    # 1. Get git info using get_git_info()
    # 2. Add git info to tags (merge with any existing tags)
    # 3. Call log_experiment() with updated tags

    raise NotImplementedError("Implement log_experiment_with_git()")


# =============================================================================
# Utility functions for testing without MLflow
# =============================================================================

@dataclass
class MockRun:
    """Mock MLflow run for testing."""
    run_id: str = "mock_run_123"
    run_name: str = "test_run"
    experiment_id: str = "mock_exp_456"
    artifact_uri: str = "/tmp/mlruns"
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)


class MockMLflowContext:
    """Mock MLflow context for testing without MLflow installed."""

    def __init__(self, run_name: str = "test"):
        self.run = MockRun(run_name=run_name)
        self._params = {}
        self._metrics = {}
        self._artifacts = []

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value

    def log_params(self, params: dict) -> None:
        self._params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        self._metrics[key] = value

    def log_metrics(self, metrics: dict) -> None:
        self._metrics.update(metrics)

    def log_artifact(self, path: str) -> None:
        self._artifacts.append(path)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Test run name generation
    print("Testing generate_run_name...")
    # name = generate_run_name("experiment")
    # print(f"Generated: {name}")

    # Test parameter flattening
    print("\nTesting flatten_params...")
    nested = {
        "model": {"type": "logistic", "C": 1.0},
        "data": {"train_size": 0.8}
    }
    # flat = flatten_params(nested)
    # print(f"Flattened: {flat}")

    # Test git info
    print("\nTesting get_git_info...")
    # info = get_git_info()
    # print(f"Git info: {info}")


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Conceptual) ===
For log_experiment:
- MLflow context manager: `with mlflow.start_run(run_name=name) as run:`
- Log params: mlflow.log_params(params)
- Log metrics: mlflow.log_metrics(metrics)
- Log artifacts: mlflow.log_artifact(path)

For generate_run_name:
- datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

For get_git_info:
- subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)

=== HINT 2 (More specific) ===
For flatten_params:
```python
result = {}
for key, value in params.items():
    new_key = f"{parent_key}.{key}" if parent_key else key
    if isinstance(value, dict):
        result.update(flatten_params(value, new_key))
    else:
        result[new_key] = value
return result
```

For fixed_train_with_logging:
```python
# Fix 1: Disable autolog
mlflow.autolog(disable=True)

# Fix 3: Set experiment (creates if needed, reuses if exists)
mlflow.set_experiment("training")

# Fix 2: Use context manager
with mlflow.start_run(run_name="training") as run:
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", 0.95)
    return run.info.run_id
```

=== HINT 3 (Nearly complete solution) ===
def log_experiment(...) -> ExperimentResult:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    if run_name is None:
        run_name = generate_run_name()

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        if params:
            flat_params = flatten_params(params)
            mlflow.log_params(flat_params)

        # Log metrics
        if metrics:
            validated = validate_metrics(metrics)
            mlflow.log_metrics(validated)

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(str(path))

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        return ExperimentResult(
            run_id=run.info.run_id,
            run_name=run_name,
            experiment_id=run.info.experiment_id,
            artifact_uri=run.info.artifact_uri,
            params=params or {},
            metrics=metrics or {}
        )

def get_git_info() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": str(bool(status)).lower()
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown"}
"""
