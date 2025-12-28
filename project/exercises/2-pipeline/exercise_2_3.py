"""
Exercise 2.3: Model Registry
Difficulty: ★★★
Topic: MLflow Model Registry and model lifecycle management

Learning Objectives:
- Understand model versioning and staging concepts
- Implement safe model promotion with validation
- Handle edge cases in model loading gracefully

Instructions:

PART A - Write & Verify (50%):
Implement `promote_model()` that safely transitions a model between
stages (None -> Staging -> Production -> Archived) with validation.

PART B - Debug (30%):
Fix `buggy_load_model()` which doesn't handle missing models,
wrong versions, or network errors gracefully.

PART C - Extend (20%):
Add model comparison before promotion (don't promote worse models).

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


# MLflow import with fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelStage(Enum):
    """Valid stages in MLflow Model Registry."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


# Valid stage transitions
VALID_TRANSITIONS = {
    ModelStage.NONE: [ModelStage.STAGING, ModelStage.ARCHIVED],
    ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.ARCHIVED, ModelStage.NONE],
    ModelStage.PRODUCTION: [ModelStage.STAGING, ModelStage.ARCHIVED],
    ModelStage.ARCHIVED: [ModelStage.STAGING, ModelStage.NONE],
}


@dataclass
class PromotionResult:
    """Result of model promotion."""
    model_name: str
    version: int
    from_stage: str
    to_stage: str
    success: bool
    message: str


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    version: int
    stage: str
    run_id: str
    metrics: dict[str, float]
    creation_timestamp: int


# =============================================================================
# PART A: Write & Verify - Model Promotion
# =============================================================================

def promote_model(
    model_name: str,
    version: int,
    to_stage: ModelStage,
    validate_transition: bool = True,
    archive_existing: bool = True,
    tracking_uri: str | None = None
) -> PromotionResult:
    """Promote a model version to a new stage in the Model Registry.

    This function handles the common workflow of moving models through
    stages: None -> Staging -> Production -> Archived

    Safety features:
    - Validates that the transition is allowed
    - Optionally archives existing models in the target stage
    - Returns detailed result with success/failure info

    Args:
        model_name: Name of the registered model.
        version: Version number to promote.
        to_stage: Target stage (Staging, Production, or Archived).
        validate_transition: If True, only allow valid stage transitions.
        archive_existing: If True, archive models currently in to_stage.
        tracking_uri: MLflow tracking server URI.

    Returns:
        PromotionResult with details of the operation.

    Raises:
        ValueError: If transition is invalid and validate_transition=True.

    Example:
        >>> result = promote_model("sentiment-model", version=3, to_stage=ModelStage.PRODUCTION)
        >>> if result.success:
        ...     print(f"Model v{result.version} is now in {result.to_stage}")
    """
    # TODO: Implement this function
    #
    # Steps:
    # 1. Create MlflowClient
    # 2. Get current model version info
    # 3. Validate transition if required
    # 4. Archive existing models in target stage if required
    # 5. Transition the model to new stage
    # 6. Return PromotionResult

    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow is required. Install with: pip install mlflow")

    raise NotImplementedError("Implement promote_model()")


def validate_stage_transition(from_stage: ModelStage, to_stage: ModelStage) -> bool:
    """Check if a stage transition is valid.

    Args:
        from_stage: Current stage of the model.
        to_stage: Target stage.

    Returns:
        True if transition is allowed, False otherwise.
    """
    # TODO: Implement validation using VALID_TRANSITIONS

    raise NotImplementedError("Implement validate_stage_transition()")


def archive_models_in_stage(
    client: Any,  # MlflowClient
    model_name: str,
    stage: ModelStage
) -> list[int]:
    """Archive all model versions currently in a given stage.

    Args:
        client: MLflow client instance.
        model_name: Name of the registered model.
        stage: Stage to clear.

    Returns:
        List of version numbers that were archived.
    """
    # TODO: Implement archiving
    #
    # Steps:
    # 1. Get all versions of the model
    # 2. Filter to those in the specified stage
    # 3. Transition each to Archived
    # 4. Return list of archived versions

    raise NotImplementedError("Implement archive_models_in_stage()")


def get_model_version_info(
    client: Any,  # MlflowClient
    model_name: str,
    version: int
) -> ModelInfo:
    """Get detailed information about a model version.

    Args:
        client: MLflow client instance.
        model_name: Name of the registered model.
        version: Version number.

    Returns:
        ModelInfo with model details.

    Raises:
        ValueError: If model or version doesn't exist.
    """
    # TODO: Implement info retrieval

    raise NotImplementedError("Implement get_model_version_info()")


# =============================================================================
# PART B: Debug - Fix Model Loading
# =============================================================================

def buggy_load_model(model_name: str, stage: str = "Production"):
    """BUGGY: This function doesn't handle errors gracefully!

    Bugs:
    1. No check if model exists before loading
    2. No handling of network errors
    3. No fallback if Production model doesn't exist
    4. No version validation
    """
    if not MLFLOW_AVAILABLE:
        return None

    # Bug 1: Crashes if model doesn't exist
    model_uri = f"models:/{model_name}/{stage}"

    # Bug 2: No try/except for network issues
    model = mlflow.pyfunc.load_model(model_uri)

    # Bug 3: No fallback to Staging or latest version

    return model


def fixed_load_model(
    model_name: str,
    stage: str = "Production",
    fallback_to_staging: bool = True,
    fallback_to_latest: bool = True
):
    """FIXED: Gracefully load a model with proper error handling.

    Improvements:
    1. Check if model exists before loading
    2. Handle network and MLflow errors
    3. Fallback to Staging if Production unavailable
    4. Fallback to latest version as last resort

    Args:
        model_name: Name of the registered model.
        stage: Preferred stage to load from.
        fallback_to_staging: If True, try Staging if preferred stage fails.
        fallback_to_latest: If True, try latest version as last resort.

    Returns:
        Loaded model or None if all attempts fail.

    Raises:
        Nothing - returns None on failure (with logged warning).
    """
    # TODO: Fix all four bugs
    #
    # Approach:
    # 1. Try to load from preferred stage
    # 2. If fails and fallback_to_staging, try Staging
    # 3. If fails and fallback_to_latest, try latest version
    # 4. Catch specific exceptions (MlflowException, ConnectionError)
    # 5. Return None with warning if all fail

    if not MLFLOW_AVAILABLE:
        return None

    raise NotImplementedError("Implement fixed_load_model()")


class ModelNotFoundError(Exception):
    """Raised when a model cannot be found in the registry."""
    pass


class ModelVersionNotFoundError(Exception):
    """Raised when a specific model version cannot be found."""
    pass


def safe_load_model(
    model_name: str,
    version: int | None = None,
    stage: str | None = None
):
    """Load a model with explicit error types.

    This version raises specific exceptions instead of returning None,
    which is better for debugging and control flow.

    Args:
        model_name: Name of the registered model.
        version: Specific version to load (mutually exclusive with stage).
        stage: Stage to load from (mutually exclusive with version).

    Returns:
        Loaded model.

    Raises:
        ModelNotFoundError: If model doesn't exist in registry.
        ModelVersionNotFoundError: If specified version doesn't exist.
        ValueError: If both version and stage specified.
    """
    # TODO: Implement with specific exceptions

    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow is required")

    raise NotImplementedError("Implement safe_load_model()")


# =============================================================================
# PART C: Extend - Smart Promotion with Comparison
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing two model versions."""
    better_version: int
    candidate_metrics: dict[str, float]
    current_metrics: dict[str, float]
    improvement: dict[str, float]  # positive = candidate is better
    should_promote: bool
    reason: str


def compare_models(
    model_name: str,
    candidate_version: int,
    current_version: int,
    metric_name: str = "f1",
    higher_is_better: bool = True,
    min_improvement: float = 0.0
) -> ComparisonResult:
    """Compare two model versions to decide if promotion is warranted.

    This prevents accidentally promoting a worse model to production.

    Args:
        model_name: Name of the registered model.
        candidate_version: Version being considered for promotion.
        current_version: Version currently in target stage.
        metric_name: Primary metric to compare.
        higher_is_better: If True, higher metric values are better.
        min_improvement: Minimum improvement required (as fraction, e.g., 0.01 = 1%).

    Returns:
        ComparisonResult with comparison details and recommendation.

    Example:
        >>> result = compare_models("sentiment-model", candidate_version=5, current_version=3)
        >>> if result.should_promote:
        ...     promote_model("sentiment-model", version=5, to_stage=ModelStage.PRODUCTION)
    """
    # TODO: Implement model comparison
    #
    # Steps:
    # 1. Get metrics for both versions from MLflow
    # 2. Calculate improvement (candidate - current) or ratio
    # 3. Determine if candidate meets promotion criteria
    # 4. Return detailed comparison result

    raise NotImplementedError("Implement compare_models()")


def smart_promote(
    model_name: str,
    version: int,
    to_stage: ModelStage,
    require_improvement: bool = True,
    metric_name: str = "f1",
    min_improvement: float = 0.0
) -> PromotionResult:
    """Promote a model only if it's better than the current one.

    Combines comparison and promotion into a single safe operation.

    Args:
        model_name: Name of the registered model.
        version: Version to promote.
        to_stage: Target stage.
        require_improvement: If True, only promote if better than current.
        metric_name: Metric to use for comparison.
        min_improvement: Minimum improvement required.

    Returns:
        PromotionResult with details.
    """
    # TODO: Implement smart promotion
    #
    # Steps:
    # 1. Get current model in target stage (if any)
    # 2. If require_improvement and current exists, compare
    # 3. Only promote if comparison passes
    # 4. Return result with comparison details in message

    raise NotImplementedError("Implement smart_promote()")


# =============================================================================
# Mock classes for testing without MLflow
# =============================================================================

@dataclass
class MockModelVersion:
    """Mock MLflow ModelVersion for testing."""
    name: str
    version: str
    current_stage: str
    run_id: str = "mock_run_123"
    creation_timestamp: int = 1704067200000


class MockMlflowClient:
    """Mock MLflow client for testing."""

    def __init__(self):
        self.models = {}
        self.transitions = []

    def create_registered_model(self, name: str):
        self.models[name] = {}

    def create_model_version(self, name: str, source: str, run_id: str) -> MockModelVersion:
        if name not in self.models:
            self.models[name] = {}
        version = len(self.models[name]) + 1
        mv = MockModelVersion(name=name, version=str(version), current_stage="None", run_id=run_id)
        self.models[name][version] = mv
        return mv

    def transition_model_version_stage(self, name: str, version: str, stage: str) -> MockModelVersion:
        v = int(version)
        self.models[name][v].current_stage = stage
        self.transitions.append((name, version, stage))
        return self.models[name][v]

    def get_model_version(self, name: str, version: str) -> MockModelVersion:
        v = int(version)
        if name not in self.models or v not in self.models[name]:
            raise ValueError(f"Model {name} version {version} not found")
        return self.models[name][v]


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Test stage validation
    print("Testing validate_stage_transition...")
    # print(validate_stage_transition(ModelStage.STAGING, ModelStage.PRODUCTION))  # True
    # print(validate_stage_transition(ModelStage.NONE, ModelStage.PRODUCTION))  # False

    # Test mock client
    print("\nTesting with mock client...")
    mock = MockMlflowClient()
    mock.create_registered_model("test-model")
    v1 = mock.create_model_version("test-model", "source", "run1")
    print(f"Created version: {v1.version}, stage: {v1.current_stage}")


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Conceptual) ===
For promote_model:
- MlflowClient().transition_model_version_stage(name, version, stage)
- Get current stage: client.get_model_version(name, str(version)).current_stage
- Search for models in a stage: client.search_model_versions(f"name='{model_name}'")

For validate_stage_transition:
- Just check: to_stage in VALID_TRANSITIONS.get(from_stage, [])

For fixed_load_model:
- Use try/except with MlflowException and ConnectionError
- Model URI format: f"models:/{model_name}/{stage}" or f"models:/{model_name}/{version}"

=== HINT 2 (More specific) ===
For archive_models_in_stage:
```python
versions = client.search_model_versions(f"name='{model_name}'")
archived = []
for v in versions:
    if v.current_stage == stage.value:
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage=ModelStage.ARCHIVED.value
        )
        archived.append(int(v.version))
return archived
```

For fixed_load_model:
```python
try:
    return mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
except MlflowException:
    if fallback_to_staging:
        try:
            return mlflow.pyfunc.load_model(f"models:/{model_name}/Staging")
        except MlflowException:
            pass
    if fallback_to_latest:
        try:
            return mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        except MlflowException:
            pass
    return None
```

=== HINT 3 (Nearly complete solution) ===
def promote_model(...) -> PromotionResult:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    try:
        model_version = client.get_model_version(model_name, str(version))
        current_stage = ModelStage(model_version.current_stage)
    except MlflowException as e:
        return PromotionResult(
            model_name=model_name, version=version,
            from_stage="unknown", to_stage=to_stage.value,
            success=False, message=f"Model not found: {e}"
        )

    if validate_transition:
        if not validate_stage_transition(current_stage, to_stage):
            return PromotionResult(
                model_name=model_name, version=version,
                from_stage=current_stage.value, to_stage=to_stage.value,
                success=False,
                message=f"Invalid transition: {current_stage.value} -> {to_stage.value}"
            )

    if archive_existing and to_stage in [ModelStage.STAGING, ModelStage.PRODUCTION]:
        archive_models_in_stage(client, model_name, to_stage)

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=to_stage.value
    )

    return PromotionResult(
        model_name=model_name, version=version,
        from_stage=current_stage.value, to_stage=to_stage.value,
        success=True, message=f"Successfully promoted to {to_stage.value}"
    )
"""
