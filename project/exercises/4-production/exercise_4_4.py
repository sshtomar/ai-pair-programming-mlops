"""
Exercise 4.4: Capstone - Complete MLOps Pipeline
Difficulty: **** (Integration challenge)
Topic: End-to-end MLOps pipeline design and implementation

Instructions:
This capstone exercise brings together concepts from all previous lessons.
You'll create an MLOpsPipeline class that orchestrates the entire lifecycle
of an ML model: from data validation through deployment and monitoring.

This is a larger exercise designed to test your understanding of how all
the pieces fit together in a production system.

Estimated time: 60-90 minutes

Hints available: Type /hint 1, /hint 2, /hint 3, /hint 4 for progressive help
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict, Protocol, Any
import json


# =============================================================================
# PIPELINE COMPONENTS - Protocols for dependency injection
# =============================================================================


class DataValidator(Protocol):
    """Protocol for data validation components."""

    def validate(self, data_path: Path) -> "ValidationReport":
        """Validate data quality and schema."""
        ...


class ExperimentTracker(Protocol):
    """Protocol for experiment tracking (like MLflow)."""

    def start_run(self, run_name: str) -> str:
        """Start a new experiment run, return run_id."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics."""
        ...

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log model artifact."""
        ...

    def end_run(self) -> None:
        """End current run."""
        ...


class ModelRegistry(Protocol):
    """Protocol for model registry (like MLflow Model Registry)."""

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Register a model, return version."""
        ...

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: str,
    ) -> None:
        """Transition model to new stage (staging/production)."""
        ...

    def get_latest_version(self, name: str, stage: str) -> str | None:
        """Get latest version in a stage."""
        ...


class ModelTrainer(Protocol):
    """Protocol for model training."""

    def train(
        self,
        train_data_path: Path,
        val_data_path: Path,
        hyperparameters: dict[str, Any],
    ) -> "TrainingResult":
        """Train a model, return results."""
        ...


class DeploymentManager(Protocol):
    """Protocol for deployment management."""

    def generate_config(
        self,
        model_version: str,
        environment: str,
    ) -> "DeploymentConfig":
        """Generate deployment configuration."""
        ...

    def deploy(self, config: "DeploymentConfig") -> "DeploymentResult":
        """Deploy the model."""
        ...


class MonitoringSetup(Protocol):
    """Protocol for monitoring configuration."""

    def configure_alerts(
        self,
        model_name: str,
        thresholds: dict[str, float],
    ) -> list[str]:
        """Configure monitoring alerts, return alert IDs."""
        ...

    def configure_dashboards(
        self,
        model_name: str,
    ) -> str:
        """Configure dashboards, return dashboard URL."""
        ...


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class PipelineStage(Enum):
    """Stages of the MLOps pipeline."""
    DATA_VALIDATION = "data_validation"
    TRAINING = "training"
    MODEL_VALIDATION = "model_validation"
    REGISTRATION = "registration"
    DEPLOYMENT_CONFIG = "deployment_config"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationStatus(Enum):
    """Status of validation checks."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationReport:
    """Result of data validation."""
    status: ValidationStatus
    checks: dict[str, bool]
    warnings: list[str]
    errors: list[str]
    statistics: dict[str, Any]


@dataclass
class TrainingResult:
    """Result of model training."""
    model_path: Path
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    training_duration_seconds: float


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_version: str
    environment: str
    replicas: int
    resources: dict[str, str]
    health_check_path: str
    rollout_strategy: str


@dataclass
class DeploymentResult:
    """Result of deployment."""
    success: bool
    endpoint_url: str | None
    deployment_id: str | None
    error: str | None


class PipelineResult(TypedDict):
    """Final result of pipeline execution."""
    success: bool
    stage_reached: str
    run_id: str | None
    model_version: str | None
    deployment_url: str | None
    dashboard_url: str | None
    duration_seconds: float
    errors: list[str]
    warnings: list[str]


@dataclass
class PipelineConfig:
    """Configuration for the MLOps pipeline."""
    # Data paths
    train_data_path: Path
    val_data_path: Path
    test_data_path: Path

    # Model configuration
    model_name: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Validation thresholds
    min_accuracy: float = 0.85
    max_latency_ms: float = 100.0
    min_data_rows: int = 1000

    # Deployment configuration
    target_environment: str = "staging"
    replicas: int = 2

    # Monitoring thresholds
    alert_latency_p99_ms: float = 200.0
    alert_error_rate: float = 0.01
    alert_accuracy_drop: float = 0.05


# =============================================================================
# MAIN EXERCISE: MLOpsPipeline Class
# =============================================================================


class MLOpsPipeline:
    """
    Orchestrates the complete ML lifecycle.

    This class ties together all MLOps components into a cohesive pipeline
    that can be executed end-to-end or stage-by-stage.

    The pipeline stages are:
    1. Data Validation - Ensure training data quality
    2. Training - Train model with experiment tracking
    3. Model Validation - Verify model meets quality bar
    4. Registration - Register model in model registry
    5. Deployment Config - Generate deployment configuration
    6. Deployment - Deploy to target environment
    7. Monitoring - Configure alerts and dashboards

    Example usage:
        >>> pipeline = MLOpsPipeline(
        ...     config=PipelineConfig(
        ...         train_data_path=Path("data/train.csv"),
        ...         val_data_path=Path("data/val.csv"),
        ...         test_data_path=Path("data/test.csv"),
        ...         model_name="sentiment-classifier",
        ...     ),
        ...     data_validator=MyDataValidator(),
        ...     experiment_tracker=MLflowTracker(),
        ...     model_trainer=SentimentTrainer(),
        ...     model_registry=MLflowRegistry(),
        ...     deployment_manager=K8sDeployer(),
        ...     monitoring_setup=PrometheusSetup(),
        ... )
        >>> result = pipeline.run()
        >>> if result['success']:
        ...     print(f"Model deployed to {result['deployment_url']}")
    """

    def __init__(
        self,
        config: PipelineConfig,
        data_validator: DataValidator,
        experiment_tracker: ExperimentTracker,
        model_trainer: ModelTrainer,
        model_registry: ModelRegistry,
        deployment_manager: DeploymentManager,
        monitoring_setup: MonitoringSetup,
    ):
        """
        Initialize the pipeline with all required components.

        Args:
            config: Pipeline configuration
            data_validator: Component for data validation
            experiment_tracker: Component for experiment tracking
            model_trainer: Component for model training
            model_registry: Component for model registry
            deployment_manager: Component for deployment
            monitoring_setup: Component for monitoring setup
        """
        self.config = config
        self.data_validator = data_validator
        self.experiment_tracker = experiment_tracker
        self.model_trainer = model_trainer
        self.model_registry = model_registry
        self.deployment_manager = deployment_manager
        self.monitoring_setup = monitoring_setup

        # Pipeline state
        self._current_stage: PipelineStage = PipelineStage.DATA_VALIDATION
        self._run_id: str | None = None
        self._model_version: str | None = None
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._start_time: datetime | None = None

    def run(self) -> PipelineResult:
        """
        Execute the complete pipeline.

        This method runs all stages in sequence. If any stage fails,
        the pipeline stops and returns partial results.

        Returns:
            PipelineResult with final status and all outputs

        The implementation should:
        1. Track start time
        2. Run each stage in order
        3. Stop on failure, continuing on warnings
        4. Collect all errors and warnings
        5. Return comprehensive result
        """
        # TODO: Implement the full pipeline execution
        #
        # Pseudocode:
        # self._start_time = datetime.now()
        # try:
        #     self._run_data_validation()
        #     self._run_training()
        #     self._run_model_validation()
        #     self._run_registration()
        #     self._run_deployment_config()
        #     self._run_deployment()
        #     self._run_monitoring_setup()
        #     self._current_stage = PipelineStage.COMPLETED
        # except PipelineError as e:
        #     self._errors.append(str(e))
        #     self._current_stage = PipelineStage.FAILED
        # finally:
        #     return self._build_result()

        # Placeholder - implement your solution
        return PipelineResult(
            success=False,
            stage_reached="not_started",
            run_id=None,
            model_version=None,
            deployment_url=None,
            dashboard_url=None,
            duration_seconds=0.0,
            errors=["Not implemented"],
            warnings=[],
        )

    def run_stage(self, stage: PipelineStage) -> bool:
        """
        Run a specific pipeline stage.

        This allows running individual stages for testing or recovery.

        Args:
            stage: The stage to run

        Returns:
            True if stage succeeded, False otherwise
        """
        # TODO: Implement stage-specific execution
        # Map stage to appropriate method and call it
        stage_methods = {
            PipelineStage.DATA_VALIDATION: self._run_data_validation,
            PipelineStage.TRAINING: self._run_training,
            PipelineStage.MODEL_VALIDATION: self._run_model_validation,
            PipelineStage.REGISTRATION: self._run_registration,
            PipelineStage.DEPLOYMENT_CONFIG: self._run_deployment_config,
            PipelineStage.DEPLOYMENT: self._run_deployment,
            PipelineStage.MONITORING: self._run_monitoring_setup,
        }

        if stage not in stage_methods:
            self._errors.append(f"Unknown stage: {stage}")
            return False

        try:
            stage_methods[stage]()
            return True
        except Exception as e:
            self._errors.append(f"Stage {stage.value} failed: {str(e)}")
            return False

    def _run_data_validation(self) -> None:
        """
        Stage 1: Validate training data.

        Should check:
        - Data files exist
        - Schema is correct
        - Data quality meets thresholds
        - Sufficient samples for training

        Raises:
            PipelineError if validation fails
        """
        # TODO: Implement data validation
        # 1. Validate training data
        # 2. Validate validation data
        # 3. Check minimum row requirements
        # 4. Collect warnings for borderline issues
        pass

    def _run_training(self) -> None:
        """
        Stage 2: Train the model with experiment tracking.

        Should:
        - Start experiment run
        - Log hyperparameters
        - Train model
        - Log metrics
        - Log model artifact
        - End experiment run

        Raises:
            PipelineError if training fails
        """
        # TODO: Implement training with tracking
        pass

    def _run_model_validation(self) -> None:
        """
        Stage 3: Validate model meets quality requirements.

        Should check:
        - Accuracy >= min_accuracy
        - Latency <= max_latency_ms
        - Prediction distribution is reasonable

        Raises:
            PipelineError if model doesn't meet requirements
        """
        # TODO: Implement model validation
        # Use concepts from exercise_4_1.validate_model_before_deploy
        pass

    def _run_registration(self) -> None:
        """
        Stage 4: Register model in model registry.

        Should:
        - Register model with metadata
        - Assign version
        - Tag with experiment run info

        Raises:
            PipelineError if registration fails
        """
        # TODO: Implement model registration
        pass

    def _run_deployment_config(self) -> None:
        """
        Stage 5: Generate deployment configuration.

        Should:
        - Generate config for target environment
        - Validate configuration
        - Store for deployment stage

        Raises:
            PipelineError if config generation fails
        """
        # TODO: Implement deployment config generation
        pass

    def _run_deployment(self) -> None:
        """
        Stage 6: Deploy the model.

        Should:
        - Deploy to target environment
        - Wait for health check
        - Verify deployment succeeded

        Raises:
            PipelineError if deployment fails
        """
        # TODO: Implement deployment
        pass

    def _run_monitoring_setup(self) -> None:
        """
        Stage 7: Configure monitoring and alerting.

        Should:
        - Configure latency alerts
        - Configure error rate alerts
        - Configure drift detection alerts
        - Set up dashboard

        Raises:
            PipelineError if monitoring setup fails
        """
        # TODO: Implement monitoring setup
        # Use concepts from exercise_4_2 (AlertRule) and exercise_4_3 (drift)
        pass

    def _build_result(self) -> PipelineResult:
        """Build the final pipeline result."""
        duration = 0.0
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()

        return PipelineResult(
            success=self._current_stage == PipelineStage.COMPLETED,
            stage_reached=self._current_stage.value,
            run_id=self._run_id,
            model_version=self._model_version,
            deployment_url=None,  # TODO: Fill from deployment result
            dashboard_url=None,  # TODO: Fill from monitoring setup
            duration_seconds=duration,
            errors=self._errors,
            warnings=self._warnings,
        )

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "current_stage": self._current_stage.value,
            "run_id": self._run_id,
            "model_version": self._model_version,
            "errors": self._errors,
            "warnings": self._warnings,
        }


# =============================================================================
# MOCK IMPLEMENTATIONS FOR TESTING
# =============================================================================
#
# These are simple mock implementations to help you test your pipeline.
# In production, these would be real integrations with MLflow, Kubernetes, etc.


class MockDataValidator:
    """Mock data validator for testing."""

    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass

    def validate(self, data_path: Path) -> ValidationReport:
        if not self.should_pass:
            return ValidationReport(
                status=ValidationStatus.FAILED,
                checks={"exists": False},
                warnings=[],
                errors=["Data file not found"],
                statistics={},
            )
        return ValidationReport(
            status=ValidationStatus.PASSED,
            checks={"exists": True, "schema": True, "quality": True},
            warnings=[],
            errors=[],
            statistics={"rows": 10000, "columns": 3},
        )


class MockExperimentTracker:
    """Mock experiment tracker for testing."""

    def __init__(self):
        self.runs: dict[str, dict] = {}
        self._current_run: str | None = None

    def start_run(self, run_name: str) -> str:
        run_id = f"run_{len(self.runs) + 1}"
        self.runs[run_id] = {"name": run_name, "params": {}, "metrics": {}}
        self._current_run = run_id
        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self._current_run:
            self.runs[self._current_run]["params"].update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._current_run:
            self.runs[self._current_run]["metrics"].update(metrics)

    def log_model(self, model: Any, artifact_path: str) -> None:
        if self._current_run:
            self.runs[self._current_run]["model_path"] = artifact_path

    def end_run(self) -> None:
        self._current_run = None


class MockModelTrainer:
    """Mock model trainer for testing."""

    def __init__(self, accuracy: float = 0.88):
        self.accuracy = accuracy

    def train(
        self,
        train_data_path: Path,
        val_data_path: Path,
        hyperparameters: dict[str, Any],
    ) -> TrainingResult:
        return TrainingResult(
            model_path=Path("models/mock_model.pkl"),
            metrics={"accuracy": self.accuracy, "f1": self.accuracy - 0.02},
            hyperparameters=hyperparameters,
            training_duration_seconds=120.0,
        )


class MockModelRegistry:
    """Mock model registry for testing."""

    def __init__(self):
        self.models: dict[str, list[dict]] = {}

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        if name not in self.models:
            self.models[name] = []
        version = str(len(self.models[name]) + 1)
        self.models[name].append({
            "uri": model_uri,
            "version": version,
            "tags": tags or {},
            "stage": "None",
        })
        return version

    def transition_stage(self, name: str, version: str, stage: str) -> None:
        if name in self.models:
            for model in self.models[name]:
                if model["version"] == version:
                    model["stage"] = stage

    def get_latest_version(self, name: str, stage: str) -> str | None:
        if name not in self.models:
            return None
        for model in reversed(self.models[name]):
            if model["stage"] == stage:
                return model["version"]
        return None


class MockDeploymentManager:
    """Mock deployment manager for testing."""

    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.deployments: list[dict] = []

    def generate_config(
        self,
        model_version: str,
        environment: str,
    ) -> DeploymentConfig:
        return DeploymentConfig(
            model_version=model_version,
            environment=environment,
            replicas=2,
            resources={"cpu": "500m", "memory": "512Mi"},
            health_check_path="/health",
            rollout_strategy="rolling",
        )

    def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        if not self.should_succeed:
            return DeploymentResult(
                success=False,
                endpoint_url=None,
                deployment_id=None,
                error="Deployment failed: container crashed",
            )
        deployment_id = f"deploy_{len(self.deployments) + 1}"
        self.deployments.append({"id": deployment_id, "config": config})
        return DeploymentResult(
            success=True,
            endpoint_url=f"https://{config.environment}.example.com/api/v1",
            deployment_id=deployment_id,
            error=None,
        )


class MockMonitoringSetup:
    """Mock monitoring setup for testing."""

    def __init__(self):
        self.alerts: list[dict] = []
        self.dashboards: list[dict] = []

    def configure_alerts(
        self,
        model_name: str,
        thresholds: dict[str, float],
    ) -> list[str]:
        alert_ids = []
        for metric, threshold in thresholds.items():
            alert_id = f"alert_{len(self.alerts) + 1}"
            self.alerts.append({
                "id": alert_id,
                "model": model_name,
                "metric": metric,
                "threshold": threshold,
            })
            alert_ids.append(alert_id)
        return alert_ids

    def configure_dashboards(self, model_name: str) -> str:
        dashboard_id = f"dash_{len(self.dashboards) + 1}"
        self.dashboards.append({"id": dashboard_id, "model": model_name})
        return f"https://grafana.example.com/d/{dashboard_id}"


# =============================================================================
# BONUS: Pipeline Error Handling
# =============================================================================


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(self, stage: PipelineStage, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage.value}] {message}")


class DataValidationError(PipelineError):
    """Raised when data validation fails."""

    def __init__(self, message: str):
        super().__init__(PipelineStage.DATA_VALIDATION, message)


class ModelValidationError(PipelineError):
    """Raised when model validation fails."""

    def __init__(self, message: str):
        super().__init__(PipelineStage.MODEL_VALIDATION, message)


class DeploymentError(PipelineError):
    """Raised when deployment fails."""

    def __init__(self, message: str):
        super().__init__(PipelineStage.DEPLOYMENT, message)


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Pipeline Structure) ===

The key is proper state management. Here's a skeleton:

```python
def run(self) -> PipelineResult:
    self._start_time = datetime.now()
    self._errors = []
    self._warnings = []

    try:
        # Stage 1: Data Validation
        self._current_stage = PipelineStage.DATA_VALIDATION
        self._run_data_validation()

        # Stage 2: Training
        self._current_stage = PipelineStage.TRAINING
        self._run_training()

        # ... more stages ...

        self._current_stage = PipelineStage.COMPLETED

    except PipelineError as e:
        self._errors.append(str(e))
        self._current_stage = PipelineStage.FAILED

    return self._build_result()
```

Each stage method should:
- Do its work
- Update pipeline state (run_id, model_version, etc.)
- Raise PipelineError on failure
- Add to warnings for non-fatal issues


=== HINT 2 (Stage Implementations) ===

Here's how _run_data_validation might look:

```python
def _run_data_validation(self) -> None:
    # Validate training data
    train_report = self.data_validator.validate(self.config.train_data_path)
    if train_report.status == ValidationStatus.FAILED:
        raise DataValidationError(
            f"Training data validation failed: {train_report.errors}"
        )

    # Collect warnings
    self._warnings.extend(train_report.warnings)

    # Check minimum rows
    rows = train_report.statistics.get("rows", 0)
    if rows < self.config.min_data_rows:
        raise DataValidationError(
            f"Insufficient training data: {rows} < {self.config.min_data_rows}"
        )

    # Validate validation data
    val_report = self.data_validator.validate(self.config.val_data_path)
    if val_report.status == ValidationStatus.FAILED:
        raise DataValidationError(
            f"Validation data check failed: {val_report.errors}"
        )
```


=== HINT 3 (Training Stage) ===

Here's how _run_training might look:

```python
def _run_training(self) -> None:
    # Start experiment run
    run_name = f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    self._run_id = self.experiment_tracker.start_run(run_name)

    try:
        # Log hyperparameters
        self.experiment_tracker.log_params(self.config.hyperparameters)

        # Train model
        result = self.model_trainer.train(
            self.config.train_data_path,
            self.config.val_data_path,
            self.config.hyperparameters,
        )

        # Log metrics
        self.experiment_tracker.log_metrics(result.metrics)

        # Log model
        self.experiment_tracker.log_model(
            result.model_path,
            "model"
        )

        # Store for later stages
        self._training_result = result

    finally:
        self.experiment_tracker.end_run()
```


=== HINT 4 (Complete Implementation Pattern) ===

Here's the registration and deployment pattern:

```python
def _run_registration(self) -> None:
    if not hasattr(self, '_training_result'):
        raise PipelineError(
            PipelineStage.REGISTRATION,
            "No training result available. Run training first."
        )

    # Register model
    self._model_version = self.model_registry.register_model(
        model_uri=str(self._training_result.model_path),
        name=self.config.model_name,
        tags={
            "run_id": self._run_id or "unknown",
            "accuracy": str(self._training_result.metrics.get("accuracy", 0)),
        },
    )

    # Transition to staging
    self.model_registry.transition_stage(
        self.config.model_name,
        self._model_version,
        "Staging",
    )


def _run_deployment(self) -> None:
    if not self._model_version:
        raise PipelineError(
            PipelineStage.DEPLOYMENT,
            "No model version available. Run registration first."
        )

    # Deploy
    config = self.deployment_manager.generate_config(
        self._model_version,
        self.config.target_environment,
    )
    result = self.deployment_manager.deploy(config)

    if not result.success:
        raise DeploymentError(result.error or "Deployment failed")

    self._deployment_result = result


def _run_monitoring_setup(self) -> None:
    # Configure alerts
    thresholds = {
        "latency_p99_ms": self.config.alert_latency_p99_ms,
        "error_rate": self.config.alert_error_rate,
        "accuracy_drop": self.config.alert_accuracy_drop,
    }
    self._alert_ids = self.monitoring_setup.configure_alerts(
        self.config.model_name,
        thresholds,
    )

    # Configure dashboard
    self._dashboard_url = self.monitoring_setup.configure_dashboards(
        self.config.model_name,
    )
```

Remember to update _build_result to include deployment_url and dashboard_url!
"""
