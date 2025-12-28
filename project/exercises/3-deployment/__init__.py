"""
Level 3: Deployment Exercises

This module contains exercises for building production ML deployment skills:
- 3.1: Model Serving Options (batch vs real-time)
- 3.2: FastAPI Basics (APIs, validation, health checks)
- 3.3: Containerization Deep Dive (optimization, multi-stage builds)
- 3.4: Cloud Deployment (configuration, scaling, cost estimation)

Exercise Distribution:
- 40% "Fix this issue" - debug and fix broken code
- 40% Write & verify - implement from scratch
- 20% Design decisions - analyze requirements and recommend approaches

Run tests with: pytest test_level_3.py -v
"""

from .exercise_3_1 import (
    BatchPredictor,
    FastRealTimePredictor,
    ServingRequirements,
    SlowRealTimePredictor,
    recommend_serving_strategy,
)
from .exercise_3_2 import (
    APIError,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BuggyAPI,
    FixedAPI,
    HealthCheckResponse,
    HealthStatus,
    HealthyAPI,
    PredictionRequest,
    PredictionResponse,
    SentimentLabel,
    predict_sentiment,
)
from .exercise_3_3 import (
    BuildRequirements,
    DockerAnalysisResult,
    DockerfileIssue,
    MultiStageBuildConfig,
    analyze_bloated_container,
    design_multi_stage_build,
    extract_imports,
    optimize_requirements,
    parse_requirements,
)
from .exercise_3_4 import (
    CloudRunConfig,
    CloudRunConfigError,
    CostEstimate,
    FailingDeploymentConfig,
    ProductionReadyConfig,
    TrafficPattern,
    compare_deployment_options,
    diagnose_failing_deployment,
    estimate_cloud_costs,
    format_memory,
    generate_cloud_run_config,
    parse_memory,
)

__all__ = [
    # 3.1
    "BatchPredictor",
    "SlowRealTimePredictor",
    "FastRealTimePredictor",
    "ServingRequirements",
    "recommend_serving_strategy",
    # 3.2
    "SentimentLabel",
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResponse",
    "BatchPredictionResponse",
    "predict_sentiment",
    "BuggyAPI",
    "APIError",
    "FixedAPI",
    "HealthStatus",
    "HealthCheckResponse",
    "HealthyAPI",
    # 3.3
    "optimize_requirements",
    "parse_requirements",
    "extract_imports",
    "DockerfileIssue",
    "DockerAnalysisResult",
    "analyze_bloated_container",
    "BuildRequirements",
    "MultiStageBuildConfig",
    "design_multi_stage_build",
    # 3.4
    "CloudRunConfig",
    "CloudRunConfigError",
    "generate_cloud_run_config",
    "parse_memory",
    "format_memory",
    "FailingDeploymentConfig",
    "ProductionReadyConfig",
    "diagnose_failing_deployment",
    "TrafficPattern",
    "CostEstimate",
    "estimate_cloud_costs",
    "compare_deployment_options",
]
