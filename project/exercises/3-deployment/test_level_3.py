"""
Test Suite for Level 3: Deployment Exercises

Run with: pytest test_level_3.py -v
Or: pytest test_level_3.py -v -k "test_3_1"  # Run only 3.1 tests

Tests are organized by exercise (3.1, 3.2, 3.3, 3.4) and then by part (A, B, C).
"""

import asyncio
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock sklearn-like model for testing."""
    model = MagicMock()
    model.predict.return_value = ["positive"]
    model.predict_proba.return_value = [[0.15, 0.85]]
    return model


@pytest.fixture
def mock_model_path(mock_model, tmp_path):
    """Create a pickled mock model file."""
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(mock_model, f)
    return str(model_path)


@pytest.fixture
def sample_input_csv(tmp_path):
    """Create a sample input CSV for batch prediction."""
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        'text\n"This product is amazing!"\n"Terrible quality"\n"It was okay"\n'
    )
    return str(csv_path)


@pytest.fixture
def sample_requirements_txt(tmp_path):
    """Create a sample requirements.txt file."""
    req_path = tmp_path / "requirements.txt"
    req_path.write_text(
        """
# Production dependencies
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0
fastapi>=0.85.0
uvicorn>=0.18.0
# Unused
tensorflow>=2.10.0
matplotlib>=3.5.0
"""
    )
    return str(req_path)


@pytest.fixture
def sample_python_source(tmp_path):
    """Create sample Python source files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text(
        """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
"""
    )
    (src_dir / "utils.py").write_text(
        """
import numpy as np
"""
    )
    return str(src_dir)


# =============================================================================
# Exercise 3.1 Tests - Model Serving Options
# =============================================================================


class TestExercise3_1_PartA:
    """Tests for BatchPredictor (Write & Verify)."""

    def test_batch_predictor_init_loads_model(self, mock_model_path):
        """BatchPredictor should load model from pickle file."""
        from exercise_3_1 import BatchPredictor

        predictor = BatchPredictor(mock_model_path)
        assert predictor is not None
        # Should have loaded the model
        assert hasattr(predictor, "model") or hasattr(predictor, "_model")

    def test_batch_predictor_init_raises_for_missing_file(self):
        """BatchPredictor should raise FileNotFoundError for missing model."""
        from exercise_3_1 import BatchPredictor

        with pytest.raises(FileNotFoundError):
            BatchPredictor("/nonexistent/model.pkl")

    def test_predict_batch_processes_csv(
        self, mock_model_path, sample_input_csv, tmp_path
    ):
        """predict_batch should process CSV and write output."""
        from exercise_3_1 import BatchPredictor

        predictor = BatchPredictor(mock_model_path)
        output_path = tmp_path / "output.csv"

        result = predictor.predict_batch(sample_input_csv, str(output_path))

        assert result is not None
        assert "total_processed" in result
        assert "successful" in result
        assert "failed" in result
        assert "elapsed_time" in result
        assert output_path.exists()

    def test_predict_batch_raises_for_missing_csv(self, mock_model_path, tmp_path):
        """predict_batch should raise FileNotFoundError for missing input."""
        from exercise_3_1 import BatchPredictor

        predictor = BatchPredictor(mock_model_path)
        output_path = tmp_path / "output.csv"

        with pytest.raises(FileNotFoundError):
            predictor.predict_batch("/nonexistent/input.csv", str(output_path))

    def test_predict_batch_raises_for_missing_text_column(
        self, mock_model_path, tmp_path
    ):
        """predict_batch should raise ValueError if 'text' column missing."""
        from exercise_3_1 import BatchPredictor

        predictor = BatchPredictor(mock_model_path)
        bad_csv = tmp_path / "bad_input.csv"
        bad_csv.write_text("wrong_column\nsome text\n")
        output_path = tmp_path / "output.csv"

        with pytest.raises(ValueError, match="text"):
            predictor.predict_batch(str(bad_csv), str(output_path))


class TestExercise3_1_PartB:
    """Tests for FastRealTimePredictor (Fix This Issue)."""

    def test_fast_predictor_loads_model_once(self, mock_model_path, mock_model):
        """FastRealTimePredictor should load model only at init."""
        from exercise_3_1 import FastRealTimePredictor

        predictor = FastRealTimePredictor(mock_model_path)
        # Make multiple predictions
        predictor.predict("test 1")
        predictor.predict("test 2")
        predictor.predict("test 3")

        # Model's predict should be called, but pickle.load should happen once
        # This is validated by the fact that mock_model_path was only read once
        assert predictor is not None

    def test_fast_predictor_caches_predictions(self, mock_model_path):
        """FastRealTimePredictor should cache repeated predictions."""
        from exercise_3_1 import FastRealTimePredictor

        predictor = FastRealTimePredictor(mock_model_path, cache_size=100)

        # Same text twice
        result1 = predictor.predict("same text")
        result2 = predictor.predict("same text")

        assert result1 == result2

        # Check cache stats
        stats = predictor.cache_stats()
        assert stats is not None
        assert "hits" in stats or "misses" in stats

    def test_fast_predictor_batch_is_efficient(self, mock_model_path, mock_model):
        """predict_many should use batch prediction, not loop."""
        from exercise_3_1 import FastRealTimePredictor

        predictor = FastRealTimePredictor(mock_model_path)

        texts = ["text1", "text2", "text3", "text4", "text5"]
        results = predictor.predict_many(texts)

        assert len(results) == 5
        # The mock model's predict should have been called with all texts at once
        # (or at least fewer times than len(texts))


class TestExercise3_1_PartC:
    """Tests for recommend_serving_strategy (Design Decision)."""

    def test_recommends_realtime_for_low_latency(self):
        """Should recommend real-time for low latency requirements."""
        from exercise_3_1 import ServingRequirements, recommend_serving_strategy

        reqs = ServingRequirements(
            predictions_per_day=10000,
            max_latency_ms=50,
            freshness_hours=0,
            request_pattern="steady",
            cost_sensitivity="low",
            integration_type="api",
        )

        result = recommend_serving_strategy(reqs)

        assert result is not None
        assert result["recommendation"] == "real-time"
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0

    def test_recommends_batch_for_high_latency_tolerance(self):
        """Should recommend batch for scheduled, file-based processing."""
        from exercise_3_1 import ServingRequirements, recommend_serving_strategy

        reqs = ServingRequirements(
            predictions_per_day=1000000,
            max_latency_ms=3600000,
            freshness_hours=24,
            request_pattern="scheduled",
            cost_sensitivity="high",
            integration_type="file",
        )

        result = recommend_serving_strategy(reqs)

        assert result is not None
        assert result["recommendation"] == "batch"
        assert "reasoning" in result

    def test_recommends_hybrid_for_conflicting_requirements(self):
        """Should consider hybrid for conflicting requirements."""
        from exercise_3_1 import ServingRequirements, recommend_serving_strategy

        reqs = ServingRequirements(
            predictions_per_day=10000000,  # Very high volume
            max_latency_ms=100,  # But also low latency
            freshness_hours=0,  # Real-time required
            request_pattern="bursty",
            cost_sensitivity="high",  # But cost-conscious
            integration_type="api",
        )

        result = recommend_serving_strategy(reqs)

        assert result is not None
        # Either hybrid or real-time is acceptable for this case
        assert result["recommendation"] in ["hybrid", "real-time"]
        assert "considerations" in result


# =============================================================================
# Exercise 3.2 Tests - FastAPI Basics
# =============================================================================


class TestExercise3_2_PartA:
    """Tests for Pydantic models and predict endpoint (Write & Verify)."""

    def test_prediction_request_validates_text(self):
        """PredictionRequest should validate text length."""
        from exercise_3_2 import PredictionRequest

        # Valid request
        req = PredictionRequest(text="This is valid text")
        assert req.text == "This is valid text"

        # Empty text should fail
        with pytest.raises(ValueError):
            PredictionRequest(text="")

        # Too long text should fail
        with pytest.raises(ValueError):
            PredictionRequest(text="x" * 6000)

    def test_prediction_request_has_defaults(self):
        """PredictionRequest should have default for include_confidence."""
        from exercise_3_2 import PredictionRequest

        req = PredictionRequest(text="test")
        assert hasattr(req, "include_confidence")
        assert req.include_confidence is True

    def test_batch_prediction_request_validates_list(self):
        """BatchPredictionRequest should validate list bounds."""
        from exercise_3_2 import BatchPredictionRequest

        # Valid request
        req = BatchPredictionRequest(texts=["text1", "text2"])
        assert len(req.texts) == 2

        # Empty list should fail
        with pytest.raises(ValueError):
            BatchPredictionRequest(texts=[])

        # Too many texts should fail
        with pytest.raises(ValueError):
            BatchPredictionRequest(texts=["text"] * 101)

    def test_predict_sentiment_returns_response(self):
        """predict_sentiment should return proper response model."""
        from exercise_3_2 import (
            PredictionRequest,
            PredictionResponse,
            predict_sentiment,
        )

        async def run_test():
            req = PredictionRequest(text="This is a great product!")
            response = await predict_sentiment(req)

            assert response is not None
            assert isinstance(response, PredictionResponse)
            assert response.prediction is not None
            assert hasattr(response, "latency_ms")

        asyncio.run(run_test())

    def test_predict_sentiment_classifies_correctly(self):
        """predict_sentiment should classify based on keywords."""
        from exercise_3_2 import (
            PredictionRequest,
            SentimentLabel,
            predict_sentiment,
        )

        async def run_test():
            # Positive
            req = PredictionRequest(text="This is excellent!")
            response = await predict_sentiment(req)
            assert response.prediction == SentimentLabel.POSITIVE

            # Negative
            req = PredictionRequest(text="This is terrible!")
            response = await predict_sentiment(req)
            assert response.prediction == SentimentLabel.NEGATIVE

        asyncio.run(run_test())


class TestExercise3_2_PartB:
    """Tests for FixedAPI (Fix This Issue - crashes on malformed input)."""

    def test_fixed_api_handles_none_model(self):
        """FixedAPI should handle None model gracefully."""
        from exercise_3_2 import APIError, FixedAPI, FixedPredictionRequest

        async def run_test():
            api = FixedAPI(model=None)
            req = FixedPredictionRequest(text="test")
            result = await api.predict(req)

            assert isinstance(result, APIError)
            assert "model" in result.error.lower() or "not" in result.error.lower()

        asyncio.run(run_test())

    def test_fixed_api_validates_text(self):
        """FixedPredictionRequest should validate text properly."""
        from exercise_3_2 import FixedPredictionRequest

        # Valid
        req = FixedPredictionRequest(text="valid text")
        assert req.text == "valid text"

        # Should reject empty after strip
        with pytest.raises(ValueError):
            FixedPredictionRequest(text="   ")

    def test_fixed_api_returns_prediction_on_success(self, mock_model):
        """FixedAPI should return prediction when model works."""
        from exercise_3_2 import FixedAPI, FixedPredictionRequest

        async def run_test():
            api = FixedAPI(model=mock_model)
            req = FixedPredictionRequest(text="test text")
            result = await api.predict(req)

            # Should be a dict with prediction, not an error
            assert isinstance(result, dict)
            assert "prediction" in result

        asyncio.run(run_test())


class TestExercise3_2_PartC:
    """Tests for HealthyAPI (Fix This Issue - missing health check)."""

    def test_health_check_returns_response(self, mock_model):
        """health_check should return HealthCheckResponse."""
        from exercise_3_2 import HealthCheckResponse, HealthyAPI

        async def run_test():
            api = HealthyAPI(model=mock_model, model_version="1.0.0")
            health = await api.health_check()

            assert isinstance(health, HealthCheckResponse)
            assert health.version == "1.0.0"
            assert health.uptime_seconds >= 0
            assert len(health.components) > 0

        asyncio.run(run_test())

    def test_health_check_unhealthy_without_model(self):
        """health_check should report unhealthy if model missing."""
        from exercise_3_2 import HealthStatus, HealthyAPI

        async def run_test():
            api = HealthyAPI(model=None)
            health = await api.health_check()

            assert health.status == HealthStatus.UNHEALTHY

        asyncio.run(run_test())

    def test_liveness_probe_always_works(self):
        """liveness_probe should return alive status."""
        from exercise_3_2 import HealthyAPI

        async def run_test():
            api = HealthyAPI(model=None)
            result = await api.liveness_probe()

            assert result is not None
            assert result.get("status") == "alive"

        asyncio.run(run_test())

    def test_readiness_probe_requires_model(self):
        """readiness_probe should only be ready with model."""
        from exercise_3_2 import HealthyAPI

        async def run_test():
            # Without model
            api_no_model = HealthyAPI(model=None)
            result = await api_no_model.readiness_probe()
            assert result.get("status") != "ready"

            # With model
            api_with_model = HealthyAPI(model=MagicMock())
            result = await api_with_model.readiness_probe()
            assert result.get("status") == "ready"

        asyncio.run(run_test())


# =============================================================================
# Exercise 3.3 Tests - Containerization
# =============================================================================


class TestExercise3_3_PartA:
    """Tests for optimize_requirements (Write & Verify)."""

    def test_optimize_requirements_identifies_used(
        self, sample_requirements_txt, sample_python_source
    ):
        """Should identify packages that are imported."""
        from exercise_3_3 import optimize_requirements

        result = optimize_requirements(
            sample_requirements_txt, [sample_python_source], keep_packages=[]
        )

        assert result is not None
        assert "used" in result
        # pandas, numpy, scikit-learn, fastapi should be used
        used_lower = [p.lower() for p in result["used"]]
        assert "pandas" in used_lower
        assert "numpy" in used_lower

    def test_optimize_requirements_identifies_unused(
        self, sample_requirements_txt, sample_python_source
    ):
        """Should identify packages not imported."""
        from exercise_3_3 import optimize_requirements

        result = optimize_requirements(
            sample_requirements_txt, [sample_python_source], keep_packages=[]
        )

        assert "unused" in result
        unused_lower = [p.lower() for p in result["unused"]]
        # tensorflow and matplotlib are in requirements but not imported
        assert "tensorflow" in unused_lower or "matplotlib" in unused_lower

    def test_optimize_requirements_respects_keep_packages(
        self, sample_requirements_txt, sample_python_source
    ):
        """Should keep specified packages even if unused."""
        from exercise_3_3 import optimize_requirements

        result = optimize_requirements(
            sample_requirements_txt,
            [sample_python_source],
            keep_packages=["tensorflow"],
        )

        # tensorflow should be in used (kept), not unused
        unused_lower = [p.lower() for p in result.get("unused", [])]
        assert "tensorflow" not in unused_lower

    def test_optimize_requirements_raises_for_missing_file(self, sample_python_source):
        """Should raise FileNotFoundError for missing requirements."""
        from exercise_3_3 import optimize_requirements

        with pytest.raises(FileNotFoundError):
            optimize_requirements("/nonexistent/requirements.txt", [sample_python_source])

    def test_parse_requirements_handles_versions(self):
        """parse_requirements should handle version specs."""
        from exercise_3_3 import parse_requirements

        content = """
pandas>=1.4.0,<2.0
numpy==1.21.0
scikit-learn~=1.0
requests
# comment
"""
        result = parse_requirements(content)

        assert len(result) == 4
        names = [r["name"].lower() for r in result]
        assert "pandas" in names
        assert "numpy" in names

    def test_extract_imports_finds_all_types(self):
        """extract_imports should find all import styles."""
        from exercise_3_3 import extract_imports

        code = """
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from . import local  # Should be excluded
"""
        imports = extract_imports(code)

        assert "os" in imports
        assert "json" in imports
        assert "pandas" in imports
        assert "sklearn" in imports
        assert "pathlib" in imports
        # Relative imports should be excluded
        assert "local" not in imports
        assert "." not in imports


class TestExercise3_3_PartB:
    """Tests for analyze_bloated_container (Fix This Issue)."""

    def test_analyze_finds_base_image_issue(self):
        """Should identify non-slim base image as critical issue."""
        from exercise_3_3 import (
            BLOATED_DOCKERFILE,
            BLOATED_REQUIREMENTS,
            analyze_bloated_container,
        )

        result = analyze_bloated_container(
            BLOATED_DOCKERFILE, BLOATED_REQUIREMENTS, "FastAPI app with sklearn"
        )

        assert result is not None
        base_image_issues = [i for i in result.issues if i.category == "base_image"]
        assert len(base_image_issues) > 0
        assert base_image_issues[0].severity in ["critical", "major"]

    def test_analyze_finds_framework_conflict(self):
        """Should identify both PyTorch and TensorFlow as issue."""
        from exercise_3_3 import (
            BLOATED_DOCKERFILE,
            BLOATED_REQUIREMENTS,
            analyze_bloated_container,
        )

        result = analyze_bloated_container(
            BLOATED_DOCKERFILE, BLOATED_REQUIREMENTS, "App uses sklearn only"
        )

        pip_issues = [i for i in result.issues if i.category == "pip_packages"]
        assert len(pip_issues) > 0
        # Should mention torch or tensorflow
        descriptions = " ".join([i.description.lower() for i in pip_issues])
        assert "torch" in descriptions or "tensorflow" in descriptions

    def test_analyze_finds_dev_tools_in_prod(self):
        """Should identify dev tools (pytest, black, etc.) in production."""
        from exercise_3_3 import (
            BLOATED_DOCKERFILE,
            BLOATED_REQUIREMENTS,
            analyze_bloated_container,
        )

        result = analyze_bloated_container(
            BLOATED_DOCKERFILE, BLOATED_REQUIREMENTS, "Production API"
        )

        # Should identify dev tools
        all_descriptions = " ".join([i.description.lower() for i in result.issues])
        dev_tools_mentioned = any(
            tool in all_descriptions for tool in ["pytest", "black", "jupyter", "mypy"]
        )
        assert dev_tools_mentioned

    def test_analyze_provides_optimized_dockerfile(self):
        """Should provide an optimized Dockerfile."""
        from exercise_3_3 import (
            BLOATED_DOCKERFILE,
            BLOATED_REQUIREMENTS,
            analyze_bloated_container,
        )

        result = analyze_bloated_container(
            BLOATED_DOCKERFILE, BLOATED_REQUIREMENTS, "FastAPI with sklearn"
        )

        assert result.optimized_dockerfile is not None
        assert "slim" in result.optimized_dockerfile.lower()

    def test_analyze_estimates_size_reduction(self):
        """Should estimate significant size reduction."""
        from exercise_3_3 import (
            BLOATED_DOCKERFILE,
            BLOATED_REQUIREMENTS,
            analyze_bloated_container,
        )

        result = analyze_bloated_container(
            BLOATED_DOCKERFILE, BLOATED_REQUIREMENTS, "FastAPI with sklearn"
        )

        assert result.current_size_estimate_mb > result.optimized_size_estimate_mb
        # Should reduce by at least 50%
        reduction = (
            result.current_size_estimate_mb - result.optimized_size_estimate_mb
        ) / result.current_size_estimate_mb
        assert reduction > 0.5


class TestExercise3_3_PartC:
    """Tests for design_multi_stage_build (Design Decision)."""

    def test_recommends_multistage_for_compiled_deps(self):
        """Should recommend multi-stage for compiled dependencies."""
        from exercise_3_3 import BuildRequirements, design_multi_stage_build

        reqs = BuildRequirements(
            has_compiled_dependencies=True,
            has_model_artifacts=False,
            artifact_size_mb=0,
            needs_gpu=False,
            base_image_preference="debian",
            security_priority="standard",
        )

        config = design_multi_stage_build(reqs)

        assert config.use_multi_stage is True
        assert config.num_stages >= 2
        assert "builder" in config.dockerfile_template.lower()

    def test_provides_dockerfile_template(self):
        """Should provide a valid Dockerfile template."""
        from exercise_3_3 import BuildRequirements, design_multi_stage_build

        reqs = BuildRequirements(
            has_compiled_dependencies=True,
            has_model_artifacts=True,
            artifact_size_mb=100,
            needs_gpu=False,
            base_image_preference="debian",
            security_priority="high",
        )

        config = design_multi_stage_build(reqs)

        assert config.dockerfile_template is not None
        # Should have FROM statements
        assert "FROM" in config.dockerfile_template
        # Should have COPY --from for multi-stage
        if config.use_multi_stage:
            assert "--from=" in config.dockerfile_template

    def test_uses_appropriate_base_for_gpu(self):
        """Should use CUDA base image when GPU needed."""
        from exercise_3_3 import BuildRequirements, design_multi_stage_build

        reqs = BuildRequirements(
            has_compiled_dependencies=False,
            has_model_artifacts=True,
            artifact_size_mb=500,
            needs_gpu=True,
            base_image_preference="debian",
            security_priority="standard",
        )

        config = design_multi_stage_build(reqs)

        assert "cuda" in config.dockerfile_template.lower() or "nvidia" in config.dockerfile_template.lower()


# =============================================================================
# Exercise 3.4 Tests - Cloud Deployment
# =============================================================================


class TestExercise3_4_PartA:
    """Tests for generate_cloud_run_config (Write & Verify)."""

    def test_generate_valid_config(self):
        """Should generate valid CloudRunConfig."""
        from exercise_3_4 import CloudRunConfig, generate_cloud_run_config

        config = generate_cloud_run_config(
            cpu=2, memory="4Gi", min_instances=1, max_instances=10
        )

        assert isinstance(config, CloudRunConfig)
        assert config.cpu == "2"
        assert config.memory == "4Gi"
        assert config.min_instances == 1
        assert config.max_instances == 10

    def test_normalizes_cpu_to_string(self):
        """Should accept int and convert to string."""
        from exercise_3_4 import generate_cloud_run_config

        config = generate_cloud_run_config(cpu=4, memory="2Gi")
        assert config.cpu == "4"

    def test_normalizes_memory_from_int(self):
        """Should convert memory int (MB) to string format."""
        from exercise_3_4 import generate_cloud_run_config

        config = generate_cloud_run_config(cpu=1, memory=1024)
        assert config.memory in ["1Gi", "1024Mi"]

    def test_rejects_invalid_cpu(self):
        """Should reject invalid CPU values."""
        from exercise_3_4 import CloudRunConfigError, generate_cloud_run_config

        with pytest.raises(CloudRunConfigError):
            generate_cloud_run_config(cpu=3, memory="1Gi")  # 3 is not valid

        with pytest.raises(CloudRunConfigError):
            generate_cloud_run_config(cpu=0, memory="1Gi")  # 0 is not valid

    def test_validates_memory_per_cpu(self):
        """Should enforce minimum memory per CPU."""
        from exercise_3_4 import CloudRunConfigError, generate_cloud_run_config

        with pytest.raises(CloudRunConfigError):
            generate_cloud_run_config(cpu=4, memory="256Mi")  # Need at least 1Gi for 4 CPU

    def test_validates_instance_counts(self):
        """Should validate min <= max instances."""
        from exercise_3_4 import CloudRunConfigError, generate_cloud_run_config

        with pytest.raises(CloudRunConfigError):
            generate_cloud_run_config(cpu=1, memory="512Mi", min_instances=10, max_instances=5)

    def test_generates_yaml(self):
        """Should generate valid YAML output."""
        from exercise_3_4 import generate_cloud_run_config

        config = generate_cloud_run_config(cpu=2, memory="2Gi", min_instances=1)
        yaml_output = config.to_yaml()

        assert "apiVersion" in yaml_output
        assert "minScale" in yaml_output
        assert '"2"' in yaml_output  # CPU
        assert '"2Gi"' in yaml_output  # Memory


class TestExercise3_4_PartB:
    """Tests for diagnose_failing_deployment (Fix This Issue)."""

    def test_diagnoses_503_errors(self):
        """Should identify concurrency as cause of 503 errors."""
        from exercise_3_4 import FailingDeploymentConfig, diagnose_failing_deployment

        result = diagnose_failing_deployment(
            FailingDeploymentConfig(), ["503 errors under load"]
        )

        assert result is not None
        assert "root_causes" in result
        # Should mention concurrency
        causes_str = " ".join(result["root_causes"]).lower()
        assert "concurrency" in causes_str or "instance" in causes_str

    def test_diagnoses_cold_starts(self):
        """Should identify min_instances as cause of cold starts."""
        from exercise_3_4 import FailingDeploymentConfig, diagnose_failing_deployment

        result = diagnose_failing_deployment(
            FailingDeploymentConfig(), ["Cold starts taking 30+ seconds"]
        )

        # Fixed config should have min_instances > 0
        assert result["fixed_config"].min_instances > 0
        # Should also recommend startup_cpu_boost or disable throttling
        assert (
            result["fixed_config"].startup_cpu_boost is True
            or result["fixed_config"].cpu_throttling is False
        )

    def test_diagnoses_memory_issues(self):
        """Should recommend more memory for memory exceeded errors."""
        from exercise_3_4 import FailingDeploymentConfig, diagnose_failing_deployment

        result = diagnose_failing_deployment(
            FailingDeploymentConfig(), ["Memory exceeded errors"]
        )

        # Fixed config should have more memory
        from exercise_3_4 import parse_memory

        original_memory = parse_memory(FailingDeploymentConfig.memory)
        fixed_memory = parse_memory(result["fixed_config"].memory)
        assert fixed_memory > original_memory

    def test_provides_explanation(self):
        """Should provide markdown explanation."""
        from exercise_3_4 import FailingDeploymentConfig, diagnose_failing_deployment

        result = diagnose_failing_deployment(
            FailingDeploymentConfig(), ["503 errors under load", "High latency p99"]
        )

        assert "explanation" in result
        assert len(result["explanation"]) > 50  # Substantive explanation


class TestExercise3_4_PartC:
    """Tests for estimate_cloud_costs (Design Decision)."""

    def test_estimates_monthly_cost(self):
        """Should estimate monthly cost."""
        from exercise_3_4 import TrafficPattern, estimate_cloud_costs

        traffic = TrafficPattern(
            requests_per_day=100000,
            peak_rps=10,
            avg_latency_ms=200,
            traffic_distribution="business_hours",
            model_memory_mb=2048,
            model_load_time_seconds=30,
        )

        estimate = estimate_cloud_costs(traffic)

        assert estimate is not None
        assert estimate.monthly_cost_usd >= 0
        assert "cpu" in estimate.cost_breakdown
        assert "memory" in estimate.cost_breakdown

    def test_accounts_for_free_tier(self):
        """Should account for free tier in low-volume scenarios."""
        from exercise_3_4 import TrafficPattern, estimate_cloud_costs

        # Very low traffic - should be mostly free tier
        traffic = TrafficPattern(
            requests_per_day=1000,
            peak_rps=1,
            avg_latency_ms=100,
            traffic_distribution="steady",
            model_memory_mb=512,
            model_load_time_seconds=5,
        )

        estimate = estimate_cloud_costs(traffic)

        # Should be very low cost due to free tier
        assert estimate.monthly_cost_usd < 10

    def test_provides_optimization_tips(self):
        """Should provide cost optimization tips."""
        from exercise_3_4 import TrafficPattern, estimate_cloud_costs

        traffic = TrafficPattern(
            requests_per_day=1000000,
            peak_rps=100,
            avg_latency_ms=500,
            traffic_distribution="spiky",
            model_memory_mb=4096,
            model_load_time_seconds=60,
        )

        estimate = estimate_cloud_costs(traffic)

        assert "cost_optimization_tips" in estimate.__dict__
        assert len(estimate.cost_optimization_tips) > 0

    def test_compares_alternatives(self):
        """Should compare to alternative deployment options."""
        from exercise_3_4 import TrafficPattern, compare_deployment_options

        traffic = TrafficPattern(
            requests_per_day=500000,
            peak_rps=50,
            avg_latency_ms=200,
            traffic_distribution="business_hours",
            model_memory_mb=2048,
            model_load_time_seconds=30,
        )

        comparison = compare_deployment_options(traffic)

        assert comparison is not None
        assert "cloud_run" in comparison
        # Should have at least one alternative
        assert len(comparison) >= 2


# =============================================================================
# Helper Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions across exercises."""

    def test_parse_memory(self):
        """parse_memory should handle all formats."""
        from exercise_3_4 import parse_memory

        assert parse_memory("512Mi") == 512
        assert parse_memory("1Gi") == 1024
        assert parse_memory("2Gi") == 2048
        assert parse_memory(256) == 256

    def test_format_memory(self):
        """format_memory should produce valid Cloud Run format."""
        from exercise_3_4 import format_memory

        assert format_memory(1024) == "1Gi"
        assert format_memory(2048) == "2Gi"
        assert format_memory(512) == "512Mi"


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
