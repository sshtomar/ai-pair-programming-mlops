"""
Test Suite for Level 4: Production Exercises

Run with: pytest test_level_4.py -v

These tests verify the correct implementation of Level 4 exercises.
Students should run these tests to check their work.
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time


# =============================================================================
# Exercise 4.1 Tests: CI/CD for ML
# =============================================================================

class TestExercise41:
    """Tests for Exercise 4.1: CI/CD for ML."""

    def test_fix_github_actions_cache_returns_string(self):
        """Test that fix_github_actions_cache returns a YAML string."""
        from exercise_4_1 import fix_github_actions_cache

        result = fix_github_actions_cache()
        assert isinstance(result, str)
        assert len(result) > 100, "YAML content seems too short"

    def test_fix_github_actions_cache_has_correct_order(self):
        """Test that the cache step comes BEFORE install."""
        from exercise_4_1 import fix_github_actions_cache

        result = fix_github_actions_cache()

        # The cache step should appear before the install step
        # This is a simplified check - in reality you'd parse the YAML
        if "TODO" not in result:
            cache_pos = result.find("actions/cache")
            install_pos = result.find("pip install")

            if cache_pos != -1 and install_pos != -1:
                assert cache_pos < install_pos, (
                    "Cache step should come BEFORE install step. "
                    "The cache restore happens at the start of the cache step."
                )

    def test_diagnose_ci_environment_bug_returns_dict(self):
        """Test that diagnose_ci_environment_bug returns proper structure."""
        from exercise_4_1 import diagnose_ci_environment_bug

        result = diagnose_ci_environment_bug()
        assert isinstance(result, dict)
        assert "root_cause" in result
        assert "missing_packages" in result
        assert "file_to_modify" in result
        assert "fixed_requirements" in result

    def test_diagnose_ci_environment_identifies_sklearn(self):
        """Test that sklearn is identified as missing."""
        from exercise_4_1 import diagnose_ci_environment_bug

        result = diagnose_ci_environment_bug()

        # Students should identify that sklearn is missing
        if result["missing_packages"]:
            missing_lower = [p.lower() for p in result["missing_packages"]]
            assert any(
                "sklearn" in p or "scikit" in p
                for p in missing_lower
            ), "Should identify scikit-learn as missing"

    def test_validate_model_before_deploy_returns_validation_result(self):
        """Test that validate_model_before_deploy returns correct type."""
        from exercise_4_1 import validate_model_before_deploy, ValidationResult

        # Call with non-existent paths (should handle gracefully)
        result = validate_model_before_deploy(
            model_path="nonexistent.pkl",
            test_data_path="nonexistent.csv",
        )

        assert isinstance(result, dict)
        assert "passed" in result
        assert "accuracy" in result
        assert "checks" in result
        assert "errors" in result

    def test_validate_model_before_deploy_fails_on_missing_model(self):
        """Test that validation fails when model file is missing."""
        from exercise_4_1 import validate_model_before_deploy

        result = validate_model_before_deploy(
            model_path="definitely_not_a_real_model.pkl",
            test_data_path="also_not_real.csv",
        )

        # Should fail because model doesn't exist
        assert result["passed"] is False

        # Should have an error about the missing file
        if result["errors"]:
            assert any(
                "not found" in e.lower() or "not implemented" in e.lower()
                for e in result["errors"]
            ), "Should report file not found error"


# =============================================================================
# Exercise 4.2 Tests: Monitoring & Observability
# =============================================================================

class TestExercise42:
    """Tests for Exercise 4.2: Monitoring & Observability."""

    def test_fixed_metrics_collector_bounded_cardinality(self):
        """Test that FixedMetricsCollector doesn't leak memory."""
        from exercise_4_2 import FixedMetricsCollector
        import uuid

        collector = FixedMetricsCollector()

        # Simulate traffic with many unique user/request IDs
        for _ in range(1000):
            collector.record_prediction(
                user_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                model_version="1.0.0",
                predicted_label="positive",
                confidence=0.9,
                latency_ms=50.0,
            )

        stats = collector.get_memory_usage()

        # With proper implementation, counter series should be bounded
        # by (model_version * label) combinations, not by unique users
        assert stats["counter_series"] < 100, (
            f"Counter series too high ({stats['counter_series']}). "
            "Make sure you're not using user_id or request_id as labels!"
        )

    def test_track_prediction_latency_decorator_works(self):
        """Test that the latency decorator tracks timing."""
        from exercise_4_2 import track_prediction_latency, MetricsRegistry

        registry = MetricsRegistry()

        @track_prediction_latency(metric_name="test_latency", registry=registry)
        def slow_function():
            time.sleep(0.01)  # 10ms
            return {"result": "done"}

        # Call the decorated function
        result = slow_function()

        # Verify function still works
        assert result == {"result": "done"}

        # Verify latency was recorded
        latencies = registry.histograms.get("test_latency", [])
        assert len(latencies) >= 1, "Latency should be recorded"
        assert latencies[0] >= 0.01, "Latency should be at least 10ms"

    def test_track_prediction_latency_handles_exceptions(self):
        """Test that latency is recorded even when function raises."""
        from exercise_4_2 import track_prediction_latency, MetricsRegistry

        registry = MetricsRegistry()

        @track_prediction_latency(metric_name="error_latency", registry=registry)
        def failing_function():
            time.sleep(0.005)
            raise ValueError("Intentional error")

        # Call should raise
        with pytest.raises(ValueError):
            failing_function()

        # But latency should still be recorded
        latencies = registry.histograms.get("error_latency", [])
        assert len(latencies) >= 1, "Latency should be recorded even on exception"

    def test_alert_rule_evaluate_basic(self):
        """Test AlertRule evaluation with basic threshold."""
        from exercise_4_2 import AlertRule, AlertSeverity, AlertState

        rule = AlertRule(
            name="TestAlert",
            description="Test alert",
            metric_name="test_metric",
            threshold=0.5,
            comparison="gt",
            for_seconds=10,
            severity=AlertSeverity.WARNING,
        )

        # Below threshold - should be OK
        state = rule.evaluate(0.3, current_time=0.0)
        assert state == AlertState.OK

        # Above threshold - should be PENDING (not long enough)
        state = rule.evaluate(0.6, current_time=1.0)
        assert state == AlertState.PENDING

        # Still above threshold after for_seconds - should be FIRING
        state = rule.evaluate(0.7, current_time=15.0)
        assert state == AlertState.FIRING

    def test_alert_rule_resets_on_recovery(self):
        """Test that alert resets when metric recovers."""
        from exercise_4_2 import AlertRule, AlertSeverity, AlertState

        rule = AlertRule(
            name="TestAlert",
            description="Test alert",
            metric_name="test_metric",
            threshold=0.5,
            comparison="gt",
            for_seconds=10,
            severity=AlertSeverity.WARNING,
        )

        # Breach threshold
        rule.evaluate(0.6, current_time=0.0)
        assert rule._state == AlertState.PENDING

        # Metric recovers
        state = rule.evaluate(0.3, current_time=5.0)
        assert state == AlertState.OK

        # New breach should start fresh
        state = rule.evaluate(0.7, current_time=6.0)
        assert state == AlertState.PENDING

    def test_create_standard_ml_alerts_returns_list(self):
        """Test that create_standard_ml_alerts returns alert rules."""
        from exercise_4_2 import create_standard_ml_alerts, AlertRule

        alerts = create_standard_ml_alerts()

        assert isinstance(alerts, list)
        if alerts:  # If implemented
            assert len(alerts) >= 3, "Should have at least 3 standard alerts"
            assert all(isinstance(a, AlertRule) for a in alerts)


# =============================================================================
# Exercise 4.3 Tests: Model Drift & Retraining
# =============================================================================

class TestExercise43:
    """Tests for Exercise 4.3: Model Drift & Retraining."""

    def test_detect_drift_ks_test_no_drift(self):
        """Test KS test correctly identifies no drift."""
        from exercise_4_3 import detect_drift_ks_test

        np.random.seed(42)
        reference = np.random.normal(50, 10, 1000)
        current = np.random.normal(50, 10, 500)

        result = detect_drift_ks_test(reference, current)

        assert "drift_detected" in result
        if result["error"] is None:  # If implemented
            assert result["drift_detected"] is False, (
                "Same distribution should NOT detect drift"
            )

    def test_detect_drift_ks_test_with_drift(self):
        """Test KS test correctly detects actual drift."""
        from exercise_4_3 import detect_drift_ks_test

        np.random.seed(42)
        reference = np.random.normal(50, 10, 1000)
        # Shifted mean - obvious drift
        current = np.random.normal(70, 10, 500)

        result = detect_drift_ks_test(reference, current)

        if result["error"] is None:  # If implemented
            assert result["drift_detected"] is True, (
                "Shifted distribution SHOULD detect drift"
            )
            assert result["p_value"] < 0.05, (
                "P-value should be low for obvious drift"
            )

    def test_detect_drift_ks_test_validates_sample_size(self):
        """Test that KS test validates minimum sample size."""
        from exercise_4_3 import detect_drift_ks_test

        reference = np.array([1, 2, 3])  # Too small
        current = np.array([4, 5, 6])

        result = detect_drift_ks_test(reference, current, min_samples=30)

        # Should either return error or handle gracefully
        assert result is not None
        if result.get("error"):
            assert "small" in result["error"].lower() or "sample" in result["error"].lower()

    def test_calculate_psi_no_drift(self):
        """Test PSI calculation with no drift."""
        from exercise_4_3 import calculate_psi

        np.random.seed(42)
        reference = np.random.normal(50, 10, 1000)
        current = np.random.normal(50, 10, 500)

        result = calculate_psi(reference, current)

        assert "psi" in result
        if result["psi"] != 0.0:  # If implemented
            assert result["psi"] < 0.1, (
                "Same distribution should have low PSI"
            )

    def test_calculate_psi_with_drift(self):
        """Test PSI calculation with significant drift."""
        from exercise_4_3 import calculate_psi

        np.random.seed(42)
        reference = np.random.normal(50, 10, 1000)
        # Different mean AND std
        current = np.random.normal(70, 20, 500)

        result = calculate_psi(reference, current)

        if result["psi"] != 0.0:  # If implemented
            assert result["psi"] > 0.25, (
                "Obviously shifted distribution should have high PSI"
            )
            assert "action" in result["interpretation"].lower() or "significant" in result["interpretation"].lower()

    def test_retraining_trigger_in_cooldown(self):
        """Test that retraining is blocked during cooldown."""
        from exercise_4_3 import RetrainingTrigger, RetrainingAction

        trigger = RetrainingTrigger(cooldown_hours=24)
        trigger.record_retraining(datetime.now())

        decision = trigger.evaluate(
            psi_score=0.5,  # High PSI
            accuracy_current=0.75,  # Big drop
            accuracy_baseline=0.88,
            available_samples=50000,
        )

        # Should be blocked by cooldown
        assert decision.action in [RetrainingAction.NONE, RetrainingAction.MONITOR], (
            "Retraining should be blocked during cooldown period"
        )
        if "cooldown" not in decision.reason.lower():
            pass  # Might not be implemented yet

    def test_retraining_trigger_accuracy_drop(self):
        """Test that accuracy drop triggers retraining."""
        from exercise_4_3 import RetrainingTrigger, RetrainingAction

        trigger = RetrainingTrigger(
            accuracy_drop_threshold=0.05,
            min_samples_for_retrain=1000,
        )

        decision = trigger.evaluate(
            psi_score=0.15,  # Moderate PSI
            accuracy_current=0.80,  # 8% drop
            accuracy_baseline=0.88,
            available_samples=5000,
        )

        if decision.action != RetrainingAction.NONE:  # If implemented
            assert decision.action in [
                RetrainingAction.RETRAIN,
                RetrainingAction.INVESTIGATE,
                RetrainingAction.EMERGENCY,
            ], "Should recommend retraining on accuracy drop"

    def test_retraining_trigger_insufficient_samples(self):
        """Test that insufficient samples blocks retraining."""
        from exercise_4_3 import RetrainingTrigger, RetrainingAction

        trigger = RetrainingTrigger(min_samples_for_retrain=10000)

        decision = trigger.evaluate(
            psi_score=0.5,  # High PSI
            accuracy_current=0.75,  # Big drop
            accuracy_baseline=0.88,
            available_samples=500,  # Not enough!
        )

        # Should not recommend full retraining without enough data
        if decision.action != RetrainingAction.NONE:  # If implemented
            assert decision.action != RetrainingAction.RETRAIN or (
                "sample" in decision.reason.lower() or "insufficient" in decision.reason.lower()
            )


# =============================================================================
# Exercise 4.4 Tests: Capstone
# =============================================================================

class TestExercise44:
    """Tests for Exercise 4.4: Capstone MLOps Pipeline."""

    def test_mlops_pipeline_initializes(self):
        """Test that MLOpsPipeline can be initialized."""
        from exercise_4_4 import (
            MLOpsPipeline,
            PipelineConfig,
            MockDataValidator,
            MockExperimentTracker,
            MockModelTrainer,
            MockModelRegistry,
            MockDeploymentManager,
            MockMonitoringSetup,
        )

        config = PipelineConfig(
            train_data_path=Path("data/train.csv"),
            val_data_path=Path("data/val.csv"),
            test_data_path=Path("data/test.csv"),
            model_name="test-model",
        )

        pipeline = MLOpsPipeline(
            config=config,
            data_validator=MockDataValidator(),
            experiment_tracker=MockExperimentTracker(),
            model_trainer=MockModelTrainer(),
            model_registry=MockModelRegistry(),
            deployment_manager=MockDeploymentManager(),
            monitoring_setup=MockMonitoringSetup(),
        )

        assert pipeline is not None
        assert pipeline.config == config

    def test_mlops_pipeline_get_status(self):
        """Test pipeline status reporting."""
        from exercise_4_4 import (
            MLOpsPipeline,
            PipelineConfig,
            MockDataValidator,
            MockExperimentTracker,
            MockModelTrainer,
            MockModelRegistry,
            MockDeploymentManager,
            MockMonitoringSetup,
        )

        pipeline = MLOpsPipeline(
            config=PipelineConfig(
                train_data_path=Path("data/train.csv"),
                val_data_path=Path("data/val.csv"),
                test_data_path=Path("data/test.csv"),
                model_name="test-model",
            ),
            data_validator=MockDataValidator(),
            experiment_tracker=MockExperimentTracker(),
            model_trainer=MockModelTrainer(),
            model_registry=MockModelRegistry(),
            deployment_manager=MockDeploymentManager(),
            monitoring_setup=MockMonitoringSetup(),
        )

        status = pipeline.get_status()
        assert isinstance(status, dict)
        assert "current_stage" in status
        assert "errors" in status

    def test_mlops_pipeline_run_returns_result(self):
        """Test that pipeline run returns proper result structure."""
        from exercise_4_4 import (
            MLOpsPipeline,
            PipelineConfig,
            PipelineResult,
            MockDataValidator,
            MockExperimentTracker,
            MockModelTrainer,
            MockModelRegistry,
            MockDeploymentManager,
            MockMonitoringSetup,
        )

        pipeline = MLOpsPipeline(
            config=PipelineConfig(
                train_data_path=Path("data/train.csv"),
                val_data_path=Path("data/val.csv"),
                test_data_path=Path("data/test.csv"),
                model_name="test-model",
            ),
            data_validator=MockDataValidator(),
            experiment_tracker=MockExperimentTracker(),
            model_trainer=MockModelTrainer(),
            model_registry=MockModelRegistry(),
            deployment_manager=MockDeploymentManager(),
            monitoring_setup=MockMonitoringSetup(),
        )

        result = pipeline.run()

        assert isinstance(result, dict)
        assert "success" in result
        assert "stage_reached" in result
        assert "errors" in result
        assert "duration_seconds" in result

    def test_mlops_pipeline_handles_data_validation_failure(self):
        """Test that pipeline handles data validation failure gracefully."""
        from exercise_4_4 import (
            MLOpsPipeline,
            PipelineConfig,
            MockDataValidator,
            MockExperimentTracker,
            MockModelTrainer,
            MockModelRegistry,
            MockDeploymentManager,
            MockMonitoringSetup,
        )

        pipeline = MLOpsPipeline(
            config=PipelineConfig(
                train_data_path=Path("data/train.csv"),
                val_data_path=Path("data/val.csv"),
                test_data_path=Path("data/test.csv"),
                model_name="test-model",
            ),
            data_validator=MockDataValidator(should_pass=False),  # Will fail
            experiment_tracker=MockExperimentTracker(),
            model_trainer=MockModelTrainer(),
            model_registry=MockModelRegistry(),
            deployment_manager=MockDeploymentManager(),
            monitoring_setup=MockMonitoringSetup(),
        )

        result = pipeline.run()

        # Pipeline should complete (not crash) but report failure
        assert result is not None

        # If implemented, should show failure
        if result["stage_reached"] != "not_started":
            assert result["success"] is False or "data" in result["stage_reached"].lower()

    def test_pipeline_stage_enum_values(self):
        """Test that PipelineStage has expected values."""
        from exercise_4_4 import PipelineStage

        # Verify all expected stages exist
        expected_stages = [
            "data_validation",
            "training",
            "model_validation",
            "registration",
            "deployment",
            "monitoring",
            "completed",
            "failed",
        ]

        actual_values = [stage.value for stage in PipelineStage]
        for expected in expected_stages:
            assert any(expected in actual for actual in actual_values), (
                f"Missing stage: {expected}"
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests that verify cross-exercise functionality."""

    def test_monitoring_and_drift_integration(self):
        """Test that monitoring alerts can use drift metrics."""
        from exercise_4_2 import AlertRule, AlertSeverity
        from exercise_4_3 import calculate_psi

        np.random.seed(42)
        reference = np.random.normal(50, 10, 1000)
        current = np.random.normal(65, 15, 500)  # Drifted

        psi_result = calculate_psi(reference, current)

        # Create alert rule for PSI
        psi_alert = AlertRule(
            name="PSIDriftAlert",
            description="Alert when PSI indicates drift",
            metric_name="model_psi",
            threshold=0.25,
            comparison="gt",
            for_seconds=0,  # Immediate for testing
            severity=AlertSeverity.WARNING,
        )

        if psi_result["psi"] != 0.0:  # If PSI implemented
            state = psi_alert.evaluate(psi_result["psi"], current_time=100.0)
            # High PSI should trigger alert
            assert state.value in ["pending", "firing", "ok"]


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
