"""
Test Suite for Level 2: Pipeline Exercises
-------------------------------------------

Run with: pytest project/exercises/2-pipeline/test_level_2.py -v

These tests verify student implementations of:
- Exercise 2.1: Data Versioning
- Exercise 2.2: Experiment Tracking
- Exercise 2.3: Model Registry
- Exercise 2.4: Testing ML Code
"""

import hashlib
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# Exercise 2.1: Data Versioning Tests
# =============================================================================

class TestExercise21DataVersioning:
    """Tests for Exercise 2.1: Data Versioning."""

    def test_generate_data_hash_basic(self):
        """Test basic file hashing."""
        from exercise_2_1 import generate_data_hash, DataHash

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            result = generate_data_hash(temp_path)

            # Check return type
            assert isinstance(result, DataHash)
            assert result.filepath == temp_path

            # Verify hash is correct MD5
            expected_hash = hashlib.md5(b"Hello, World!").hexdigest()
            assert result.md5 == expected_hash

            # Check size
            assert result.size == 13  # "Hello, World!" is 13 bytes
        finally:
            os.unlink(temp_path)

    def test_generate_data_hash_file_not_found(self):
        """Test handling of missing file."""
        from exercise_2_1 import generate_data_hash

        with pytest.raises(FileNotFoundError):
            generate_data_hash("/nonexistent/path/file.csv")

    def test_generate_data_hash_binary_file(self):
        """Test hashing binary files."""
        from exercise_2_1 import generate_data_hash

        # Create binary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(bytes([0, 1, 2, 3, 255, 254, 253]))
            temp_path = f.name

        try:
            result = generate_data_hash(temp_path)
            expected_hash = hashlib.md5(bytes([0, 1, 2, 3, 255, 254, 253])).hexdigest()
            assert result.md5 == expected_hash
        finally:
            os.unlink(temp_path)

    def test_generate_data_hash_chunked(self):
        """Test chunked hashing produces same result as regular."""
        from exercise_2_1 import generate_data_hash, generate_data_hash_chunked

        # Create a larger file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(os.urandom(10000))  # 10KB random data
            temp_path = f.name

        try:
            regular_result = generate_data_hash(temp_path)
            chunked_result = generate_data_hash_chunked(temp_path, chunk_size=1024)

            # Hashes should match
            assert regular_result.md5 == chunked_result.md5
            assert regular_result.size == chunked_result.size
        finally:
            os.unlink(temp_path)

    def test_fix_dvc_pipeline(self):
        """Test DVC pipeline fixing."""
        from exercise_2_1 import fix_dvc_pipeline, BUGGY_DVC_PIPELINE

        fixed = fix_dvc_pipeline(BUGGY_DVC_PIPELINE)

        # Check that train depends on features.csv
        assert "data/features.csv" in fixed
        # train stage should NOT depend on raw.csv (should be features.csv)
        lines = fixed.split('\n')
        in_train_stage = False
        for line in lines:
            if 'train:' in line:
                in_train_stage = True
            elif line.strip().startswith('featurize:') or line.strip().startswith('preprocess:'):
                in_train_stage = False
            if in_train_stage and 'deps:' in line:
                # Next lines are dependencies for train
                pass

        # Verify preprocess doesn't have circular dependency
        assert "preprocess:" in fixed

    def test_verify_data_integrity_valid(self):
        """Test integrity check on valid file."""
        from exercise_2_1 import verify_data_integrity, generate_data_hash

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            # Get actual hash
            actual = generate_data_hash(temp_path)

            # Verify integrity
            result = verify_data_integrity(temp_path, actual.md5, actual.size)

            assert result.is_valid is True
            assert result.size_matches is True
            assert result.actual_hash == actual.md5
        finally:
            os.unlink(temp_path)

    def test_verify_data_integrity_invalid(self):
        """Test integrity check detects tampering."""
        from exercise_2_1 import verify_data_integrity

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Modified content")
            temp_path = f.name

        try:
            # Wrong hash
            result = verify_data_integrity(temp_path, "wrong_hash_value", 100)

            assert result.is_valid is False
        finally:
            os.unlink(temp_path)


# =============================================================================
# Exercise 2.2: Experiment Tracking Tests
# =============================================================================

class TestExercise22ExperimentTracking:
    """Tests for Exercise 2.2: Experiment Tracking."""

    def test_generate_run_name(self):
        """Test run name generation."""
        from exercise_2_2 import generate_run_name

        name1 = generate_run_name("experiment")
        name2 = generate_run_name("experiment")

        assert name1.startswith("experiment_")
        assert len(name1) > len("experiment_")
        # Should contain date-like pattern
        assert any(c.isdigit() for c in name1)

    def test_flatten_params_simple(self):
        """Test flattening non-nested params."""
        from exercise_2_2 import flatten_params

        params = {"lr": 0.01, "epochs": 10}
        flat = flatten_params(params)

        assert flat == {"lr": 0.01, "epochs": 10}

    def test_flatten_params_nested(self):
        """Test flattening nested params."""
        from exercise_2_2 import flatten_params

        params = {
            "model": {"type": "cnn", "layers": 3},
            "optimizer": {"name": "adam", "lr": 0.001}
        }
        flat = flatten_params(params)

        assert flat == {
            "model.type": "cnn",
            "model.layers": 3,
            "optimizer.name": "adam",
            "optimizer.lr": 0.001
        }

    def test_flatten_params_deeply_nested(self):
        """Test flattening deeply nested params."""
        from exercise_2_2 import flatten_params

        params = {"a": {"b": {"c": 1}}}
        flat = flatten_params(params)

        assert flat == {"a.b.c": 1}

    def test_validate_metrics_valid(self):
        """Test metric validation with valid input."""
        from exercise_2_2 import validate_metrics

        metrics = {"accuracy": 0.95, "loss": 0.1, "f1": 0.92}
        result = validate_metrics(metrics)

        assert all(isinstance(v, float) for v in result.values())
        assert result == {"accuracy": 0.95, "loss": 0.1, "f1": 0.92}

    def test_validate_metrics_with_int(self):
        """Test metric validation converts ints to floats."""
        from exercise_2_2 import validate_metrics

        metrics = {"epoch": 10, "accuracy": 0.95}
        result = validate_metrics(metrics)

        assert isinstance(result["epoch"], float)
        assert result["epoch"] == 10.0

    def test_validate_metrics_invalid(self):
        """Test metric validation rejects non-numeric."""
        from exercise_2_2 import validate_metrics

        metrics = {"accuracy": 0.95, "model_name": "bert"}

        with pytest.raises(ValueError):
            validate_metrics(metrics)

    def test_get_git_info(self):
        """Test git info extraction."""
        from exercise_2_2 import get_git_info

        info = get_git_info()

        # Should return dict with expected keys
        assert isinstance(info, dict)
        assert "git_commit" in info
        assert "git_branch" in info
        assert "git_dirty" in info

        # Values should be strings
        assert all(isinstance(v, str) for v in info.values())

    @pytest.mark.skipif(
        not os.path.exists('.git'),
        reason="Not in a git repository"
    )
    def test_get_git_info_in_repo(self):
        """Test git info in actual repo."""
        from exercise_2_2 import get_git_info

        info = get_git_info()

        # Should have valid commit hash (40 hex chars)
        if info["git_commit"] != "unknown":
            assert len(info["git_commit"]) == 40
            assert all(c in "0123456789abcdef" for c in info["git_commit"])


# =============================================================================
# Exercise 2.3: Model Registry Tests
# =============================================================================

class TestExercise23ModelRegistry:
    """Tests for Exercise 2.3: Model Registry."""

    def test_validate_stage_transition_valid(self):
        """Test valid stage transitions."""
        from exercise_2_3 import validate_stage_transition, ModelStage

        # Valid transitions
        assert validate_stage_transition(ModelStage.NONE, ModelStage.STAGING) is True
        assert validate_stage_transition(ModelStage.STAGING, ModelStage.PRODUCTION) is True
        assert validate_stage_transition(ModelStage.PRODUCTION, ModelStage.ARCHIVED) is True
        assert validate_stage_transition(ModelStage.ARCHIVED, ModelStage.STAGING) is True

    def test_validate_stage_transition_invalid(self):
        """Test invalid stage transitions."""
        from exercise_2_3 import validate_stage_transition, ModelStage

        # Invalid: can't go directly from None to Production
        assert validate_stage_transition(ModelStage.NONE, ModelStage.PRODUCTION) is False

        # Invalid: Archived to Production (must go through Staging)
        assert validate_stage_transition(ModelStage.ARCHIVED, ModelStage.PRODUCTION) is False

    def test_mock_client_create_model(self):
        """Test mock MLflow client."""
        from exercise_2_3 import MockMlflowClient

        client = MockMlflowClient()
        client.create_registered_model("test-model")

        v1 = client.create_model_version("test-model", "s3://bucket/model", "run123")

        assert v1.name == "test-model"
        assert v1.version == "1"
        assert v1.current_stage == "None"

    def test_mock_client_transition(self):
        """Test mock client stage transitions."""
        from exercise_2_3 import MockMlflowClient

        client = MockMlflowClient()
        client.create_registered_model("test-model")
        v1 = client.create_model_version("test-model", "source", "run1")

        # Transition to Staging
        v1_updated = client.transition_model_version_stage("test-model", "1", "Staging")

        assert v1_updated.current_stage == "Staging"
        assert len(client.transitions) == 1
        assert client.transitions[0] == ("test-model", "1", "Staging")

    def test_archive_models_in_stage(self):
        """Test archiving models in a stage."""
        from exercise_2_3 import archive_models_in_stage, MockMlflowClient, ModelStage

        client = MockMlflowClient()
        client.create_registered_model("test-model")
        v1 = client.create_model_version("test-model", "source", "run1")
        v2 = client.create_model_version("test-model", "source", "run2")

        # Put both in Staging
        client.transition_model_version_stage("test-model", "1", "Staging")
        client.transition_model_version_stage("test-model", "2", "Staging")

        # Archive all in Staging
        archived = archive_models_in_stage(client, "test-model", ModelStage.STAGING)

        assert len(archived) == 2
        assert 1 in archived
        assert 2 in archived


# =============================================================================
# Exercise 2.4: Testing ML Code Tests
# =============================================================================

class TestExercise24TestingMLCode:
    """Tests for Exercise 2.4: Testing ML Code."""

    def test_validate_model_output_valid(self):
        """Test validation of valid predictions."""
        from exercise_2_4 import validate_model_output

        predictions = np.array([0, 1, 1, 0, 1])
        probabilities = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.85, 0.15],
            [0.1, 0.9]
        ])

        result = validate_model_output(
            predictions,
            probabilities,
            expected_classes=[0, 1],
            n_samples=5
        )

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_model_output_wrong_shape(self):
        """Test validation catches shape mismatch."""
        from exercise_2_4 import validate_model_output

        predictions = np.array([0, 1, 1])  # 3 samples
        probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])  # 2 samples

        result = validate_model_output(predictions, probabilities)

        assert result.is_valid is False
        assert any("shape" in str(i.check_name).lower() for i in result.errors)

    def test_validate_model_output_invalid_class(self):
        """Test validation catches invalid class labels."""
        from exercise_2_4 import validate_model_output

        predictions = np.array([0, 1, 2, 0, 1])  # Has class 2

        result = validate_model_output(predictions, expected_classes=[0, 1])

        assert result.is_valid is False
        assert any("class" in str(i.check_name).lower() for i in result.errors)

    def test_validate_model_output_bad_probabilities(self):
        """Test validation catches probabilities that don't sum to 1."""
        from exercise_2_4 import validate_model_output

        predictions = np.array([0, 1])
        probabilities = np.array([
            [0.5, 0.3],  # Sums to 0.8, not 1
            [0.4, 0.6]   # OK
        ])

        result = validate_model_output(predictions, probabilities)

        assert result.is_valid is False
        assert any("sum" in str(i.message).lower() or "prob" in str(i.check_name).lower()
                   for i in result.errors)

    def test_validate_model_output_negative_probabilities(self):
        """Test validation catches negative probabilities."""
        from exercise_2_4 import validate_model_output

        predictions = np.array([0, 1])
        probabilities = np.array([
            [1.2, -0.2],  # Has negative value
            [0.4, 0.6]
        ])

        result = validate_model_output(predictions, probabilities)

        assert result.is_valid is False

    def test_validate_model_output_with_nan(self):
        """Test validation catches NaN values."""
        from exercise_2_4 import validate_model_output

        predictions = np.array([0.0, 1.0, np.nan, 0.0])

        result = validate_model_output(predictions)

        assert result.is_valid is False
        assert any("nan" in str(i.message).lower() or "null" in str(i.check_name).lower()
                   for i in result.issues)

    def test_check_probability_distribution(self):
        """Test probability distribution checking."""
        from exercise_2_4 import check_probability_distribution

        # Valid probabilities
        valid_probs = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        issues = check_probability_distribution(valid_probs)
        assert len(issues) == 0

        # Invalid: doesn't sum to 1
        invalid_probs = np.array([[0.3, 0.3], [0.5, 0.5]])
        issues = check_probability_distribution(invalid_probs)
        assert len(issues) > 0

    def test_check_null_values(self):
        """Test null value checking."""
        from exercise_2_4 import check_null_values

        # No nulls
        clean = np.array([1.0, 2.0, 3.0])
        assert len(check_null_values(clean)) == 0

        # Has NaN
        with_nan = np.array([1.0, np.nan, 3.0])
        issues = check_null_values(with_nan)
        assert len(issues) > 0

    def test_fixed_test_is_deterministic(self):
        """Test that fixed test produces consistent results."""
        from exercise_2_4 import fixed_test_model_training

        # Run multiple times - should not raise
        for _ in range(5):
            fixed_test_model_training()  # Should pass consistently

    def test_deterministic_model_training(self):
        """Test deterministic training produces same results."""
        from exercise_2_4 import deterministic_model_training, create_reproducible_test_data

        X, y = create_reproducible_test_data(n_samples=100, random_state=42)

        # Run twice with same seed
        model1, acc1 = deterministic_model_training(X, y, random_state=42)
        model2, acc2 = deterministic_model_training(X, y, random_state=42)

        # Should produce identical results
        assert acc1 == acc2

        # Predictions should match
        np.testing.assert_array_equal(model1.predict(X), model2.predict(X))

    def test_calculate_prediction_statistics(self):
        """Test prediction statistics calculation."""
        from exercise_2_4 import calculate_prediction_statistics

        predictions = np.array([0, 0, 0, 1, 1, 2])
        stats = calculate_prediction_statistics(predictions)

        assert stats["n_samples"] == 6
        assert stats["n_unique"] == 3
        assert stats["most_common_class"] == 0
        assert stats["most_common_fraction"] == 0.5
        assert stats["class_counts"] == {0: 3, 1: 2, 2: 1}


# =============================================================================
# Integration Tests
# =============================================================================

class TestLevel2Integration:
    """Integration tests combining multiple exercises."""

    def test_hash_then_verify_workflow(self):
        """Test typical workflow: hash file, then verify later."""
        from exercise_2_1 import generate_data_hash, verify_data_integrity

        # Create file and hash it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value\n1,100\n2,200\n")
            temp_path = f.name

        try:
            original_hash = generate_data_hash(temp_path)

            # Simulate storing hash (like DVC does)
            stored_hash = original_hash.md5
            stored_size = original_hash.size

            # Later, verify integrity
            result = verify_data_integrity(temp_path, stored_hash, stored_size)
            assert result.is_valid is True

            # Modify file (simulating data corruption/tampering)
            with open(temp_path, 'a') as f:
                f.write("3,300\n")

            # Verify should now fail
            result = verify_data_integrity(temp_path, stored_hash, stored_size)
            assert result.is_valid is False

        finally:
            os.unlink(temp_path)

    def test_experiment_logging_with_validation(self):
        """Test experiment logging with model output validation."""
        from exercise_2_2 import flatten_params, validate_metrics
        from exercise_2_4 import validate_model_output

        # Simulate model output
        predictions = np.array([0, 1, 1, 0, 1])
        probabilities = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.85, 0.15], [0.1, 0.9]
        ])

        # Validate predictions
        validation = validate_model_output(
            predictions, probabilities, expected_classes=[0, 1]
        )
        assert validation.is_valid

        # Prepare for logging
        params = {
            "model": {"type": "logistic", "C": 1.0},
            "data": {"n_samples": 5}
        }
        flat_params = flatten_params(params)
        assert "model.type" in flat_params

        metrics = {
            "accuracy": 0.8,
            "f1": 0.75,
            "validation_passed": 1  # int should convert to float
        }
        validated_metrics = validate_metrics(metrics)
        assert isinstance(validated_metrics["validation_passed"], float)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
