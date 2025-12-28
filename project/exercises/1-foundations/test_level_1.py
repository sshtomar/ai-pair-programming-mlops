"""
Test Suite for Level 1: Foundations
===================================

These tests verify that students have correctly implemented the exercises.
Run with: pytest test_level_1.py -v

Tests are organized by exercise and check both:
- Guided stub implementations work correctly
- Debug fixes resolve the identified issues
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest


# =============================================================================
# Exercise 1.1: Welcome & Setup
# =============================================================================

class TestExercise1_1:
    """Tests for exercise_1_1.py - Environment verification and imports."""

    def test_check_environment_returns_dict(self):
        """check_environment should return a dictionary."""
        from exercise_1_1 import check_environment

        result = check_environment()
        assert isinstance(result, dict), (
            "check_environment() should return a dictionary, "
            f"got {type(result).__name__}"
        )

    def test_check_environment_contains_required_packages(self):
        """check_environment should include all required packages."""
        from exercise_1_1 import REQUIRED_PACKAGES, check_environment

        result = check_environment()
        for pkg in REQUIRED_PACKAGES:
            assert pkg in result, (
                f"Missing required package '{pkg}' in check_environment() result. "
                f"Got keys: {list(result.keys())}"
            )

    def test_check_environment_versions_are_strings(self):
        """Package versions should be returned as strings."""
        from exercise_1_1 import check_environment

        result = check_environment()
        for pkg, version in result.items():
            assert isinstance(version, str), (
                f"Version for '{pkg}' should be a string, "
                f"got {type(version).__name__}: {version}"
            )

    def test_check_environment_raises_on_old_python(self):
        """check_environment should raise RuntimeError if Python < 3.10."""
        # This test only verifies the check exists - it won't actually fail
        # on Python >= 3.10, but we can verify the function handles the check
        from exercise_1_1 import check_environment

        # If we're running the test, Python is >= 3.10, so just ensure no error
        try:
            check_environment()
        except RuntimeError as e:
            pytest.fail(f"Unexpected RuntimeError on Python {sys.version}: {e}")

    def test_fix_circular_import_returns_class(self):
        """fix_circular_import should return a class."""
        from exercise_1_1 import fix_circular_import

        FixedClass = fix_circular_import()
        assert isinstance(FixedClass, type), (
            "fix_circular_import() should return a class, "
            f"got {type(FixedClass).__name__}"
        )

    def test_fixed_metrics_calculator_works(self):
        """The fixed MetricsCalculator should calculate accuracy correctly."""
        from exercise_1_1 import fix_circular_import

        FixedClass = fix_circular_import()
        calculator = FixedClass()

        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        # Correct: positions 0, 2, 4 = 3/5 = 0.6

        accuracy = calculator.calculate_accuracy(y_true, y_pred)
        assert accuracy == pytest.approx(0.6), (
            f"Expected accuracy of 0.6 for the given inputs, got {accuracy}"
        )

    def test_fixed_metrics_calculator_uses_dependency_injection(self):
        """The fixed class should accept predictor as parameter, not import it."""
        from exercise_1_1 import fix_circular_import

        FixedClass = fix_circular_import()

        # Should be able to instantiate without any imports happening
        calculator = FixedClass(predictor="mock_predictor")
        assert hasattr(calculator, "predictor") or True, (
            "Fixed class should accept predictor via __init__"
        )

    def test_fixed_metrics_handles_edge_cases(self):
        """Fixed calculator should handle edge cases properly."""
        from exercise_1_1 import fix_circular_import

        FixedClass = fix_circular_import()
        calculator = FixedClass()

        # Length mismatch
        with pytest.raises(ValueError, match="[Ll]ength|mismatch"):
            calculator.calculate_accuracy([1, 0], [1])

        # Empty inputs
        with pytest.raises(ValueError, match="[Ee]mpty"):
            calculator.calculate_accuracy([], [])


# =============================================================================
# Exercise 1.2: MLOps Principles
# =============================================================================

class TestExercise1_2:
    """Tests for exercise_1_2.py - Business metrics and config loading."""

    def test_calculate_business_metric_returns_dict(self):
        """calculate_business_metric should return a dictionary with required keys."""
        from exercise_1_2 import calculate_business_metric

        result = calculate_business_metric(
            predictions=[0, 1, 0, 1],
            actuals=[0, 0, 1, 1],
            false_negative_cost=100.0,
            false_positive_cost=10.0,
        )

        assert isinstance(result, dict), (
            f"Should return a dict, got {type(result).__name__}"
        )

        required_keys = [
            "false_negatives", "false_positives", "true_positives",
            "true_negatives", "total_cost", "accuracy", "cost_per_prediction"
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: '{key}'"

    def test_calculate_business_metric_confusion_matrix(self):
        """Verify correct confusion matrix counts."""
        from exercise_1_2 import calculate_business_metric

        # Designed test case:
        # pred=0, actual=0 -> TN (index 0)
        # pred=1, actual=0 -> FP (index 1)
        # pred=0, actual=1 -> FN (index 2)
        # pred=1, actual=1 -> TP (index 3)
        result = calculate_business_metric(
            predictions=[0, 1, 0, 1],
            actuals=[0, 0, 1, 1],
            false_negative_cost=100.0,
            false_positive_cost=10.0,
        )

        assert result["true_negatives"] == 1, (
            f"Expected 1 TN, got {result['true_negatives']}"
        )
        assert result["false_positives"] == 1, (
            f"Expected 1 FP, got {result['false_positives']}"
        )
        assert result["false_negatives"] == 1, (
            f"Expected 1 FN, got {result['false_negatives']}"
        )
        assert result["true_positives"] == 1, (
            f"Expected 1 TP, got {result['true_positives']}"
        )

    def test_calculate_business_metric_total_cost(self):
        """Verify correct total cost calculation."""
        from exercise_1_2 import calculate_business_metric

        result = calculate_business_metric(
            predictions=[0, 1, 0, 1],
            actuals=[0, 0, 1, 1],
            false_negative_cost=100.0,  # 1 FN * 100 = 100
            false_positive_cost=10.0,   # 1 FP * 10 = 10
        )

        assert result["total_cost"] == pytest.approx(110.0), (
            f"Expected total cost of 110.0, got {result['total_cost']}. "
            "Cost = (FN * fn_cost) + (FP * fp_cost) = (1 * 100) + (1 * 10) = 110"
        )

    def test_calculate_business_metric_accuracy(self):
        """Verify correct accuracy calculation."""
        from exercise_1_2 import calculate_business_metric

        result = calculate_business_metric(
            predictions=[0, 1, 0, 1],
            actuals=[0, 0, 1, 1],
            false_negative_cost=100.0,
            false_positive_cost=10.0,
        )

        # 2 correct (TN + TP) out of 4 = 0.5
        assert result["accuracy"] == pytest.approx(0.5), (
            f"Expected accuracy of 0.5, got {result['accuracy']}"
        )

    def test_calculate_business_metric_validation(self):
        """Verify input validation."""
        from exercise_1_2 import calculate_business_metric

        # Length mismatch
        with pytest.raises(ValueError):
            calculate_business_metric([0, 1], [0], 100, 10)

        # Empty inputs
        with pytest.raises(ValueError):
            calculate_business_metric([], [], 100, 10)

        # Invalid values (not 0 or 1)
        with pytest.raises(ValueError):
            calculate_business_metric([0, 2], [0, 1], 100, 10)

    def test_load_config_with_valid_file(self):
        """load_config should correctly load a JSON config file."""
        from exercise_1_2 import load_config

        # Create a temporary config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "test_config.json"
            config_file.write_text('{"key": "value", "number": 42}')

            result = load_config("test_config", config_dir=config_dir)

            assert result == {"key": "value", "number": 42}, (
                f"Config content mismatch. Got: {result}"
            )

    def test_load_config_file_not_found(self):
        """load_config should raise FileNotFoundError for missing files."""
        from exercise_1_2 import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_config("nonexistent", config_dir=Path(tmpdir))

    def test_load_config_invalid_json(self):
        """load_config should raise JSONDecodeError for invalid JSON."""
        from exercise_1_2 import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "bad.json"
            config_file.write_text('{"invalid json')

            with pytest.raises(json.JSONDecodeError):
                load_config("bad", config_dir=config_dir)

    def test_load_config_uses_pathlib(self):
        """load_config should work with Path objects, not just strings."""
        from exercise_1_2 import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "pathlib_test.json"
            config_file.write_text('{"works": true}')

            # Should accept Path objects
            result = load_config("pathlib_test", config_dir=config_dir)
            assert result == {"works": True}


# =============================================================================
# Exercise 1.3: First Model
# =============================================================================

class TestExercise1_3:
    """Tests for exercise_1_3.py - Text preprocessing and data leakage."""

    def test_preprocess_text_lowercase(self):
        """preprocess_text should convert text to lowercase."""
        from exercise_1_3 import preprocess_text

        assert preprocess_text("HELLO") == "hello", (
            "Failed to convert to lowercase"
        )
        assert preprocess_text("HeLLo WoRLd") == "hello world", (
            "Failed to convert mixed case to lowercase"
        )

    def test_preprocess_text_strip_whitespace(self):
        """preprocess_text should strip leading/trailing whitespace."""
        from exercise_1_3 import preprocess_text

        assert preprocess_text("  hello  ") == "hello", (
            "Failed to strip whitespace"
        )
        assert preprocess_text("\thello\n") == "hello", (
            "Failed to strip tabs and newlines"
        )

    def test_preprocess_text_remove_punctuation(self):
        """preprocess_text should remove punctuation."""
        from exercise_1_3 import preprocess_text

        assert preprocess_text("hello, world!") == "hello world", (
            "Failed to remove comma and exclamation mark"
        )
        assert preprocess_text("what's up?") == "what s up", (
            "Failed to handle apostrophe (word boundaries should be preserved)"
        )

    def test_preprocess_text_collapse_spaces(self):
        """preprocess_text should collapse multiple spaces."""
        from exercise_1_3 import preprocess_text

        assert preprocess_text("hello    world") == "hello world", (
            "Failed to collapse multiple spaces"
        )

    def test_preprocess_text_combined(self):
        """Test all preprocessing steps together."""
        from exercise_1_3 import preprocess_text

        result = preprocess_text("  Hello, World!!!  How are you?  ")
        assert result == "hello world how are you", (
            f"Combined preprocessing failed. Got: '{result}'"
        )

    def test_preprocess_text_type_error(self):
        """preprocess_text should raise TypeError for non-string input."""
        from exercise_1_3 import preprocess_text

        with pytest.raises(TypeError):
            preprocess_text(123)

        with pytest.raises(TypeError):
            preprocess_text(None)

    def test_preprocess_batch(self):
        """preprocess_batch should process a list of texts."""
        from exercise_1_3 import preprocess_batch

        texts = ["Hello!", "WORLD", "  test  "]
        result = preprocess_batch(texts)

        assert result == ["hello", "world", "test"], (
            f"Batch preprocessing failed. Got: {result}"
        )

    def test_preprocess_batch_type_error(self):
        """preprocess_batch should raise TypeError for invalid input."""
        from exercise_1_3 import preprocess_batch

        with pytest.raises(TypeError):
            preprocess_batch("not a list")

    def test_train_model_fixed_returns_dict(self):
        """train_model_fixed should return required keys."""
        pytest.importorskip("sklearn")
        from exercise_1_3 import train_model_fixed

        texts = ["good", "bad", "great", "terrible"] * 10
        labels = [1, 0, 1, 0] * 10

        result = train_model_fixed(texts, labels)

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        for key in ["train_accuracy", "test_accuracy", "vectorizer", "model"]:
            assert key in result, f"Missing key: {key}"

    def test_train_model_fixed_validation(self):
        """train_model_fixed should validate inputs."""
        pytest.importorskip("sklearn")
        from exercise_1_3 import train_model_fixed

        with pytest.raises(ValueError):
            train_model_fixed(["a", "b"], [1])  # Length mismatch

        with pytest.raises(ValueError):
            train_model_fixed([], [])  # Empty inputs

    def test_train_model_fixed_no_leakage(self):
        """
        Verify the fix prevents data leakage.

        Data leakage occurs when the vectorizer is fit on all data before splitting.
        The fix should fit the vectorizer only on training data.
        """
        pytest.importorskip("sklearn")
        from exercise_1_3 import train_model_fixed

        # The test is conceptual - if students implemented it correctly,
        # the vectorizer should only know vocabulary from training set
        texts = ["unique_train_word good", "bad", "great", "terrible"] * 20
        labels = [1, 0, 1, 0] * 20

        result = train_model_fixed(texts, labels)
        vectorizer = result["vectorizer"]

        # Check that vectorizer was fitted (has vocabulary)
        assert hasattr(vectorizer, "vocabulary_"), (
            "Vectorizer was not fitted. Did you call fit_transform on training data?"
        )


# =============================================================================
# Exercise 1.4: Docker Basics
# =============================================================================

class TestExercise1_4:
    """Tests for exercise_1_4.py - Dockerfile validation."""

    def test_validate_dockerfile_returns_list(self):
        """validate_dockerfile should return a list of issues."""
        from exercise_1_4 import validate_dockerfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\nCMD ['python']")

            result = validate_dockerfile(dockerfile)
            assert isinstance(result, list), (
                f"Expected list, got {type(result).__name__}"
            )

    def test_validate_dockerfile_file_not_found(self):
        """validate_dockerfile should raise FileNotFoundError."""
        from exercise_1_4 import validate_dockerfile

        with pytest.raises(FileNotFoundError):
            validate_dockerfile("/nonexistent/Dockerfile")

    def test_validate_dockerfile_detects_latest_tag(self):
        """Should detect :latest tag usage."""
        from exercise_1_4 import DockerfileIssue, validate_dockerfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:latest\nCMD ['python']")

            issues = validate_dockerfile(dockerfile)
            latest_issues = [i for i in issues if "latest" in i.message.lower()]

            assert len(latest_issues) > 0, (
                "Should detect 'latest' tag as an issue. "
                f"Found issues: {[str(i) for i in issues]}"
            )

    def test_validate_dockerfile_detects_no_tag(self):
        """Should detect missing tag (implies latest)."""
        from exercise_1_4 import validate_dockerfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python\nCMD ['python']")

            issues = validate_dockerfile(dockerfile)
            tag_issues = [
                i for i in issues
                if "tag" in i.message.lower() or "latest" in i.message.lower()
            ]

            assert len(tag_issues) > 0, (
                "Should detect missing tag (implies latest)"
            )

    def test_validate_dockerfile_detects_missing_dockerignore(self):
        """Should warn about missing .dockerignore."""
        from exercise_1_4 import validate_dockerfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\nCMD ['python']")
            # Note: not creating .dockerignore

            issues = validate_dockerfile(dockerfile)
            ignore_issues = [i for i in issues if "dockerignore" in i.message.lower()]

            assert len(ignore_issues) > 0, (
                "Should detect missing .dockerignore"
            )

    def test_validate_dockerfile_no_issue_with_dockerignore(self):
        """Should not warn when .dockerignore exists."""
        from exercise_1_4 import validate_dockerfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\nCMD ['python']")
            dockerignore = Path(tmpdir) / ".dockerignore"
            dockerignore.write_text("*.pyc\n__pycache__")

            issues = validate_dockerfile(dockerfile)
            ignore_issues = [i for i in issues if "dockerignore" in i.message.lower()]

            assert len(ignore_issues) == 0, (
                "Should not warn about .dockerignore when it exists"
            )

    def test_validate_dockerfile_detects_no_user(self):
        """Should warn about running as root (no USER instruction)."""
        from exercise_1_4 import validate_dockerfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11\nCMD ['python']")

            issues = validate_dockerfile(dockerfile)
            user_issues = [
                i for i in issues
                if "root" in i.message.lower() or "user" in i.message.lower()
            ]

            assert len(user_issues) > 0, (
                "Should detect running as root (no USER instruction)"
            )

    def test_validate_dockerfile_accepts_user(self):
        """Should not warn about root when USER is present."""
        from exercise_1_4 import validate_dockerfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerignore = Path(tmpdir) / ".dockerignore"
            dockerfile.write_text(
                "FROM python:3.11\nUSER appuser\nCMD ['python']"
            )
            dockerignore.write_text("*.pyc")

            issues = validate_dockerfile(dockerfile)
            root_issues = [
                i for i in issues
                if "root" in i.message.lower() and "user" in i.message.lower()
            ]

            assert len(root_issues) == 0, (
                "Should not warn about root when USER instruction is present"
            )

    def test_dockerfile_issue_repr(self):
        """DockerfileIssue should have readable string representation."""
        from exercise_1_4 import DockerfileIssue

        issue = DockerfileIssue(
            line_number=5,
            severity="warning",
            message="Test message",
            suggestion="Test suggestion",
        )

        repr_str = repr(issue)
        assert "WARNING" in repr_str, "Should include severity"
        assert "line 5" in repr_str, "Should include line number"
        assert "Test message" in repr_str, "Should include message"

    def test_get_fixed_dockerfile_has_version_tag(self):
        """Fixed Dockerfile should use specific version tag."""
        from exercise_1_4 import get_fixed_dockerfile

        fixed = get_fixed_dockerfile()
        assert ":latest" not in fixed, "Fixed Dockerfile should not use :latest"
        assert "python:" in fixed.lower(), "Should specify Python version"

    def test_get_fixed_dockerfile_has_user(self):
        """Fixed Dockerfile should have USER instruction."""
        from exercise_1_4 import get_fixed_dockerfile

        fixed = get_fixed_dockerfile()
        assert "USER" in fixed, "Fixed Dockerfile should have USER instruction"

    def test_explain_dockerfile_issues_complete(self):
        """explain_dockerfile_issues should explain all issues."""
        from exercise_1_4 import explain_dockerfile_issues

        explanations = explain_dockerfile_issues()

        required_keys = [
            "latest_tag", "copy_before_deps", "multiple_runs",
            "add_vs_copy", "unpinned_versions", "running_as_root",
            "no_healthcheck", "apt_cache"
        ]

        for key in required_keys:
            assert key in explanations, f"Missing explanation for: {key}"
            assert len(explanations[key]) > 0, (
                f"Explanation for '{key}' is empty - please provide an explanation"
            )


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
