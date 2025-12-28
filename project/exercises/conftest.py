"""
Pytest configuration for MLOps course exercises.

This conftest.py enables pytest to discover and run tests across all exercise levels.
"""

import sys
from pathlib import Path

import pytest

# Add exercises directory to path so imports work
exercises_dir = Path(__file__).parent
sys.path.insert(0, str(exercises_dir))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "level1: marks tests as Level 1 - Foundations"
    )
    config.addinivalue_line(
        "markers", "level2: marks tests as Level 2 - Pipeline"
    )
    config.addinivalue_line(
        "markers", "level3: marks tests as Level 3 - Deployment"
    )
    config.addinivalue_line(
        "markers", "level4: marks tests as Level 4 - Production"
    )
    config.addinivalue_line(
        "markers", "exercise(id): marks test with exercise ID (e.g., '1.2.1')"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test location."""
    for item in items:
        # Add level markers based on path
        if "1-foundations" in str(item.fspath):
            item.add_marker(pytest.mark.level1)
        elif "2-pipeline" in str(item.fspath):
            item.add_marker(pytest.mark.level2)
        elif "3-deployment" in str(item.fspath):
            item.add_marker(pytest.mark.level3)
        elif "4-production" in str(item.fspath):
            item.add_marker(pytest.mark.level4)


@pytest.fixture
def sample_texts():
    """Sample texts for testing preprocessing and prediction."""
    return [
        "This product is amazing! I love it.",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
        "GREAT SERVICE!!! Very happy :)",
        "the worst. never again.",
    ]


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing validation."""
    return ["positive", "negative", "neutral", "positive", "negative"]


@pytest.fixture
def sample_probabilities():
    """Sample probability predictions."""
    return [
        [0.1, 0.2, 0.7],  # positive
        [0.8, 0.1, 0.1],  # negative
        [0.2, 0.6, 0.2],  # neutral
        [0.05, 0.05, 0.9],  # positive
        [0.9, 0.05, 0.05],  # negative
    ]


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
