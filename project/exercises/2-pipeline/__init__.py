"""
Level 2: Pipeline Exercises
===========================

This module contains exercises for the Pipeline level of the MLOps course.

Exercises:
- 2.1 Data Versioning: Content-addressable hashing and DVC pipelines
- 2.2 Experiment Tracking: MLflow logging wrappers and run management
- 2.3 Model Registry: Model promotion and lifecycle management
- 2.4 Testing ML Code: Output validation and property-based testing

Run tests with:
    pytest project/exercises/2-pipeline/test_level_2.py -v

Exercise distribution:
- 50% Write & Verify: Implement core functionality
- 30% Debug: Fix broken code
- 20% Extend: Add features to existing code
"""

from pathlib import Path

EXERCISES_DIR = Path(__file__).parent
