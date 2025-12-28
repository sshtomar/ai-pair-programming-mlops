# Lesson 2.4: Testing ML Code

## Learning Objectives

By the end of this lesson, students will:
1. Distinguish unit, integration, and ML-specific tests
2. Test data pipelines and model predictions
3. Implement behavioral testing for models

## Duration: 45 minutes

---

## Part 1: Why ML Testing is Different

### The Unique Challenges

Traditional software testing verifies deterministic outputs: `add(2, 3)` must return `5`. ML systems break these assumptions:

| Challenge | Example |
|-----------|---------|
| Non-determinism | Random seeds, GPU non-determinism, shuffle order |
| Threshold-based assertions | Accuracy > 0.85, not accuracy == 0.8734 |
| Data dependencies | Tests fail when data changes, not code |
| Expensive computation | Training takes hours, can't run on every commit |
| Behavioral expectations | "Similar inputs should give similar outputs" |

### What Accuracy Metrics Miss

A model can have 95% accuracy and still:
- Predict "positive" for "This product is NOT good" (negation blindness)
- Flip predictions on minor typos ("excelent" vs "excellent")
- Fail on edge cases that matter most to users
- Degrade silently when input distribution shifts

**Behavioral tests catch bugs that accuracy metrics miss.** This is the central insight of this lesson.

### The ML Testing Pyramid

```
ML Testing Pyramid: What to Test and When
═══════════════════════════════════════════════════════════════════════════════

                              ▲
                             ╱ ╲
                            ╱   ╲
                           ╱     ╲               MODEL VALIDATION
                          ╱       ╲              ─────────────────
                         ╱ Model   ╲             - Behavioral tests (negation, typos)
                        ╱ Validation╲            - Performance thresholds (acc > 0.85)
                       ╱             ╲           - Drift detection
                      ╱───────────────╲          - A/B test comparisons
                     ╱                 ╲
                    ╱   Data            ╲        DATA VALIDATION
                   ╱   Validation        ╲       ────────────────
                  ╱                       ╲      - Schema checks (columns, types)
                 ╱─────────────────────────╲     - Distribution checks
                ╱                           ╲    - Missing value limits
               ╱      Integration            ╲   - Class balance
              ╱         Tests                 ╲
             ╱─────────────────────────────────╲ INTEGRATION TESTS
            ╱                                   ╲ ─────────────────
           ╱                                     ╲ - Full pipeline: data → model → pred
          ╱            Unit Tests                 ╲ - Model serialization roundtrip
         ╱─────────────────────────────────────────╲
        ╱                                           ╲ UNIT TESTS
       ╱                                             ╲ ──────────
      ╱                                               ╲ - Data loading functions
     ╱                                                 ╲ - Text preprocessing
    ╱                                                   ╲ - Feature extraction
   ╱─────────────────────────────────────────────────────╲ - Input validation
  ╱                                                       ╲

  ═══════════════════════════════════════════════════════════════════════════

  When to Run Each Level:
  ┌─────────────────────┬────────────────────┬─────────────────────────────┐
  │ Test Type           │ When to Run        │ Speed                       │
  ├─────────────────────┼────────────────────┼─────────────────────────────┤
  │ Unit Tests          │ Every commit       │ Seconds (fast, many tests)  │
  │ Integration Tests   │ Every PR           │ Minutes                     │
  │ Data Validation     │ Before training    │ Minutes                     │
  │ Model Validation    │ After training     │ Minutes to hours            │
  └─────────────────────┴────────────────────┴─────────────────────────────┘
```

Run unit tests on every commit. Run integration tests on PR. Run ML-specific tests on model changes.

---

## Part 2: Unit Tests for Data and Preprocessing

### Project Structure for Tests

```
project/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Shared fixtures
│   ├── test_data.py        # Data loading tests
│   ├── test_features.py    # Feature engineering tests
│   └── test_model.py       # Model behavior tests
├── data/
│   └── sample/             # Small test dataset
│       └── test_reviews.csv
└── pytest.ini
```

### The Complete test_data.py

Create `tests/test_data.py`:

```python
"""Unit tests for data loading and preprocessing."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import (
    load_data,
    clean_text,
    validate_dataframe,
    split_data,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create a minimal valid dataframe for testing."""
    return pd.DataFrame({
        "text": [
            "Great product, love it!",
            "Terrible quality, broke immediately.",
            "It's okay, nothing special.",
            "  Extra whitespace  and CAPS  ",
            "",  # Empty string edge case
        ],
        "label": ["positive", "negative", "neutral", "positive", "neutral"],
    })


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """Write sample data to a temporary CSV file."""
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestLoadData:
    """Tests for the load_data function."""

    def test_load_valid_csv(self, sample_csv):
        """Should load a valid CSV file without errors."""
        df = load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "text" in df.columns
        assert "label" in df.columns

    def test_load_missing_file_raises(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_data(Path("/nonexistent/path.csv"))

    def test_load_preserves_columns(self, sample_csv):
        """Should preserve all expected columns."""
        df = load_data(sample_csv)
        expected_columns = {"text", "label"}
        assert expected_columns.issubset(set(df.columns))


# =============================================================================
# Text Cleaning Tests
# =============================================================================

class TestCleanText:
    """Tests for text preprocessing."""

    @pytest.mark.parametrize("input_text,expected", [
        ("Hello World", "hello world"),
        ("  extra  spaces  ", "extra spaces"),
        ("ALLCAPS", "allcaps"),
        ("MiXeD CaSe", "mixed case"),
        ("", ""),
        ("   ", ""),
    ])
    def test_clean_text_normalization(self, input_text, expected):
        """Should normalize text consistently."""
        result = clean_text(input_text)
        assert result == expected

    def test_clean_text_preserves_meaning(self):
        """Should not remove semantically important content."""
        text = "I don't like this product!"
        cleaned = clean_text(text)
        # Negation and punctuation should be handled appropriately
        assert "don" in cleaned or "not" in cleaned or "n't" in cleaned

    def test_clean_text_handles_unicode(self):
        """Should handle unicode characters gracefully."""
        text = "Great product! Cafe"
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestValidateDataframe:
    """Tests for dataframe validation."""

    def test_valid_dataframe_passes(self, sample_df):
        """Should accept valid dataframes without raising."""
        # Should not raise
        validate_dataframe(sample_df)

    def test_missing_text_column_raises(self):
        """Should raise ValueError if 'text' column is missing."""
        df = pd.DataFrame({"label": ["positive"]})
        with pytest.raises(ValueError, match="text"):
            validate_dataframe(df)

    def test_missing_label_column_raises(self):
        """Should raise ValueError if 'label' column is missing."""
        df = pd.DataFrame({"text": ["hello"]})
        with pytest.raises(ValueError, match="label"):
            validate_dataframe(df)

    def test_invalid_labels_raises(self):
        """Should raise ValueError for unexpected label values."""
        df = pd.DataFrame({
            "text": ["hello"],
            "label": ["invalid_label"],
        })
        with pytest.raises(ValueError, match="label"):
            validate_dataframe(df)

    def test_empty_dataframe_raises(self):
        """Should raise ValueError for empty dataframes."""
        df = pd.DataFrame({"text": [], "label": []})
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)


# =============================================================================
# Data Splitting Tests
# =============================================================================

class TestSplitData:
    """Tests for train/validation/test splitting."""

    def test_split_returns_three_sets(self, sample_df):
        """Should return train, validation, and test sets."""
        train, val, test = split_data(sample_df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_split_no_overlap(self, sample_df):
        """Train, validation, and test sets should not overlap."""
        train, val, test = split_data(sample_df)

        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_split_preserves_all_data(self, sample_df):
        """All original data should be in exactly one split."""
        train, val, test = split_data(sample_df)
        total_rows = len(train) + len(val) + len(test)
        assert total_rows == len(sample_df)

    def test_split_respects_ratios(self):
        """Should approximately respect specified split ratios."""
        # Create larger dataset for meaningful ratio testing
        df = pd.DataFrame({
            "text": [f"text_{i}" for i in range(100)],
            "label": ["positive"] * 50 + ["negative"] * 50,
        })

        train, val, test = split_data(df, train_ratio=0.7, val_ratio=0.15)

        # Allow 10% tolerance for small sample sizes
        assert 60 <= len(train) <= 80
        assert 10 <= len(val) <= 20
        assert 10 <= len(test) <= 20

    def test_split_reproducible_with_seed(self, sample_df):
        """Same seed should produce identical splits."""
        train1, val1, test1 = split_data(sample_df, random_state=42)
        train2, val2, test2 = split_data(sample_df, random_state=42)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)
```

### Key Testing Patterns

**1. Parametrized tests** for multiple input/output pairs:
```python
@pytest.mark.parametrize("input,expected", [...])
def test_function(input, expected):
    assert function(input) == expected
```

**2. Fixtures** for reusable test data:
```python
@pytest.fixture
def sample_df():
    return pd.DataFrame(...)
```

**3. Temporary files** for I/O tests:
```python
def test_load(tmp_path):
    csv_path = tmp_path / "test.csv"
    # tmp_path is automatically cleaned up
```

**4. Specific exception testing**:
```python
with pytest.raises(ValueError, match="expected substring"):
    function_that_should_fail()
```

---

## Part 3: Behavioral Tests for Models

Behavioral tests verify model behavior without checking exact predictions. They catch bugs that accuracy metrics miss.

### The Complete test_model.py

Create `tests/test_model.py`:

```python
"""Behavioral tests for the sentiment classifier.

These tests verify model behavior without checking exact predictions.
They catch bugs that aggregate accuracy metrics miss.
"""
import pytest
import numpy as np
from typing import List, Tuple

from src.predict import SentimentClassifier


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def classifier():
    """Load the trained classifier once for all tests in this module.

    Using module scope avoids reloading the model for every test,
    which would be slow.
    """
    return SentimentClassifier.load("models/sentiment_model.pkl")


@pytest.fixture
def predict(classifier):
    """Convenience fixture that returns the predict function."""
    return classifier.predict


# =============================================================================
# Negation Tests (Directional Expectations)
# =============================================================================

class TestNegationHandling:
    """Model should understand that negation reverses sentiment.

    This is a CRITICAL capability. Many models fail these tests
    because bag-of-words features miss negation scope.
    """

    NEGATION_PAIRS = [
        ("This product is good", "This product is not good"),
        ("I love this", "I do not love this"),
        ("Great quality", "Not great quality"),
        ("Would recommend", "Would not recommend"),
        ("Excellent service", "Not excellent service"),
    ]

    @pytest.mark.parametrize("positive,negated", NEGATION_PAIRS)
    def test_negation_changes_prediction(self, predict, positive, negated):
        """Negating a positive statement should not predict positive."""
        pos_pred = predict(positive)
        neg_pred = predict(negated)

        # The negated version should NOT have the same sentiment
        # as the positive version
        assert pos_pred != neg_pred, (
            f"Model failed to detect negation: "
            f"'{positive}' -> {pos_pred}, "
            f"'{negated}' -> {neg_pred}"
        )

    def test_double_negation(self, predict):
        """Double negation should return to original sentiment."""
        original = "This is a good product"
        double_neg = "This is not a bad product"

        # Both should be positive (or at least same sentiment)
        orig_pred = predict(original)
        double_pred = predict(double_neg)

        # This is a harder test - many models fail it
        # Mark as expected failure if your model isn't there yet
        assert orig_pred == double_pred or orig_pred == "positive"


# =============================================================================
# Invariance Tests
# =============================================================================

class TestInvariance:
    """Model should be robust to superficial input changes.

    These tests verify that irrelevant modifications don't flip predictions.
    """

    def test_case_invariance(self, predict):
        """Predictions should not change based on capitalization."""
        texts = [
            "great product",
            "GREAT PRODUCT",
            "Great Product",
            "gReAt PrOdUcT",
        ]
        predictions = [predict(t) for t in texts]

        # All should give the same prediction
        assert len(set(predictions)) == 1, (
            f"Case sensitivity detected: {list(zip(texts, predictions))}"
        )

    def test_typo_robustness(self, predict):
        """Minor typos should not flip predictions."""
        pairs = [
            ("excellent product", "excelent product"),
            ("terrible quality", "terribe quality"),
            ("highly recommend", "highly recomend"),
        ]

        for correct, typo in pairs:
            correct_pred = predict(correct)
            typo_pred = predict(typo)

            assert correct_pred == typo_pred, (
                f"Typo sensitivity: '{correct}' -> {correct_pred}, "
                f"'{typo}' -> {typo_pred}"
            )

    def test_whitespace_invariance(self, predict):
        """Extra whitespace should not change predictions."""
        base = "good product"
        variations = [
            "good product",
            "  good product  ",
            "good  product",
            "good\tproduct",
        ]

        predictions = [predict(v) for v in variations]
        assert len(set(predictions)) == 1, (
            f"Whitespace sensitivity: {list(zip(variations, predictions))}"
        )

    def test_punctuation_robustness(self, predict):
        """Punctuation variations should not flip predictions."""
        pairs = [
            ("Great product!", "Great product"),
            ("Terrible...", "Terrible"),
            ("Love it!!!", "Love it"),
        ]

        for with_punct, without_punct in pairs:
            pred1 = predict(with_punct)
            pred2 = predict(without_punct)

            assert pred1 == pred2, (
                f"Punctuation sensitivity: '{with_punct}' -> {pred1}, "
                f"'{without_punct}' -> {pred2}"
            )


# =============================================================================
# Confidence Tests
# =============================================================================

class TestConfidence:
    """Model confidence should correlate with prediction difficulty."""

    @pytest.fixture
    def predict_proba(self, classifier):
        """Return function that gives prediction probabilities."""
        return classifier.predict_proba

    def test_clear_sentiment_high_confidence(self, predict_proba):
        """Clearly positive/negative text should have high confidence."""
        clear_examples = [
            "This is absolutely wonderful, I love everything about it!",
            "Terrible product, complete waste of money, avoid at all costs.",
        ]

        for text in clear_examples:
            probs = predict_proba(text)
            max_prob = max(probs.values())
            assert max_prob > 0.7, (
                f"Low confidence ({max_prob:.2f}) on clear example: '{text}'"
            )

    def test_ambiguous_lower_confidence(self, predict_proba):
        """Ambiguous text should have lower confidence."""
        ambiguous = "It's okay I guess, has some good and bad parts."

        probs = predict_proba(ambiguous)
        max_prob = max(probs.values())

        # Ambiguous text shouldn't have extremely high confidence
        assert max_prob < 0.95, (
            f"Suspiciously high confidence ({max_prob:.2f}) on ambiguous text"
        )


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Model should handle edge cases gracefully."""

    def test_empty_string(self, classifier):
        """Empty string should not crash, should return valid prediction."""
        result = classifier.predict("")
        assert result in ["positive", "negative", "neutral"]

    def test_very_long_text(self, classifier):
        """Very long text should not crash."""
        long_text = "good product " * 1000
        result = classifier.predict(long_text)
        assert result in ["positive", "negative", "neutral"]

    def test_special_characters(self, classifier):
        """Special characters should not crash the model."""
        special = "@#$%^&*() 12345 good product []{}|"
        result = classifier.predict(special)
        assert result in ["positive", "negative", "neutral"]

    def test_unicode_text(self, classifier):
        """Unicode characters should be handled gracefully."""
        unicode_text = "Great product! Cafe"
        result = classifier.predict(unicode_text)
        assert result in ["positive", "negative", "neutral"]

    def test_only_stopwords(self, classifier):
        """Text with only stopwords should return valid prediction."""
        stopwords_only = "the a an is are was were"
        result = classifier.predict(stopwords_only)
        assert result in ["positive", "negative", "neutral"]


# =============================================================================
# Performance Thresholds
# =============================================================================

class TestPerformanceThresholds:
    """Model must meet minimum performance requirements."""

    @pytest.fixture(scope="class")
    def test_data(self):
        """Load held-out test dataset."""
        import pandas as pd
        return pd.read_csv("data/test_holdout.csv")

    def test_minimum_accuracy(self, classifier, test_data):
        """Model must achieve at least 80% accuracy on test set."""
        predictions = [classifier.predict(text) for text in test_data["text"]]
        accuracy = (predictions == test_data["label"]).mean()

        assert accuracy >= 0.80, (
            f"Accuracy {accuracy:.2%} below minimum threshold of 80%"
        )

    def test_minimum_per_class_recall(self, classifier, test_data):
        """Each class must have at least 70% recall."""
        from sklearn.metrics import recall_score

        predictions = [classifier.predict(text) for text in test_data["text"]]

        for label in ["positive", "negative", "neutral"]:
            mask = test_data["label"] == label
            if mask.sum() == 0:
                continue

            class_preds = [p for p, m in zip(predictions, mask) if m]
            class_labels = test_data.loc[mask, "label"].tolist()

            correct = sum(p == l for p, l in zip(class_preds, class_labels))
            recall = correct / len(class_labels)

            assert recall >= 0.70, (
                f"Recall for '{label}' is {recall:.2%}, below 70% threshold"
            )

    def test_prediction_latency(self, classifier):
        """Single prediction must complete within 100ms."""
        import time

        text = "This is a test product review for latency testing."

        start = time.perf_counter()
        classifier.predict(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, (
            f"Prediction took {elapsed_ms:.1f}ms, exceeding 100ms threshold"
        )
```

### Why Behavioral Tests Matter

Consider this scenario:

```python
# Model A: 92% accuracy
# - Fails on 100% of negation cases
# - Fails on 50% of typo cases

# Model B: 89% accuracy
# - Handles negation correctly
# - Robust to typos
```

**Model B is better for production** even though it has lower accuracy. Behavioral tests catch this; aggregate metrics don't.

---

## Part 4: Integration Tests

Integration tests verify the full pipeline works end-to-end.

### The Complete test_pipeline.py

Create `tests/test_pipeline.py`:

```python
"""Integration tests for the full ML pipeline."""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.data import load_data, clean_text, split_data
from src.features import extract_features
from src.train import train_model
from src.predict import SentimentClassifier


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture(scope="class")
    def pipeline_output(self, tmp_path_factory):
        """Run full pipeline once, share results across tests."""
        work_dir = tmp_path_factory.mktemp("pipeline")

        # Create minimal training data
        train_data = pd.DataFrame({
            "text": [
                "excellent product love it",
                "great quality highly recommend",
                "amazing wonderful fantastic",
                "terrible awful waste of money",
                "horrible bad do not buy",
                "disappointing poor quality",
                "okay average nothing special",
                "mediocre neither good nor bad",
                "fine acceptable adequate",
            ] * 10,  # Repeat to have enough data
            "label": (["positive"] * 3 + ["negative"] * 3 + ["neutral"] * 3) * 10,
        })

        # Run pipeline
        train_df, val_df, test_df = split_data(train_data, random_state=42)

        X_train = extract_features(train_df["text"].tolist())
        X_val = extract_features(val_df["text"].tolist())

        model = train_model(
            X_train,
            train_df["label"].tolist(),
            X_val,
            val_df["label"].tolist(),
        )

        model_path = work_dir / "model.pkl"
        model.save(model_path)

        return {
            "model_path": model_path,
            "test_df": test_df,
            "train_size": len(train_df),
        }

    def test_model_file_created(self, pipeline_output):
        """Pipeline should create a model file."""
        assert pipeline_output["model_path"].exists()

    def test_model_loadable(self, pipeline_output):
        """Saved model should be loadable."""
        model = SentimentClassifier.load(pipeline_output["model_path"])
        assert model is not None

    def test_loaded_model_predicts(self, pipeline_output):
        """Loaded model should make predictions."""
        model = SentimentClassifier.load(pipeline_output["model_path"])
        prediction = model.predict("test text")
        assert prediction in ["positive", "negative", "neutral"]

    def test_predictions_on_test_set(self, pipeline_output):
        """Model should predict on unseen test data."""
        model = SentimentClassifier.load(pipeline_output["model_path"])
        test_df = pipeline_output["test_df"]

        predictions = [model.predict(text) for text in test_df["text"]]

        # All predictions should be valid labels
        assert all(p in ["positive", "negative", "neutral"] for p in predictions)

        # Should get some predictions right (at least better than random)
        accuracy = sum(
            p == l for p, l in zip(predictions, test_df["label"])
        ) / len(predictions)

        assert accuracy > 0.33, f"Accuracy {accuracy:.2%} not better than random"


class TestDataPipelineIntegration:
    """Test data flows correctly through preprocessing."""

    def test_data_survives_full_preprocessing(self):
        """Data should maintain integrity through preprocessing."""
        raw_texts = [
            "  GREAT Product!!!  ",
            "terrible, awful, bad",
            "it's okay I guess",
        ]

        # Full preprocessing pipeline
        cleaned = [clean_text(t) for t in raw_texts]
        features = extract_features(cleaned)

        # Should have same number of samples
        assert features.shape[0] == len(raw_texts)

        # Features should be numeric
        assert features.dtype in [float, 'float64', 'float32']

        # No NaN values
        assert not pd.isna(features).any()
```

---

## Part 5: Data Validation Tests

Validate data quality before training.

### Simple Data Validation

Create `tests/test_data_quality.py`:

```python
"""Data quality validation tests.

Run these tests before training to catch data issues early.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataQuality:
    """Validate training data quality."""

    @pytest.fixture
    def training_data(self):
        """Load the training dataset."""
        return pd.read_csv("data/train.csv")

    def test_no_missing_text(self, training_data):
        """Text column should have no missing values."""
        missing = training_data["text"].isna().sum()
        assert missing == 0, f"Found {missing} missing text values"

    def test_no_missing_labels(self, training_data):
        """Label column should have no missing values."""
        missing = training_data["label"].isna().sum()
        assert missing == 0, f"Found {missing} missing labels"

    def test_valid_labels_only(self, training_data):
        """Labels should only be positive, negative, or neutral."""
        valid_labels = {"positive", "negative", "neutral"}
        actual_labels = set(training_data["label"].unique())

        invalid = actual_labels - valid_labels
        assert len(invalid) == 0, f"Invalid labels found: {invalid}"

    def test_minimum_samples_per_class(self, training_data):
        """Each class should have at least 100 samples."""
        min_samples = 100
        counts = training_data["label"].value_counts()

        for label, count in counts.items():
            assert count >= min_samples, (
                f"Class '{label}' has only {count} samples, need {min_samples}"
            )

    def test_class_balance(self, training_data):
        """No class should be more than 3x larger than another."""
        counts = training_data["label"].value_counts()
        ratio = counts.max() / counts.min()

        assert ratio <= 3.0, (
            f"Class imbalance ratio {ratio:.1f}x exceeds 3x threshold"
        )

    def test_no_duplicate_texts(self, training_data):
        """Should not have exact duplicate texts."""
        duplicates = training_data["text"].duplicated().sum()
        total = len(training_data)
        dup_ratio = duplicates / total

        assert dup_ratio < 0.01, (
            f"{duplicates} duplicates ({dup_ratio:.1%}) exceeds 1% threshold"
        )

    def test_text_length_distribution(self, training_data):
        """Text lengths should be reasonable."""
        lengths = training_data["text"].str.len()

        # No extremely short texts
        too_short = (lengths < 10).sum()
        assert too_short == 0, f"{too_short} texts shorter than 10 characters"

        # No extremely long texts
        too_long = (lengths > 5000).sum()
        assert too_long == 0, f"{too_long} texts longer than 5000 characters"

    def test_no_obvious_data_leakage(self, training_data):
        """Labels should not appear in text."""
        for label in ["positive", "negative", "neutral"]:
            contains_label = training_data["text"].str.lower().str.contains(
                f"label: {label}|sentiment: {label}",
                regex=True
            ).sum()

            assert contains_label == 0, (
                f"Found {contains_label} texts containing label '{label}'"
            )
```

---

## Part 6: Pre-commit Hooks

Automate testing before every commit.

### Install pre-commit

```bash
pip install pre-commit
```

### Create .pre-commit-config.yaml

```yaml
# .pre-commit-config.yaml
repos:
  # Code formatting
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        language_version: python3

  # Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  # Linting
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203"]

  # Type checking (optional but recommended)
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pandas-stubs, types-requests]
        args: ["--ignore-missing-imports"]

  # Run fast tests
  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest (fast tests only)
        entry: pytest tests/ -m "not slow" --tb=short -q
        language: system
        pass_filenames: false
        always_run: true
```

### Configure pytest markers

Create `pytest.ini`:

```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    behavioral: marks tests as behavioral/model tests

testpaths = tests
python_files = test_*.py
python_functions = test_*

# Show extra test summary info
addopts = -v --tb=short
```

### Mark slow tests

```python
@pytest.mark.slow
def test_full_training_pipeline():
    """This test takes minutes to run."""
    ...

@pytest.mark.slow
@pytest.mark.integration
def test_end_to_end():
    """Full integration test."""
    ...
```

### Install hooks

```bash
pre-commit install
```

Now every `git commit` will:
1. Format code with black
2. Sort imports with isort
3. Lint with flake8
4. Run fast tests with pytest

---

## Part 7: Shared Test Fixtures

### The conftest.py File

Create `tests/conftest.py`:

```python
"""Shared fixtures for all tests.

Fixtures defined here are automatically available to all test files.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_reviews():
    """Minimal set of reviews for quick tests."""
    return [
        ("Excellent product!", "positive"),
        ("Terrible, avoid!", "negative"),
        ("It's okay.", "neutral"),
    ]


@pytest.fixture
def sample_df(sample_reviews):
    """Sample reviews as a DataFrame."""
    texts, labels = zip(*sample_reviews)
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def trained_model():
    """Load trained model once per test session.

    Session scope means this expensive operation happens only once,
    and the model is shared across all tests.
    """
    from src.predict import SentimentClassifier

    model_path = Path("models/sentiment_model.pkl")
    if not model_path.exists():
        pytest.skip("Trained model not found. Run training first.")

    return SentimentClassifier.load(model_path)


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory, cleaned up after test."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "behavioral: mark as behavioral test")
```

---

## Exercises

### Exercise 2.4.1: Write a Bug-Catching Test

There's a bug in this preprocessing function:

```python
def clean_text(text: str) -> str:
    """Clean text for model input."""
    text = text.lower()
    text = text.strip()
    # Bug: this removes "not" from text!
    text = " ".join(word for word in text.split() if len(word) > 3)
    return text
```

**Task:** Write a test that fails because of this bug.

<details>
<summary>Hint</summary>

The function removes short words. "not" is 3 characters. What happens to "This is not good"?

</details>

<details>
<summary>Solution</summary>

```python
def test_clean_text_preserves_negation():
    """Negation words like 'not' must be preserved."""
    text = "This is not good"
    cleaned = clean_text(text)

    # "not" should still be in the text
    assert "not" in cleaned, (
        f"Negation lost: '{text}' became '{cleaned}'"
    )
```

This test will fail because `len("not") == 3`, and the filter requires `len(word) > 3`.

</details>

### Exercise 2.4.2: Add Invariance Tests

Add three more invariance tests to `test_model.py`:

1. **Contraction invariance**: "don't" vs "do not" should give same prediction
2. **Synonym invariance**: "good" vs "great" (both positive) should give same prediction
3. **Emoji removal**: "Great product!" vs "Great product" should give same prediction

### Exercise 2.4.3: Set Up Pre-commit

1. Install pre-commit: `pip install pre-commit`
2. Create `.pre-commit-config.yaml` in your project root
3. Run `pre-commit install`
4. Make a commit and verify hooks run

---

## Key Takeaways

1. **Behavioral tests catch bugs that accuracy metrics miss.** A model can have 95% accuracy and still fail on negation, typos, and edge cases.

2. **The testing pyramid applies to ML.** Unit tests (fast, many) at the bottom, integration tests in the middle, ML-specific tests (slow, few) at the top.

3. **Test data quality before training.** Bad data produces bad models. Validate early.

4. **Use fixtures for expensive operations.** Load models once with `scope="session"`, not per test.

5. **Automate with pre-commit.** Fast tests run on every commit. Slow tests run in CI.

6. **Mark slow tests.** Use `@pytest.mark.slow` so developers can skip them locally with `-m "not slow"`.

---

## Next Steps

**Level 2 complete!** You now have:
- Version-controlled data (DVC)
- Tracked experiments (MLflow)
- Registered models with lifecycle stages
- Comprehensive test suite

Run `/start-3-1` to begin Level 3: Deployment. You'll learn how to serve your tested, versioned model as a production API.
