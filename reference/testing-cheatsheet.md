# pytest Cheatsheet for ML

## Quick Reference

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest tests/test_model.py` | Run specific file |
| `pytest -k "accuracy"` | Run tests matching pattern |
| `pytest -v` | Verbose output |
| `pytest -x` | Stop on first failure |
| `pytest --tb=short` | Shorter tracebacks |
| `pytest -s` | Show print statements |
| `pytest --cov=src` | Run with coverage |
| `pytest -m "not slow"` | Skip slow-marked tests |

## Test Structure

### Basic Test File

```python
# tests/test_model.py
import pytest
from src.model import SentimentClassifier

class TestSentimentClassifier:
    """Group related tests in a class."""

    def test_predict_returns_valid_label(self, trained_model):
        """Test names should describe expected behavior."""
        result = trained_model.predict(["great product"])
        assert result[0] in ["positive", "negative"]

    def test_predict_batch_matches_input_length(self, trained_model):
        texts = ["good", "bad", "okay"]
        results = trained_model.predict(texts)
        assert len(results) == len(texts)
```

### Fixtures (Shared Setup)

```python
# tests/conftest.py
import pytest
import pandas as pd
from src.model import SentimentClassifier

@pytest.fixture
def sample_data():
    """Simple fixture returning data."""
    return pd.DataFrame({
        "text": ["great", "terrible", "okay"],
        "label": ["positive", "negative", "neutral"]
    })

@pytest.fixture(scope="module")
def trained_model(sample_data):
    """Expensive fixture - reused across module."""
    model = SentimentClassifier()
    model.fit(sample_data["text"], sample_data["label"])
    return model

@pytest.fixture
def tmp_model_path(tmp_path):
    """Use pytest's tmp_path for temp files."""
    return tmp_path / "model.pkl"
```

## Common Patterns for ML

### Test Data Shapes

```python
def test_preprocessing_output_shape(preprocessor, sample_data):
    result = preprocessor.transform(sample_data)
    assert result.shape[0] == sample_data.shape[0]
    assert result.shape[1] == preprocessor.n_features
```

### Test Model Predictions

```python
def test_model_predicts_correct_type(trained_model):
    predictions = trained_model.predict(["test input"])
    assert isinstance(predictions, list)
    assert all(isinstance(p, str) for p in predictions)

def test_model_probabilities_sum_to_one(trained_model):
    probs = trained_model.predict_proba(["test input"])
    assert abs(sum(probs[0]) - 1.0) < 1e-6
```

### Test Model Performance

```python
@pytest.mark.slow
def test_model_accuracy_above_threshold(trained_model, test_data):
    """Mark slow tests to skip in CI."""
    predictions = trained_model.predict(test_data["text"])
    accuracy = (predictions == test_data["label"]).mean()
    assert accuracy > 0.7, f"Accuracy {accuracy} below threshold"
```

### Test Edge Cases

```python
@pytest.mark.parametrize("input_text,expected", [
    ("", "neutral"),           # Empty string
    ("   ", "neutral"),        # Whitespace only
    ("!@#$%", "neutral"),      # Special characters
    ("a" * 10000, "neutral"),  # Very long input
])
def test_edge_cases(trained_model, input_text, expected):
    result = trained_model.predict([input_text])
    assert result[0] == expected
```

### Test Model Serialization

```python
def test_model_save_and_load(trained_model, tmp_model_path):
    # Save
    trained_model.save(tmp_model_path)
    assert tmp_model_path.exists()

    # Load
    loaded = SentimentClassifier.load(tmp_model_path)

    # Verify same predictions
    test_input = ["test text"]
    assert trained_model.predict(test_input) == loaded.predict(test_input)
```

### Test Error Handling

```python
def test_predict_raises_on_none_input(trained_model):
    with pytest.raises(ValueError, match="Input cannot be None"):
        trained_model.predict(None)

def test_fit_raises_on_mismatched_lengths(model):
    with pytest.raises(ValueError):
        model.fit(["text1", "text2"], ["label1"])  # Mismatched lengths
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `fixture not found` | Add to `conftest.py` or import properly |
| `module not found` | Add `__init__.py` to test directory |
| `tests not discovered` | Name files `test_*.py` and functions `test_*` |
| `fixture scope error` | Higher-scope fixtures can't use lower-scope ones |
| `slow tests in CI` | Mark with `@pytest.mark.slow` and skip with `-m "not slow"` |

## Best Practices

1. **Test behavior, not implementation**: Don't test private methods
2. **Use fixtures for setup**: Keep tests focused on assertions
3. **Parametrize similar tests**: Reduce code duplication
4. **Mark slow tests**: Use `@pytest.mark.slow` for integration tests
5. **Test the contract**: Input types, output types, error conditions
6. **Use tmp_path**: For any file I/O in tests
