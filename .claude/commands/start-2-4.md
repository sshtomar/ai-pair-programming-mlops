# Lesson 2.4: Testing ML Code

Read the lesson content from `lesson-modules/2-pipeline/2-4-testing-ml.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"Your model passes accuracy thresholds. Ship it? What if it predicts 'positive' for every input containing the word 'good'—even 'this is not good'?"

### 2. Socratic Question
Ask: "What's different about testing ML code compared to regular software?"

Guide them to: non-determinism, data dependencies, behavioral properties, threshold-based assertions.

### 3. Testing Pyramid for ML (10 min)
Cover the layers:

**Unit Tests**
- Data loading functions
- Feature transformations
- Preprocessing logic

**Integration Tests**
- Full pipeline execution
- Model loading and prediction
- API endpoints

**ML-Specific Tests**
- Behavioral tests (invariance, directional)
- Data validation tests
- Model performance tests

### 4. Write Unit Tests (15 min)
Guide them to create `tests/test_data.py`:
```python
import pytest
from src.data import load_data, preprocess_text

def test_load_data_returns_dataframe():
    df = load_data("data/reviews.csv")
    assert len(df) > 0
    assert "text" in df.columns
    assert "label" in df.columns

def test_preprocess_removes_html():
    result = preprocess_text("<p>Great product!</p>")
    assert "<p>" not in result
    assert "Great product" in result

def test_preprocess_handles_empty():
    result = preprocess_text("")
    assert result == ""
```

### 5. Behavioral Tests (15 min)
Introduce behavioral testing for models:
```python
def test_negation_changes_prediction():
    """Model should handle negation correctly."""
    positive = model.predict(["This movie is good"])
    negative = model.predict(["This movie is not good"])
    assert positive != negative, "Model ignores negation"

def test_invariance_to_typos():
    """Minor typos shouldn't flip predictions."""
    original = model.predict(["Excellent service"])
    typo = model.predict(["Excelent service"])
    assert original == typo
```

### 6. Pre-commit Hooks (5 min)
Set up automatic testing:
```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/ -v
        language: system
        pass_filenames: false
```

### 7. Wrap Up
- Level 2 complete: versioned data, tracked experiments, registered models, tested code
- Preview Level 3: Now we deploy this tested model
- Next: `/start-3-1` for serving options

## Teaching Notes
- Behavioral tests catch bugs accuracy metrics miss
- Testing isn't optional for production ML
- Connect to CI/CD in Level 4
