# Lesson 1.3: Your First Model

## Learning Objectives

By the end of this lesson, students will:
1. Build a sentiment classifier from scratch
2. Structure ML code for production (not notebooks)
3. Implement proper train/validation/test splits

## Duration: 45 minutes

---

## Part 1: Why Not Notebooks?

### The Notebook Problem

Notebooks are exploration tools. Production requires something different.

| Aspect | Notebook | Production Python |
|--------|----------|-------------------|
| State management | Hidden, order-dependent | Explicit, reproducible |
| Testing | Nearly impossible | Standard pytest/unittest |
| Version control | JSON diffs unreadable | Clean line-by-line diffs |
| Code reuse | Copy-paste between notebooks | Import modules |
| Error handling | Interactive debugging | Logged, monitored, recoverable |
| CI/CD | Complex workarounds | Standard tooling |

### Real Costs of Notebook-to-Production

Converting a notebook to production code typically takes 2-4x longer than writing production code from the start. Common issues:

1. **Global variables** that should be function parameters
2. **Hardcoded paths** like `/Users/alice/data/file.csv`
3. **Missing error handling** for file I/O, network calls, malformed data
4. **Implicit dependencies** on cell execution order
5. **No configuration management** for hyperparameters

### The Production Mindset

Write code that:
- **Runs headlessly**: No interactive prompts, no manual intervention
- **Fails loudly**: Explicit errors with context, not silent corruption
- **Is testable**: Functions with clear inputs/outputs
- **Is configurable**: No hardcoded values in logic

---

## Part 2: Production Code Structure

### Project Layout

```
project/
├── src/
│   ├── __init__.py        # Makes src a package
│   ├── data.py            # Data loading and preprocessing
│   ├── features.py        # Feature engineering (TF-IDF)
│   ├── train.py           # Training logic and CLI
│   └── predict.py         # Inference logic
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_train.py
├── data/
│   └── .gitkeep           # Placeholder for data files
├── models/
│   └── .gitkeep           # Placeholder for saved models
├── config/
│   └── training_config.yaml
└── requirements.txt
```

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `data.py` | Load, validate, split data | pandas, pathlib |
| `features.py` | Transform text to features | scikit-learn |
| `train.py` | Train model, save artifacts | data.py, features.py, joblib |
| `predict.py` | Load model, make predictions | features.py, joblib |

### Why This Structure?

1. **Testability**: Each module can be unit tested independently
2. **Reusability**: `predict.py` imports shared code, doesn't duplicate
3. **Deployability**: `predict.py` can be packaged alone for inference
4. **Clarity**: New team members know where to find things

---

## Part 3: Building the Classifier

### The ML Pipeline at a Glance

Before diving into code, here's what we're building:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Text Classification Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────┘

     Raw Text           Preprocessing        TF-IDF Vectorizer      Classifier
         │                    │                     │                    │
         ▼                    ▼                     ▼                    ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────┐
│ "Great job!" │ ───▶ │ "great job"  │ ───▶ │ Sparse Matrix│ ───▶ │ Logistic │
│              │      │              │      │ [0.0, 0.7,   │      │Regression│
│              │      │  lowercase   │      │  0.0, 0.3,   │      │          │
│              │      │  strip ws    │      │  0.0, ...]   │      │          │
└──────────────┘      └──────────────┘      └──────────────┘      └────┬─────┘
                                                                       │
                                            ┌──────────────────────────┘
                                            ▼
                                     ┌─────────────┐
                                     │  Prediction │
                                     │  "positive" │
                                     │  conf: 94%  │
                                     └─────────────┘

         data.py              data.py            features.py         train.py
       preprocess_text()    preprocess_text()   TextFeaturizer      train_model()
```

Each module handles one step, making the code testable and reusable.

### Step 1: Data Module (`src/data.py`)

```python
"""Data loading and preprocessing for sentiment classification."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: Path) -> pd.DataFrame:
    """Load sentiment data from CSV.

    Args:
        filepath: Path to CSV with 'text' and 'label' columns.

    Returns:
        DataFrame with validated columns.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def preprocess_text(text: str) -> str:
    """Clean text for classification.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text (lowercase, stripped whitespace).
    """
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets with stratification.

    Args:
        df: DataFrame with 'text' and 'label' columns.
        test_size: Fraction for test set (default 0.2 = 20%).
        val_size: Fraction for validation set (default 0.1 = 10%).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Note:
        With defaults, split is 70% train, 10% val, 20% test.
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    # Second split: separate validation from training
    # Adjust val_size relative to train_val size
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val_df["label"],
    )

    return train_df, val_df, test_df
```

### Step 2: Features Module (`src/features.py`)

```python
"""Feature engineering for text classification."""

from pathlib import Path
from typing import Union

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeaturizer:
    """TF-IDF based text featurizer.

    Attributes:
        vectorizer: Fitted TfidfVectorizer or None if not fitted.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
    ) -> None:
        """Initialize featurizer with TF-IDF parameters.

        Args:
            max_features: Maximum vocabulary size.
            ngram_range: (min_n, max_n) for n-gram extraction.
            min_df: Minimum document frequency for terms.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> "TextFeaturizer":
        """Fit the vectorizer on training texts.

        Args:
            texts: List of training documents.

        Returns:
            Self for method chaining.
        """
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: list[str]) -> spmatrix:
        """Transform texts to TF-IDF features.

        Args:
            texts: List of documents to transform.

        Returns:
            Sparse matrix of TF-IDF features.

        Raises:
            RuntimeError: If transform called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError("Featurizer must be fitted before transform")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]) -> spmatrix:
        """Fit and transform in one step.

        Args:
            texts: List of training documents.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        self._is_fitted = True
        return self.vectorizer.fit_transform(texts)

    def save(self, filepath: Path) -> None:
        """Save fitted vectorizer to disk.

        Args:
            filepath: Path to save the vectorizer.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted featurizer")
        joblib.dump(self.vectorizer, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "TextFeaturizer":
        """Load a fitted vectorizer from disk.

        Args:
            filepath: Path to saved vectorizer.

        Returns:
            TextFeaturizer with loaded vectorizer.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Vectorizer not found: {filepath}")

        instance = cls()
        instance.vectorizer = joblib.load(filepath)
        instance._is_fitted = True
        return instance

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the fitted vocabulary."""
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
```

### Step 3: Training Module (`src/train.py`)

```python
"""Training script for sentiment classifier."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import load_data, preprocess_text, split_data
from src.features import TextFeaturizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a logistic regression classifier.

    Args:
        X_train: Training features (sparse or dense matrix).
        y_train: Training labels.
        C: Regularization strength (smaller = stronger regularization).
        max_iter: Maximum iterations for solver.

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "dataset",
) -> Dict[str, float]:
    """Evaluate model and return metrics.

    Args:
        model: Fitted classifier.
        X: Feature matrix.
        y: True labels.
        dataset_name: Name for logging.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro"),
        "recall_macro": recall_score(y, y_pred, average="macro"),
        "f1_macro": f1_score(y, y_pred, average="macro"),
    }

    logger.info(f"\n{dataset_name} Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"\n{classification_report(y, y_pred)}")

    return metrics


def save_artifacts(
    model: LogisticRegression,
    featurizer: TextFeaturizer,
    metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save model, featurizer, and metadata.

    Args:
        model: Trained classifier.
        featurizer: Fitted text featurizer.
        metrics: Training metrics to save.
        output_dir: Directory for artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save featurizer
    featurizer_path = output_dir / "featurizer.joblib"
    featurizer.save(featurizer_path)
    logger.info(f"Featurizer saved to {featurizer_path}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "vocabulary_size": featurizer.vocabulary_size,
        "metrics": metrics,
        "model_type": "LogisticRegression",
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum vocabulary size for TF-IDF",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Regularization strength (C parameter)",
    )
    args = parser.parse_args()

    # Load and preprocess data
    logger.info(f"Loading data from {args.data_path}")
    df = load_data(args.data_path)
    df["text"] = df["text"].apply(preprocess_text)
    logger.info(f"Loaded {len(df)} samples")

    # Split data
    train_df, val_df, test_df = split_data(df)
    logger.info(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Create features
    logger.info("Creating TF-IDF features")
    featurizer = TextFeaturizer(max_features=args.max_features)
    X_train = featurizer.fit_transform(train_df["text"].tolist())
    X_val = featurizer.transform(val_df["text"].tolist())
    X_test = featurizer.transform(test_df["text"].tolist())

    logger.info(f"Vocabulary size: {featurizer.vocabulary_size}")

    # Train model
    logger.info("Training model")
    model = train_model(
        X_train,
        train_df["label"].values,
        C=args.regularization,
    )

    # Evaluate
    train_metrics = evaluate_model(model, X_train, train_df["label"].values, "Training")
    val_metrics = evaluate_model(model, X_val, val_df["label"].values, "Validation")
    test_metrics = evaluate_model(model, X_test, test_df["label"].values, "Test")

    # Save artifacts
    all_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_artifacts(model, featurizer, all_metrics, args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
```

### Step 4: Prediction Module (`src/predict.py`)

```python
"""Inference module for sentiment classification."""

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

from src.data import preprocess_text
from src.features import TextFeaturizer


class SentimentClassifier:
    """Sentiment classifier for inference.

    Loads a trained model and featurizer to make predictions.
    """

    def __init__(self, model_dir: Path) -> None:
        """Initialize classifier from saved artifacts.

        Args:
            model_dir: Directory containing model.joblib and featurizer.joblib.

        Raises:
            FileNotFoundError: If required artifacts are missing.
        """
        model_path = model_dir / "model.joblib"
        featurizer_path = model_dir / "featurizer.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.featurizer = TextFeaturizer.load(featurizer_path)

    def predict(self, texts: List[str]) -> List[str]:
        """Predict sentiment for a list of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of predicted labels.
        """
        # Preprocess
        cleaned = [preprocess_text(t) for t in texts]

        # Featurize
        X = self.featurizer.transform(cleaned)

        # Predict
        predictions = self.model.predict(X)
        return predictions.tolist()

    def predict_proba(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment with confidence scores.

        Args:
            texts: List of input texts.

        Returns:
            List of (label, confidence) tuples.
        """
        # Preprocess
        cleaned = [preprocess_text(t) for t in texts]

        # Featurize
        X = self.featurizer.transform(cleaned)

        # Get probabilities
        probas = self.model.predict_proba(X)
        predictions = self.model.predict(X)

        # Get confidence for predicted class
        results = []
        for pred, proba in zip(predictions, probas):
            confidence = float(np.max(proba))
            results.append((pred, confidence))

        return results


def main() -> None:
    """CLI for making predictions."""
    parser = argparse.ArgumentParser(description="Predict sentiment")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model artifacts",
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="Text(s) to classify",
    )
    args = parser.parse_args()

    classifier = SentimentClassifier(args.model_dir)
    results = classifier.predict_proba(args.text)

    for text, (label, confidence) in zip(args.text, results):
        print(f"Text: {text[:50]}...")
        print(f"  Sentiment: {label} (confidence: {confidence:.2%})")
        print()


if __name__ == "__main__":
    main()
```

---

## Part 4: Train/Validation/Test Splits

### Why Three Sets?

| Set | Purpose | When Used |
|-----|---------|-----------|
| Training | Model learns patterns | During training |
| Validation | Tune hyperparameters | During development |
| Test | Final performance estimate | Once, at the end |

### The Information Leakage Problem

```
BAD: Tune hyperparameters on test set
    → Overly optimistic test metrics
    → Model fails on real data

GOOD: Tune on validation, test only once
    → Honest performance estimate
    → Matches production behavior
```

### Stratification

For imbalanced datasets, random splits can create unrepresentative subsets:

```
Original: 80% positive, 20% negative
Random split (unlucky): Train 85% positive, Test 50% positive
Stratified split: Both sets maintain 80%/20% ratio
```

The `train_test_split` parameter `stratify=df["label"]` ensures proportional representation.

### Typical Split Ratios

| Dataset Size | Train | Validation | Test |
|--------------|-------|------------|------|
| < 10,000 | 60% | 20% | 20% |
| 10,000 - 100,000 | 70% | 15% | 15% |
| > 100,000 | 80% | 10% | 10% |

Larger datasets need smaller validation/test fractions because absolute numbers remain sufficient.

---

## Part 5: Running the Training Pipeline

### Sample Data

Create a sample dataset for testing:

```python
# scripts/create_sample_data.py
import pandas as pd
from pathlib import Path

data = {
    "text": [
        "This product is amazing! Best purchase ever.",
        "Terrible quality, broke after one day.",
        "It's okay, nothing special.",
        "Absolutely love it, highly recommend!",
        "Waste of money, very disappointed.",
        "Does what it's supposed to do.",
        "Exceeded my expectations!",
        "Would not buy again.",
        "Pretty good for the price.",
        "Fantastic customer service!",
    ] * 100,  # Repeat for 1000 samples
    "label": [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive",
    ] * 100,
}

df = pd.DataFrame(data)
output_path = Path("data/sample_reviews.csv")
output_path.parent.mkdir(exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Created {len(df)} samples at {output_path}")
```

### Training Command

```bash
# From project root
python -m src.train \
    --data-path data/sample_reviews.csv \
    --output-dir models/v1 \
    --max-features 5000

# Expected output:
# 2024-01-15 10:30:00 - INFO - Loading data from data/sample_reviews.csv
# 2024-01-15 10:30:00 - INFO - Loaded 1000 samples
# 2024-01-15 10:30:00 - INFO - Split sizes - Train: 700, Val: 100, Test: 200
# 2024-01-15 10:30:00 - INFO - Creating TF-IDF features
# 2024-01-15 10:30:00 - INFO - Vocabulary size: 156
# 2024-01-15 10:30:00 - INFO - Training model
# ... metrics output ...
# 2024-01-15 10:30:01 - INFO - Training complete!
```

### Making Predictions

```bash
python -m src.predict \
    --model-dir models/v1 \
    "This product is fantastic!" \
    "I want a refund immediately"

# Output:
# Text: This product is fantastic!...
#   Sentiment: positive (confidence: 94.23%)
#
# Text: I want a refund immediately...
#   Sentiment: negative (confidence: 87.65%)
```

---

## Part 6: Requirements File

```
# requirements.txt
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0
pytest>=7.4.0
```

---

## Exercises

### Exercise 1.3.1: Implement the Data Module

Create `src/data.py` with the code from Part 3. Verify it works:

```python
# Quick test
from src.data import load_data, split_data
from pathlib import Path

df = load_data(Path("data/sample_reviews.csv"))
train, val, test = split_data(df)
print(f"Splits: {len(train)}, {len(val)}, {len(test)}")
```

### Exercise 1.3.2: Implement Features and Training

Create `src/features.py` and `src/train.py`. Run a full training cycle and verify:
- Model file exists at `models/v1/model.joblib`
- Featurizer exists at `models/v1/featurizer.joblib`
- Metadata shows all three metric sets

### Exercise 1.3.3: Add a Feature

Modify `TextFeaturizer` to optionally include character n-grams (useful for typos and misspellings). Add a parameter `char_ngrams: bool = False` and update the vectorizer accordingly.

### Exercise 1.3.4: Error Handling

What happens if you call `predict.py` with an empty string? Add explicit handling for edge cases:
- Empty strings
- Very long texts (> 10,000 characters)
- Non-ASCII characters

---

## Key Takeaways

1. **Notebooks are for exploration, not production** - Start with modular Python scripts
2. **Separate concerns**: data loading, feature engineering, training, and inference belong in different modules
3. **Three-way splits prevent overfitting**: Train on train, tune on validation, evaluate once on test
4. **Stratification maintains class balance** across all splits
5. **Save all artifacts**: model, featurizer, and metadata for reproducibility

---

## Next Steps

Run `/start-1-4` to learn how to package your model for production deployment using Docker.
