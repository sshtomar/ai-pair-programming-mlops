"""Prediction utilities for sentiment classification."""

from pathlib import Path

import joblib
import numpy as np

from src.data import preprocess_text
from src.features import TextVectorizer


class SentimentClassifier:
    """Sentiment classifier for making predictions.

    Wraps the trained model and vectorizer for easy inference.

    Example:
        >>> classifier = SentimentClassifier.load("models/")
        >>> classifier.predict(["This movie was great!"])
        [1]
        >>> classifier.predict_proba(["This movie was great!"])
        array([[0.2, 0.8]])
    """

    def __init__(
        self,
        model,
        vectorizer: TextVectorizer,
        labels: list[str] | None = None
    ) -> None:
        """Initialize the classifier.

        Args:
            model: Trained sklearn classifier.
            vectorizer: Fitted TextVectorizer.
            labels: Optional label names for classes.
        """
        self._model = model
        self._vectorizer = vectorizer
        self._labels = labels or ['negative', 'positive']

    @classmethod
    def load(cls, model_dir: str | Path) -> "SentimentClassifier":
        """Load a trained classifier from disk.

        Args:
            model_dir: Directory containing model.joblib and vectorizer.joblib.

        Returns:
            Loaded SentimentClassifier instance.

        Raises:
            FileNotFoundError: If model files don't exist.
        """
        model_dir = Path(model_dir)

        model_path = model_dir / "model.joblib"
        vectorizer_path = model_dir / "vectorizer.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        model = joblib.load(model_path)
        vectorizer = TextVectorizer.load(vectorizer_path)

        return cls(model=model, vectorizer=vectorizer)

    def predict(self, texts: str | list[str]) -> list[str]:
        """Predict sentiment labels for texts.

        Args:
            texts: Single text string or list of text strings to classify.

        Returns:
            List of predicted label strings (e.g., 'positive', 'negative').
        """
        if isinstance(texts, str):
            texts = [texts]
        cleaned_texts = [preprocess_text(t) for t in texts]
        X = self._vectorizer.transform(cleaned_texts)
        predictions = self._model.predict(X)
        return predictions.tolist()

    def predict_proba(self, texts: str | list[str]) -> np.ndarray:
        """Predict class probabilities for texts.

        Args:
            texts: Single text string or list of text strings to classify.

        Returns:
            Array of shape (n_samples, n_classes) with probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]
        cleaned_texts = [preprocess_text(t) for t in texts]
        X = self._vectorizer.transform(cleaned_texts)
        return self._model.predict_proba(X)

    def predict_label(self, texts: str | list[str]) -> list[str]:
        """Predict sentiment labels as strings.

        Alias for predict() - kept for backwards compatibility.

        Args:
            texts: Single text string or list of text strings to classify.

        Returns:
            List of label strings (e.g., 'positive', 'negative').
        """
        return self.predict(texts)

    @property
    def labels(self) -> list[str]:
        """Get the class label names."""
        return self._labels
