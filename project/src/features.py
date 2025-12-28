"""Feature extraction utilities for text classification."""

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """TF-IDF vectorizer wrapper for text classification.

    Provides a consistent interface for fitting, transforming, and
    persisting text vectorization models.

    Attributes:
        max_features: Maximum number of vocabulary terms.
        ngram_range: Range of n-grams to extract.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2)
    ) -> None:
        """Initialize the vectorizer.

        Args:
            max_features: Maximum vocabulary size.
            ngram_range: Min and max n-gram lengths.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> "TextVectorizer":
        """Fit the vectorizer on training texts.

        Args:
            texts: List of text documents.

        Returns:
            Self for method chaining.
        """
        self._vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: list[str]) -> spmatrix:
        """Transform texts to TF-IDF features.

        Args:
            texts: List of text documents.

        Returns:
            Sparse matrix of TF-IDF features.

        Raises:
            RuntimeError: If vectorizer hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")
        return self._vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]) -> spmatrix:
        """Fit and transform in one step.

        Args:
            texts: List of text documents.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, filepath: str | Path) -> None:
        """Save the fitted vectorizer to disk.

        Args:
            filepath: Path to save the vectorizer.

        Raises:
            RuntimeError: If vectorizer hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted vectorizer")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'vectorizer': self._vectorizer,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "TextVectorizer":
        """Load a fitted vectorizer from disk.

        Args:
            filepath: Path to the saved vectorizer.

        Returns:
            Loaded TextVectorizer instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")

        data = joblib.load(filepath)

        instance = cls(
            max_features=data['max_features'],
            ngram_range=data['ngram_range']
        )
        instance._vectorizer = data['vectorizer']
        instance._is_fitted = True

        return instance

    @property
    def vocabulary_size(self) -> int:
        """Get the actual vocabulary size after fitting.

        Returns:
            Number of terms in vocabulary.

        Raises:
            RuntimeError: If vectorizer hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted first")
        return len(self._vectorizer.vocabulary_)
