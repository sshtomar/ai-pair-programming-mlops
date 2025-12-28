"""Data loading and preprocessing utilities for sentiment classification."""

import re
from pathlib import Path

import pandas as pd


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load sentiment data from a CSV file.

    Args:
        filepath: Path to CSV file with 'text' and 'label' columns.

    Returns:
        DataFrame with text and label columns.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = {'text', 'label'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def preprocess_text(text: str) -> str:
    """Clean and normalize text for sentiment analysis.

    Args:
        text: Raw text string to preprocess.

    Returns:
        Cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

    Args:
        df: DataFrame to split.
        train_size: Fraction for training set.
        val_size: Fraction for validation set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If split sizes don't sum to <= 1.0.
    """
    if train_size + val_size > 1.0:
        raise ValueError(
            f"train_size ({train_size}) + val_size ({val_size}) must be <= 1.0"
        )

    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n = len(df_shuffled)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]

    return train_df, val_df, test_df
