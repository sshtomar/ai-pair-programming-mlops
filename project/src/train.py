"""Training script for sentiment classifier."""

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import load_data, preprocess_text, split_data
from src.features import TextVectorizer


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def train_model(
    data_path: str | Path,
    model_dir: str | Path,
    max_features: int = 5000,
    C: float = 1.0,
    random_state: int = 42
) -> dict:
    """Train a sentiment classification model.

    Args:
        data_path: Path to training data CSV.
        model_dir: Directory to save model artifacts.
        max_features: Maximum TF-IDF vocabulary size.
        C: Logistic regression regularization strength.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with training metrics.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    df['text_clean'] = df['text'].apply(preprocess_text)

    # Split data
    print("Splitting data...")
    train_df, val_df, test_df = split_data(df, random_state=random_state)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Vectorize text
    print(f"Vectorizing text (max_features={max_features})...")
    vectorizer = TextVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df['text_clean'].tolist())
    X_val = vectorizer.transform(val_df['text_clean'].tolist())
    X_test = vectorizer.transform(test_df['text_clean'].tolist())

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Train model
    print(f"Training logistic regression (C={C})...")
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Evaluate
    def evaluate(X, y, name: str) -> dict:
        y_pred = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        print(f"\n{name} metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        return metrics

    train_metrics = evaluate(X_train, y_train, "Train")
    val_metrics = evaluate(X_val, y_val, "Validation")
    test_metrics = evaluate(X_test, y_test, "Test")

    # Save artifacts
    print(f"\nSaving model to {model_dir}...")
    vectorizer.save(model_dir / "vectorizer.joblib")
    joblib.dump(model, model_dir / "model.joblib")

    # Save metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'params': {
            'max_features': max_features,
            'C': C,
            'vocabulary_size': vectorizer.vocabulary_size
        }
    }

    with open(model_dir / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("Training complete!")
    return all_metrics


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train a sentiment classification model"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save model artifacts (default: models)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum TF-IDF vocabulary size (default: 5000)"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Logistic regression regularization strength (default: 1.0)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML (overrides other arguments)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        max_features = config.get('max_features', args.max_features)
        C = config.get('C', args.C)
        random_state = config.get('random_state', args.random_state)
    else:
        max_features = args.max_features
        C = args.C
        random_state = args.random_state

    train_model(
        data_path=args.data,
        model_dir=args.model_dir,
        max_features=max_features,
        C=C,
        random_state=random_state
    )


if __name__ == "__main__":
    main()
