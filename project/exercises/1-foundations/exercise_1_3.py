"""
Exercise 1.3: First Model
Difficulty: ★★☆
Topic: Text preprocessing and data leakage prevention

Instructions:
This exercise has two parts:

PART A - Guided Stub (preprocess_text):
Text preprocessing is the foundation of any NLP pipeline. Implement a function
that cleans and normalizes text for sentiment analysis.

PART B - Debug Exercise (train_model_broken):
The training script has a critical data leakage bug that would cause the model
to perform unrealistically well in testing but fail in production.

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import re
import string


# =============================================================================
# PART A: Guided Stub - Text Preprocessing
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Preprocess text for sentiment analysis.

    Steps to implement:
    1. Convert to lowercase
    2. Strip leading/trailing whitespace
    3. Remove punctuation (replace with space to preserve word boundaries)
    4. Collapse multiple spaces into single space
    5. Strip again to remove any leading/trailing spaces from step 4

    Args:
        text: Raw input text

    Returns:
        Cleaned and normalized text

    Raises:
        TypeError: If text is not a string

    Examples:
        >>> preprocess_text("Hello, World!")
        'hello world'
        >>> preprocess_text("  Multiple   spaces  ")
        'multiple spaces'
        >>> preprocess_text("I love this product!!!")
        'i love this product'
        >>> preprocess_text("Don't stop believing")
        'don t stop believing'
    """
    # Step 1 - Input validation
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Step 2 - Convert to lowercase
    text = text.lower()

    # Step 3 - Strip leading/trailing whitespace
    text = text.strip()

    # Step 4 - Remove punctuation (replace with space)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Step 5 - Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)

    # Step 6 - Final strip and return
    return text.strip()


def preprocess_batch(texts: list[str]) -> list[str]:
    """
    Preprocess a batch of texts.

    Args:
        texts: List of raw text strings

    Returns:
        List of preprocessed texts

    Raises:
        TypeError: If texts is not a list
        TypeError: If any element is not a string
    """
    # Validate input is a list
    if not isinstance(texts, list):
        raise TypeError("Input must be a list")

    # Apply preprocess_text to each element
    return [preprocess_text(t) for t in texts]


# =============================================================================
# PART B: Debug Exercise - Data Leakage in Training
# =============================================================================

def train_model_broken(texts: list[str], labels: list[int]) -> dict:
    """
    BUG: This training function has data leakage that will cause:
    - Artificially high test accuracy (95%+ when real would be 70%)
    - Poor production performance
    - Overconfident model selection decisions

    Find the data leakage bug and understand why it's problematic.

    The bug is subtle - this code pattern appears in many tutorials!
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # BUG: Fitting the vectorizer on ALL data before splitting
    # This means the vectorizer learns vocabulary and IDF weights
    # from the test set, which won't be available in production!
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)  # <-- LEAKAGE HERE

    # Split AFTER transformation
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate (scores will be artificially high)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    return {
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "vectorizer": vectorizer,
        "model": model,
    }


def train_model_fixed(texts: list[str], labels: list[int]) -> dict:
    """
    Fixed version of the training function without data leakage.

    The key insight: any transformation that learns from data (fit)
    must only learn from the training set.

    Args:
        texts: List of text samples
        labels: List of binary labels (0 or 1)

    Returns:
        Dictionary with:
        - "train_accuracy": float
        - "test_accuracy": float
        - "vectorizer": fitted TfidfVectorizer
        - "model": trained LogisticRegression

    Raises:
        ValueError: If texts and labels have different lengths
        ValueError: If inputs are empty
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Step 1 - Input validation
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if len(texts) == 0:
        raise ValueError("inputs cannot be empty")

    # Step 2 - Split FIRST, before any transformation
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Step 3 - Fit vectorizer on training data ONLY
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(texts_train)  # fit on train only!
    X_test = vectorizer.transform(texts_test)        # transform only, no fit!

    # Step 4 - Train and evaluate
    model = LogisticRegression()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Step 5 - Return results dict
    return {
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "vectorizer": vectorizer,
        "model": model,
    }


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
/hint 1 - preprocess_text:
For removing punctuation, string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
You can use translate() with maketrans():
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

/hint 2 - preprocess_text:
Or use regex for simpler code:
    text = re.sub(r'[^\w\s]', ' ', text)  # Keep only word chars and whitespace
    text = re.sub(r'\s+', ' ', text)       # Collapse multiple spaces

/hint 3 - preprocess_text:
Complete solution:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    text = text.lower()
    text = text.strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

/hint 1 - train_model_fixed:
The correct order is:
1. Split data into train/test
2. Fit transformers on train only
3. Transform both train and test
4. Train model on transformed train
5. Evaluate on transformed test

/hint 2 - train_model_fixed:
Key code change - split first:
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(texts_train)  # fit on train only
    X_test = vectorizer.transform(texts_test)        # transform only

/hint 3 - train_model_fixed:
Complete solution:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have same length")
    if len(texts) == 0:
        raise ValueError("inputs cannot be empty")

    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return {
        "train_accuracy": model.score(X_train, y_train),
        "test_accuracy": model.score(X_test, y_test),
        "vectorizer": vectorizer,
        "model": model,
    }
"""
