"""
Exercise 3.1: Model Serving Options
Difficulty: ★★☆
Topic: Batch vs Real-time Inference Patterns

This exercise builds your understanding of when to use batch processing
versus real-time serving, and how to implement each pattern effectively.

Instructions:
1. WRITE & VERIFY: Implement BatchPredictor class (Part A)
2. FIX THIS ISSUE: Optimize a slow real-time predictor (Part B)
3. DESIGN DECISION: Recommend serving strategy based on requirements (Part C)

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# =============================================================================
# PART A: Write & Verify - BatchPredictor Class
# =============================================================================
# Implement a BatchPredictor that:
# - Takes a path to a trained model (pickle file)
# - Has a predict_batch(input_csv_path, output_csv_path) method
# - Reads text data from input CSV (column: "text")
# - Writes predictions to output CSV (columns: "text", "prediction", "confidence")
# - Handles errors gracefully (missing files, invalid data)
# - Returns a summary dict with: total_processed, successful, failed, elapsed_time
#
# Expected CSV input format:
#   text
#   "This product is amazing!"
#   "Terrible quality, very disappointed"
#
# Expected CSV output format:
#   text,prediction,confidence
#   "This product is amazing!",positive,0.92
#   "Terrible quality, very disappointed",negative,0.87
# =============================================================================


class BatchPredictor:
    """Batch prediction processor for sentiment classification."""

    def __init__(self, model_path: str | Path):
        """
        Initialize the batch predictor with a trained model.

        Args:
            model_path: Path to the pickled model file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        # TODO: Load the model from the pickle file
        # TODO: Store the model path and validate it exists
        pass

    def predict_batch(
        self, input_csv_path: str | Path, output_csv_path: str | Path
    ) -> dict:
        """
        Process a CSV file of texts and write predictions to output CSV.

        Args:
            input_csv_path: Path to input CSV with 'text' column
            output_csv_path: Path to write predictions

        Returns:
            dict with keys: total_processed, successful, failed, elapsed_time

        Raises:
            FileNotFoundError: If input CSV doesn't exist
            ValueError: If input CSV doesn't have 'text' column
        """
        # TODO: Implement batch prediction logic
        # 1. Read input CSV
        # 2. Validate 'text' column exists
        # 3. Run predictions (handle individual failures gracefully)
        # 4. Write output CSV
        # 5. Return summary statistics
        pass


# =============================================================================
# PART B: Fix This Issue - Slow Real-time Predictor
# =============================================================================
# This predictor is way too slow for production use.
# Problems:
# - Model is loaded on every request (expensive!)
# - No caching for repeated predictions
# - No batching for multiple simultaneous requests
#
# Your task: Identify and fix the performance issues
# Target: <10ms per prediction after warmup (currently ~500ms)
# =============================================================================


class SlowRealTimePredictor:
    """A real-time predictor with performance problems. Fix it!"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        # BUG: Model is loaded fresh on every predict() call
        # This is extremely slow!

    def predict(self, text: str) -> dict:
        """Make a single prediction. Currently way too slow!"""
        import pickle

        # BUG 1: Loading model on every call (~400ms overhead)
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        # BUG 2: No caching - same text gets re-computed every time
        prediction = model.predict([text])[0]
        confidence = max(model.predict_proba([text])[0])

        return {"prediction": prediction, "confidence": float(confidence)}

    def predict_many(self, texts: list[str]) -> list[dict]:
        """Predict multiple texts. Currently processes one at a time!"""
        # BUG 3: Not using batch prediction - O(n) model loads!
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


# TODO: Create an optimized version called FastRealTimePredictor
# It should:
# 1. Load model once at initialization
# 2. Cache recent predictions (use functools.lru_cache or similar)
# 3. Use batch prediction for predict_many()


class FastRealTimePredictor:
    """Your optimized real-time predictor. Implement this!"""

    def __init__(self, model_path: str, cache_size: int = 1000):
        """
        Initialize with model loaded once and prediction cache.

        Args:
            model_path: Path to the pickled model
            cache_size: Maximum number of predictions to cache
        """
        # TODO: Load model once here
        # TODO: Set up caching mechanism
        pass

    def predict(self, text: str) -> dict:
        """Make a cached single prediction."""
        # TODO: Check cache first, then predict if needed
        pass

    def predict_many(self, texts: list[str]) -> list[dict]:
        """Efficiently predict multiple texts using batching."""
        # TODO: Use model's batch prediction capability
        pass

    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        # TODO: Return cache performance metrics
        pass


# =============================================================================
# PART C: Design Decision - Serving Strategy Recommendation
# =============================================================================
# Given a set of requirements, recommend batch vs real-time serving.
# This function should analyze the requirements and provide a reasoned
# recommendation with justification.
# =============================================================================


@dataclass
class ServingRequirements:
    """Requirements for a model serving use case."""

    predictions_per_day: int  # Expected daily volume
    max_latency_ms: int  # Maximum acceptable latency
    freshness_hours: int  # How fresh predictions need to be (0 = real-time)
    request_pattern: Literal["steady", "bursty", "scheduled"]
    cost_sensitivity: Literal["low", "medium", "high"]
    integration_type: Literal["api", "database", "file"]


def recommend_serving_strategy(requirements: ServingRequirements) -> dict:
    """
    Analyze requirements and recommend a serving strategy.

    Args:
        requirements: ServingRequirements dataclass with use case details

    Returns:
        dict with keys:
            - recommendation: "batch" or "real-time" or "hybrid"
            - confidence: float 0-1 indicating certainty
            - reasoning: list of strings explaining the decision
            - considerations: list of things to watch out for
            - estimated_cost_tier: "low", "medium", "high"

    Example:
        >>> reqs = ServingRequirements(
        ...     predictions_per_day=1000000,
        ...     max_latency_ms=50,
        ...     freshness_hours=0,
        ...     request_pattern="bursty",
        ...     cost_sensitivity="medium",
        ...     integration_type="api"
        ... )
        >>> result = recommend_serving_strategy(reqs)
        >>> result["recommendation"]
        'real-time'
        >>> len(result["reasoning"]) > 0
        True
    """
    # TODO: Implement the decision logic
    # Consider these factors:
    # - Latency requirements (< 100ms usually means real-time)
    # - Freshness (0 hours = must be real-time)
    # - Volume (very high volume might favor batch for cost)
    # - Request pattern (scheduled = batch, bursty = real-time with scaling)
    # - Cost (batch is usually cheaper for same volume)
    # - Integration type (api = real-time, file/database = often batch)
    #
    # Return a hybrid recommendation when requirements conflict
    pass


# =============================================================================
# HINTS (revealed progressively with /hint command)
# =============================================================================
"""
HINT 1 - BatchPredictor:
- Use pandas for CSV reading/writing
- Wrap individual predictions in try/except to handle failures
- Track timing with time.perf_counter() for accurate measurement
- Example structure:
    start = time.perf_counter()
    df = pd.read_csv(input_csv_path)
    if 'text' not in df.columns:
        raise ValueError("Missing 'text' column")
    # ... process ...
    elapsed = time.perf_counter() - start

HINT 2 - FastRealTimePredictor:
- Load model in __init__:
    with open(model_path, 'rb') as f:
        self.model = pickle.load(f)
- Use functools.lru_cache for caching, but note it only works on hashable args
- For predict_many, use self.model.predict(texts) directly (batch mode)
- Track cache stats with a simple counter dict: {'hits': 0, 'misses': 0}

HINT 3 - recommend_serving_strategy:
- Start with freshness: if freshness_hours == 0, real-time is required
- Then check latency: max_latency_ms < 100 strongly suggests real-time
- For hybrid: high volume + real-time requirements = hybrid (cache + batch backfill)
- Scoring approach:
    realtime_score = 0
    if requirements.freshness_hours == 0: realtime_score += 3
    if requirements.max_latency_ms < 100: realtime_score += 2
    if requirements.integration_type == "api": realtime_score += 1
    # ... then threshold the score
"""

# =============================================================================
# Test your implementation (run with: python exercise_3_1.py)
# =============================================================================
if __name__ == "__main__":
    print("Exercise 3.1: Model Serving Options")
    print("=" * 50)

    # Quick self-test
    print("\nPart C - Testing recommend_serving_strategy:")

    # Test case 1: Clearly real-time
    reqs_realtime = ServingRequirements(
        predictions_per_day=10000,
        max_latency_ms=50,
        freshness_hours=0,
        request_pattern="steady",
        cost_sensitivity="low",
        integration_type="api",
    )

    # Test case 2: Clearly batch
    reqs_batch = ServingRequirements(
        predictions_per_day=1000000,
        max_latency_ms=3600000,  # 1 hour is fine
        freshness_hours=24,
        request_pattern="scheduled",
        cost_sensitivity="high",
        integration_type="file",
    )

    result_rt = recommend_serving_strategy(reqs_realtime)
    result_batch = recommend_serving_strategy(reqs_batch)

    if result_rt and result_batch:
        print(f"Real-time scenario recommendation: {result_rt.get('recommendation')}")
        print(f"Batch scenario recommendation: {result_batch.get('recommendation')}")

        if result_rt.get("recommendation") == "real-time":
            print("  [PASS] Correctly identified real-time use case")
        else:
            print("  [FAIL] Should recommend real-time for low-latency API")

        if result_batch.get("recommendation") == "batch":
            print("  [PASS] Correctly identified batch use case")
        else:
            print("  [FAIL] Should recommend batch for scheduled file processing")
    else:
        print("  [TODO] Implement recommend_serving_strategy to test")

    print("\n" + "=" * 50)
    print("Run pytest test_level_3.py for full test coverage")
