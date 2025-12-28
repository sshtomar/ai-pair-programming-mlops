"""
Exercise 3.2: FastAPI Basics
Difficulty: ★★★
Topic: Building Production-Ready ML APIs

This exercise teaches you to build robust FastAPI endpoints for ML models.
You'll implement proper request/response models, validation, error handling,
and health checks.

Instructions:
1. WRITE & VERIFY: Create /predict endpoint with Pydantic models (Part A)
2. FIX THIS ISSUE: API that crashes on malformed input (Part B)
3. FIX THIS ISSUE: API missing health check endpoint (Part C)

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# PART A: Write & Verify - Prediction Endpoint with Proper Models
# =============================================================================
# Create Pydantic models for a sentiment prediction API:
# - PredictionRequest: validates input text
# - PredictionResponse: structures output with prediction details
# - Create the FastAPI endpoint function (we'll test it without running server)
#
# Requirements:
# - Input text must be non-empty and <= 5000 characters
# - Response includes: prediction, confidence, model_version, latency_ms
# - Handle both single text and batch predictions
# =============================================================================


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# TODO: Implement PredictionRequest model
class PredictionRequest(BaseModel):
    """
    Request model for sentiment prediction.

    Validates:
    - text: non-empty string, max 5000 characters
    - include_confidence: optional boolean (default True)
    """

    # TODO: Add field for 'text' with validation
    # - Must be at least 1 character
    # - Must be at most 5000 characters
    # - Add helpful description

    # TODO: Add field for 'include_confidence' with default True
    pass


# TODO: Implement BatchPredictionRequest model
class BatchPredictionRequest(BaseModel):
    """
    Request model for batch sentiment predictions.

    Validates:
    - texts: list of 1-100 non-empty strings
    """

    # TODO: Add field for 'texts' list
    # - Must have at least 1 text
    # - Must have at most 100 texts
    # - Each text follows same rules as PredictionRequest
    pass


# TODO: Implement PredictionResponse model
class PredictionResponse(BaseModel):
    """
    Response model for a single prediction.

    Contains prediction results with metadata.
    """

    # TODO: Add these fields:
    # - text: str (the input text, echoed back)
    # - prediction: SentimentLabel
    # - confidence: float (0.0 to 1.0), optional based on request
    # - model_version: str
    # - latency_ms: float
    pass


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: list[PredictionResponse]
    total_latency_ms: float
    avg_latency_ms: float


# TODO: Implement the prediction endpoint function
async def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """
    Predict sentiment for input text.

    This function will be mounted as a FastAPI endpoint:
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            return await predict_sentiment(request)

    Args:
        request: Validated PredictionRequest

    Returns:
        PredictionResponse with prediction and metadata
    """
    # TODO: Implement prediction logic
    # 1. Record start time
    # 2. Run model prediction (mock it for now with simple logic)
    # 3. Calculate latency
    # 4. Return PredictionResponse

    # For testing, use this mock prediction logic:
    # - If text contains "good", "great", "love", "excellent" -> positive
    # - If text contains "bad", "terrible", "hate", "awful" -> negative
    # - Otherwise -> neutral
    # - Confidence: 0.85 for keyword match, 0.60 for neutral
    pass


# =============================================================================
# PART B: Fix This Issue - API That Crashes on Malformed Input
# =============================================================================
# This API has poor error handling and crashes in production.
# Problems:
# - No input validation
# - Unhandled exceptions expose internal errors
# - No proper error response format
#
# Your task: Fix the buggy endpoint to handle errors gracefully
# =============================================================================


class BuggyPredictionRequest(BaseModel):
    """Minimal request model - not enough validation!"""

    text: Any  # BUG: Should be str with validation!
    options: dict | None = None  # BUG: Too permissive


class BuggyAPI:
    """A buggy API class. Find and fix the issues!"""

    def __init__(self, model: Any = None):
        self.model = model
        self.request_count = 0

    # BUG: This endpoint crashes on many inputs
    async def predict(self, request: BuggyPredictionRequest) -> dict:
        """
        Buggy predict endpoint. Crashes on:
        - Empty text
        - None text
        - Integer text
        - Very long text
        - Missing model
        """
        self.request_count += 1

        # BUG 1: No null check - crashes if text is None
        text_length = len(request.text)

        # BUG 2: No type check - crashes if text is not a string
        cleaned_text = request.text.lower().strip()

        # BUG 3: No model check - crashes if model is None
        prediction = self.model.predict([cleaned_text])[0]

        # BUG 4: No bounds check on confidence
        confidence = self.model.predict_proba([cleaned_text])[0][1]

        # BUG 5: Options dict accessed without validation
        max_length = request.options["max_length"]  # Crashes if options is None!

        return {
            "prediction": prediction,
            "confidence": confidence,
            "text_length": text_length,
        }


# TODO: Create FixedAPI with proper error handling
class APIError(BaseModel):
    """Standard error response format."""

    error: str
    error_code: str
    details: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FixedPredictionRequest(BaseModel):
    """Properly validated request model."""

    # TODO: Add proper validation for text field
    # - Must be string
    # - Must be non-empty after stripping
    # - Must be <= 5000 characters
    # - Add Field with description

    # TODO: Add proper options field
    # - Should have specific expected fields, not open dict
    pass


class FixedAPI:
    """Your fixed API class. Implement proper error handling!"""

    def __init__(self, model: Any = None):
        self.model = model
        self.request_count = 0

    async def predict(self, request: FixedPredictionRequest) -> dict | APIError:
        """
        Fixed predict endpoint with proper error handling.

        Should handle:
        - Missing model gracefully
        - Invalid input with helpful error messages
        - Never expose internal exception details
        """
        # TODO: Implement with proper error handling
        # 1. Validate model exists
        # 2. Safely access request fields
        # 3. Wrap prediction in try/except
        # 4. Return APIError for failures, not exceptions
        pass


# =============================================================================
# PART C: Fix This Issue - Missing Health Check
# =============================================================================
# This API has no health check endpoint, making it impossible to:
# - Know if the API is ready to serve
# - Configure load balancer health probes
# - Debug startup issues
#
# Implement comprehensive health checks
# =============================================================================


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None


class HealthCheckResponse(BaseModel):
    """Complete health check response."""

    status: HealthStatus
    version: str
    uptime_seconds: float
    components: list[ComponentHealth]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIWithoutHealthCheck:
    """API missing health checks. Students add them!"""

    def __init__(self, model: Any = None, model_version: str = "1.0.0"):
        self.model = model
        self.model_version = model_version
        self.start_time = datetime.utcnow()
        # No health check endpoint!


# TODO: Implement health check system
class HealthyAPI:
    """API with proper health checks. Implement this!"""

    def __init__(self, model: Any = None, model_version: str = "1.0.0"):
        self.model = model
        self.model_version = model_version
        self.start_time = datetime.utcnow()

    async def health_check(self) -> HealthCheckResponse:
        """
        Comprehensive health check endpoint.

        Checks:
        1. Model is loaded and can make predictions
        2. Memory usage is acceptable
        3. Response time is acceptable

        Returns:
            HealthCheckResponse with overall and component status
        """
        # TODO: Implement health check logic
        # 1. Check model is not None
        # 2. Try a test prediction
        # 3. Calculate uptime
        # 4. Aggregate component health into overall status
        pass

    async def liveness_probe(self) -> dict:
        """
        Simple liveness probe for Kubernetes.

        Returns: {"status": "alive"} if process is running
        """
        # TODO: Implement simple liveness check
        pass

    async def readiness_probe(self) -> dict:
        """
        Readiness probe - is API ready to serve traffic?

        Returns: {"status": "ready"} only if model is loaded
        """
        # TODO: Implement readiness check
        pass


# =============================================================================
# HINTS (revealed progressively with /hint command)
# =============================================================================
"""
HINT 1 - Pydantic Models:
- Use Field() for validation:
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
- For lists with bounds:
    texts: list[str] = Field(..., min_items=1, max_items=100)
- Optional with default:
    include_confidence: bool = Field(default=True)

HINT 2 - Error Handling:
- Use HTTPException for API errors:
    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail="Invalid input")
- Better pattern - catch and convert:
    try:
        result = self.model.predict(...)
    except AttributeError:
        return APIError(error="Model not loaded", error_code="MODEL_NOT_READY")
    except Exception as e:
        # Log the real error internally, return generic message
        logger.error(f"Prediction failed: {e}")
        return APIError(error="Prediction failed", error_code="PREDICTION_ERROR")

HINT 3 - Health Checks:
- Test prediction for model health:
    try:
        start = time.perf_counter()
        self.model.predict(["test"])
        latency = (time.perf_counter() - start) * 1000
        model_health = ComponentHealth(name="model", status=HealthStatus.HEALTHY, latency_ms=latency)
    except Exception as e:
        model_health = ComponentHealth(name="model", status=HealthStatus.UNHEALTHY, message=str(e))

- Aggregate status:
    if any(c.status == HealthStatus.UNHEALTHY for c in components):
        overall = HealthStatus.UNHEALTHY
    elif any(c.status == HealthStatus.DEGRADED for c in components):
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.HEALTHY
"""

# =============================================================================
# Test your implementation (run with: python exercise_3_2.py)
# =============================================================================
if __name__ == "__main__":
    import asyncio

    print("Exercise 3.2: FastAPI Basics")
    print("=" * 50)

    async def test_implementations():
        # Test Part A - Pydantic models
        print("\nPart A - Testing Pydantic Models:")
        try:
            # This should work
            req = PredictionRequest(text="This is a great product!")
            print(f"  [PASS] Created valid request: {req}")
        except Exception as e:
            print(f"  [TODO] PredictionRequest not implemented: {e}")

        try:
            # This should fail validation
            req = PredictionRequest(text="")
            print("  [FAIL] Should reject empty text")
        except ValueError:
            print("  [PASS] Correctly rejects empty text")
        except Exception as e:
            print(f"  [TODO] PredictionRequest not implemented: {e}")

        # Test predict_sentiment
        try:
            req = PredictionRequest(text="This is a great product!")
            response = await predict_sentiment(req)
            if response and response.prediction == SentimentLabel.POSITIVE:
                print("  [PASS] Correctly predicts positive sentiment")
            else:
                print("  [FAIL] Should predict positive for 'great'")
        except Exception as e:
            print(f"  [TODO] predict_sentiment not implemented: {e}")

        # Test Part C - Health checks
        print("\nPart C - Testing Health Checks:")
        try:
            api = HealthyAPI(model=None)  # No model loaded
            health = await api.health_check()
            if health and health.status == HealthStatus.UNHEALTHY:
                print("  [PASS] Reports unhealthy when model missing")
            else:
                print("  [FAIL] Should be unhealthy without model")
        except Exception as e:
            print(f"  [TODO] HealthyAPI not implemented: {e}")

    asyncio.run(test_implementations())

    print("\n" + "=" * 50)
    print("Run pytest test_level_3.py for full test coverage")
