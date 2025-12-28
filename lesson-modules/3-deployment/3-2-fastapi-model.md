# Lesson 3.2: Building an API with FastAPI

**Duration:** 50 minutes
**Prerequisites:** Lesson 3.1 (Serving Patterns)

## Learning Objectives

By the end of this lesson, you will:
1. Wrap your sentiment model in a FastAPI endpoint
2. Handle request validation and errors properly
3. Implement health checks and metadata endpoints

---

## Why FastAPI?

Before we write code, let's understand why FastAPI dominates ML serving:

| Feature | Benefit for ML |
|---------|----------------|
| **Async native** | Handle concurrent requests while model inference runs |
| **Pydantic validation** | Catch bad inputs before they hit your model |
| **Automatic OpenAPI docs** | Clients can explore your API at `/docs` |
| **Type hints everywhere** | Catch bugs at development time |
| **Performance** | One of the fastest Python frameworks |

Flask served us well for years, but FastAPI's validation and async support make it the modern choice for ML APIs.

### Request/Response Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FastAPI ML API Request Flow                          │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐         ┌───────────────────────────────────────────────────┐
  │  Client  │         │                   FastAPI Server                  │
  │  (curl,  │         │                                                   │
  │  Python, │         │  ┌─────────┐   ┌──────────┐   ┌───────────────┐  │
  │  browser)│         │  │ Request │   │  Route   │   │    Model      │  │
  └────┬─────┘         │  │ Parsing │   │ Handler  │   │   Inference   │  │
       │               │  └────┬────┘   └────┬─────┘   └───────┬───────┘  │
       │ POST /predict │       │             │                 │          │
       │ {"text":...}  │       │             │                 │          │
       ├──────────────────────►│             │                 │          │
       │               │       │ Validate    │                 │          │
       │               │       ├────────────►│                 │          │
       │               │       │   Pydantic  │ predict(text)   │          │
       │               │       │             ├────────────────►│          │
       │               │       │             │                 │ model    │
       │               │       │             │                 │.predict()│
       │               │       │             │    sentiment    │          │
       │               │       │             │◄────────────────┤          │
       │               │       │  Build      │                 │          │
       │               │       │◄────────────┤                 │          │
       │ 200 OK        │       │  response   │                 │          │
       │◄──────────────────────┤             │                 │          │
       │ {"sentiment": │       │             │                 │          │
       │  "positive",  │  └────┴────┘   └────┴─────┘   └───────┴───────┘  │
       │  "confidence":│                                                   │
       │  0.92}        │                                                   │
       │               └───────────────────────────────────────────────────┘
  ┌────┴─────┐
  │  Client  │
  └──────────┘

  Timeline: ─────────────────────────────────────────────────────────────────►
            Request    Validate    Route      Inference    Serialize    Response
            ~1ms       ~1ms        ~1ms       ~20-50ms     ~1ms         ~1ms
```

---

## Project Structure

We'll organize our API code cleanly:

```
project/
├── src/
│   ├── api.py           # FastAPI application
│   ├── schemas.py       # Pydantic request/response models
│   ├── model.py         # Your existing model code
│   └── config.py        # Configuration
├── models/
│   └── sentiment_model.pkl
├── requirements.txt
└── tests/
    └── test_api.py
```

This separation matters:
- **schemas.py**: Define your API contract independently of implementation
- **api.py**: HTTP handling, routing, lifecycle management
- **model.py**: Pure ML logic, no HTTP concepts

---

## Step 1: Define Request/Response Schemas

Create `src/schemas.py`:

```python
"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PredictionRequest(BaseModel):
    """Single text prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to classify for sentiment"
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Ensure text has actual content after stripping."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "This product exceeded my expectations!"}
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prediction result with confidence score."""

    text: str = Field(..., description="Original input text")
    sentiment: Literal["positive", "negative"] = Field(
        ..., description="Predicted sentiment"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence (0-1)"
    )
    model_version: str = Field(..., description="Model version used")


class BatchPredictionRequest(BaseModel):
    """Multiple texts for batch prediction."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to classify (max 100)"
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        """Validate each text in the batch."""
        validated = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
            if len(text) > 5000:
                raise ValueError(f"Text at index {i} exceeds 5000 characters")
            validated.append(text.strip())
        return validated


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse]
    total_processed: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    model_version: str
    training_date: str | None
    features: list[str]
    classes: list[str]
    description: str


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None
```

Key design decisions:
- **Field constraints**: `min_length`, `max_length` prevent abuse
- **Custom validators**: Business logic validation beyond type checking
- **Literal types**: Constrain outputs to known values
- **Examples**: Show up in auto-generated docs

---

## Step 2: Build the FastAPI Application

Create `src/api.py`:

```python
"""FastAPI application for sentiment prediction."""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
MODEL = None
MODEL_VERSION = "1.0.0"
MODEL_PATH = Path("models/sentiment_model.pkl")


def load_model() -> object:
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    return model


def predict_single(text: str) -> tuple[str, float]:
    """Run prediction on a single text.

    Returns:
        Tuple of (sentiment, confidence)
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    # Get prediction and probability
    prediction = MODEL.predict([text])[0]
    probabilities = MODEL.predict_proba([text])[0]
    confidence = float(max(probabilities))

    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment, confidence


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: load model on startup."""
    global MODEL

    logger.info("Starting application...")
    try:
        MODEL = load_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        # Allow startup for health checks to report unhealthy status
        MODEL = None

    yield  # Application runs here

    # Cleanup on shutdown
    logger.info("Shutting down application...")
    MODEL = None


# Create FastAPI app with metadata
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production ML API for text sentiment classification",
    version=MODEL_VERSION,
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "request_id": request_id,
        },
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Handle runtime errors (e.g., model not loaded)."""
    logger.error(f"Runtime error: {exc}")
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "detail": str(exc),
            "request_id": request_id,
        },
    )


# ============================================================
# Middleware
# ============================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID for tracing."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests."""
    start_time = datetime.now()

    response = await call_next(request)

    duration = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.1f}ms"
    )
    return response


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check() -> HealthResponse:
    """Check API and model health.

    Returns healthy status only if model is loaded and functional.
    Use this endpoint for:
    - Kubernetes liveness/readiness probes
    - Load balancer health checks
    - Monitoring systems
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        version=MODEL_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Get model metadata.

    Returns information about the currently loaded model,
    useful for debugging and model versioning.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extract model info (adjust based on your model type)
    return ModelInfoResponse(
        model_name="SentimentClassifier",
        model_version=MODEL_VERSION,
        training_date=None,  # Could load from metadata file
        features=["text"],
        classes=["negative", "positive"],
        description="Binary sentiment classifier trained on product reviews",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict sentiment for a single text.

    Takes a text input and returns the predicted sentiment
    (positive/negative) with a confidence score.

    Example:
    ```
    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"text": "This product is amazing!"}'
    ```
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sentiment, confidence = predict_single(request.text)

    return PredictionResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        model_version=MODEL_VERSION,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"]
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict sentiment for multiple texts.

    Process up to 100 texts in a single request.
    More efficient than multiple single requests for bulk processing.

    Example:
    ```
    curl -X POST http://localhost:8000/predict/batch \\
        -H "Content-Type: application/json" \\
        -d '{"texts": ["Great product!", "Terrible experience."]}'
    ```
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    for text in request.texts:
        sentiment, confidence = predict_single(text)
        predictions.append(
            PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                model_version=MODEL_VERSION,
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
    )


@app.get("/", tags=["Operations"])
async def root() -> dict[str, str]:
    """API root - provides basic info and documentation link."""
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
    }
```

Key implementation details:

**Lifespan Events (modern approach):**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once
    MODEL = load_model()
    yield
    # Shutdown: cleanup
```

This replaces the deprecated `@app.on_event("startup")` pattern.

**Why load at startup, not per-request?**
- Model loading is slow (disk I/O, deserialization)
- Memory efficiency (one copy shared across requests)
- Predictable latency (no cold starts on requests)

---

## Step 3: Running the API

Install dependencies:

```bash
pip install fastapi uvicorn joblib
```

Run with uvicorn:

```bash
# Development (auto-reload)
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

Understanding uvicorn options:
- `--reload`: Restart on code changes (development only)
- `--workers 4`: Spawn 4 worker processes (production)
- `--host 0.0.0.0`: Accept external connections

---

## Step 4: Testing Your API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Response:
# {"status":"healthy","model_loaded":true,"version":"1.0.0"}

# Single prediction
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This product exceeded all my expectations!"}'

# Response:
# {
#   "text": "This product exceeded all my expectations!",
#   "sentiment": "positive",
#   "confidence": 0.92,
#   "model_version": "1.0.0"
# }

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Great product!", "Waste of money.", "Works as expected."]}'

# Response:
# {
#   "predictions": [
#     {"text": "Great product!", "sentiment": "positive", "confidence": 0.95, ...},
#     {"text": "Waste of money.", "sentiment": "negative", "confidence": 0.89, ...},
#     {"text": "Works as expected.", "sentiment": "positive", "confidence": 0.67, ...}
#   ],
#   "total_processed": 3
# }

# Model info
curl http://localhost:8000/model/info
```

### Using httpx (Python client)

```python
import httpx

# Sync client
with httpx.Client(base_url="http://localhost:8000") as client:
    # Health check
    response = client.get("/health")
    print(response.json())

    # Prediction
    response = client.post(
        "/predict",
        json={"text": "Amazing quality!"}
    )
    result = response.json()
    print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2%})")

# Async client
async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
    response = await client.post("/predict", json={"text": "Great!"})
```

### Interactive API Docs

Navigate to `http://localhost:8000/docs` for Swagger UI where you can:
- See all endpoints with descriptions
- Test requests interactively
- View request/response schemas
- Download OpenAPI spec

---

## Error Handling in Action

Test validation:

```bash
# Empty text - validation error
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": ""}'

# Response (400):
# {
#   "detail": [
#     {
#       "type": "string_too_short",
#       "loc": ["body", "text"],
#       "msg": "String should have at least 1 character"
#     }
#   ]
# }

# Missing field
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{}'

# Response (422):
# {
#   "detail": [
#     {
#       "type": "missing",
#       "loc": ["body", "text"],
#       "msg": "Field required"
#     }
#   ]
# }

# Text too long (exceeds max_length)
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "'$(python -c "print('x' * 6000)"))'"}'

# Response (400):
# {"error": "Validation Error", "detail": "String should have at most 5000 characters"}
```

---

## Exercises

### Exercise 1: Add a Confidence Threshold

Modify the `/predict` endpoint to accept an optional `min_confidence` parameter. If the model's confidence is below this threshold, return `"sentiment": "uncertain"` instead.

**Hints:**
- Add field to `PredictionRequest`: `min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)`
- Update `PredictionResponse` to allow `"uncertain"` as a sentiment value
- Add logic in the endpoint

### Exercise 2: Add Rate Limiting Metadata

Add response headers showing rate limit status:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Seconds until reset

**Hints:**
- Create a simple in-memory counter
- Add a middleware to check and update counts
- Return 429 Too Many Requests when exceeded

### Exercise 3: Add Request Logging to File

Create a middleware that logs each prediction request to a JSON Lines file with:
- Timestamp
- Request ID
- Input text (truncated)
- Prediction result
- Latency

This pattern is essential for debugging and audit trails.

---

## Production Considerations

### What We Covered
- Input validation with Pydantic
- Structured error responses
- Request tracing with IDs
- Health checks for orchestration
- Model lifecycle management

### What's Coming in Later Lessons
- **Lesson 3.3**: Containerizing this API with Docker
- **Lesson 4.2**: Adding proper monitoring and metrics
- **Lesson 4.3**: Input drift detection

### Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | Better Approach |
|--------------|--------------|-----------------|
| Loading model per request | Slow, wastes memory | Lifespan loading |
| Catching all exceptions | Hides bugs | Specific handlers |
| No input validation | Security risk, crashes | Pydantic schemas |
| Returning raw exceptions | Leaks internals | Structured errors |
| No health endpoint | Can't integrate with k8s | Always include `/health` |

---

## Key Takeaways

1. **FastAPI + Pydantic = type-safe ML APIs** - Validation happens automatically before your code runs

2. **Load models at startup** - Use lifespan context managers, not per-request loading

3. **Structure matters** - Separate schemas, API logic, and model code

4. **Health checks are non-negotiable** - Every production API needs `/health`

5. **Request IDs enable debugging** - Trace issues across logs and services

6. **Validate inputs aggressively** - Better to reject bad input than crash on inference

---

## Files Created This Lesson

```
project/src/schemas.py   # Request/response models
project/src/api.py       # FastAPI application
```

---

## Next Steps

You now have a working API. But how do you deploy it consistently across environments?

**Next Lesson:** [3.3 - Containerization with Docker](/start-3-3)

You'll learn to:
- Write production Dockerfiles for ML
- Handle model files in containers
- Optimize image size and build time
- Run and test containers locally
