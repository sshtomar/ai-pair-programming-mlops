# Lesson 3.1: Model Serving Options

## Learning Objectives

By the end of this lesson, students will:
1. Compare batch vs real-time inference and when to use each
2. Understand latency vs throughput tradeoffs
3. Choose serving strategy based on use case requirements

## Duration: 30 minutes

---

## Part 1: The Serving Decision

Before writing any deployment code, answer one question: **When does the prediction need to happen?**

| Timing | Pattern | Example |
|--------|---------|---------|
| Before user requests | Batch | Nightly product recommendations |
| When user requests | Real-time | Fraud detection at checkout |
| Both | Hybrid | Cached recommendations + real-time personalization |

This decision cascades through your entire architecture: infrastructure, cost, complexity, and failure modes.

### Batch vs Real-Time: Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVING PATTERN COMPARISON                          │
├───────────────────────────────────┬─────────────────────────────────────────┤
│         BATCH INFERENCE           │         REAL-TIME INFERENCE             │
├───────────────────────────────────┼─────────────────────────────────────────┤
│                                   │                                         │
│   ┌──────────┐                    │         ┌─────────┐                     │
│   │  Input   │                    │         │ Request │                     │
│   │   Data   │ (Known ahead)      │         │  (now)  │ (Unknown)           │
│   └────┬─────┘                    │         └────┬────┘                     │
│        │                          │              │                          │
│        ▼                          │              ▼                          │
│   ┌──────────┐                    │         ┌─────────┐                     │
│   │  Model   │ Process all        │         │  Model  │ Process one         │
│   │  (GPU)   │ at once            │         │  (API)  │ immediately         │
│   └────┬─────┘                    │         └────┬────┘                     │
│        │                          │              │                          │
│        ▼                          │              ▼                          │
│   ┌──────────┐                    │         ┌─────────┐                     │
│   │  Store   │ Cache results      │         │ Return  │ Instant response    │
│   │ Results  │                    │         │ Result  │                     │
│   └──────────┘                    │         └─────────┘                     │
│                                   │                                         │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ Latency:     Minutes to hours     │ Latency:     Milliseconds               │
│ Throughput:  Very high            │ Throughput:  Per-request                │
│ Cost:        Low (scale to zero)  │ Cost:        Higher (always on)         │
│ Freshness:   Stale                │ Freshness:   Real-time                  │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

---

## Part 2: Batch Inference

### When to Use Batch

Use batch inference when:
- Predictions can be precomputed (known inputs)
- Latency tolerance is high (minutes to hours acceptable)
- Input data arrives in batches (daily logs, weekly reports)
- Cost efficiency matters more than freshness

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Data       │────▶│  Feature    │────▶│  Model      │────▶│  Result     │
│  Warehouse  │     │  Pipeline   │     │  Inference  │     │  Store      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                                                            │
      └──────────────── Scheduled (cron, Airflow) ─────────────────┘
```

### Real-World Examples

**Recommendation Systems (Netflix, Spotify)**
- Compute recommendations for all users overnight
- Store in fast key-value cache (Redis, DynamoDB)
- Serve precomputed results instantly
- Recompute daily or when behavior changes significantly

**Credit Risk Scoring**
- Score all accounts monthly
- Flag high-risk accounts for review
- Batch job runs on large cluster, then shuts down

**Email Classification**
- Process incoming emails in 5-minute batches
- Categorize, tag, and route
- Small delay acceptable for email

### Batch Inference Code Pattern

```python
"""Batch inference pipeline."""
import pandas as pd
from datetime import datetime
from pathlib import Path

def run_batch_inference(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    batch_size: int = 1000,
) -> dict:
    """Run batch inference on a dataset.

    Args:
        input_path: Path to input data (CSV, Parquet, etc.)
        output_path: Path to write predictions
        model_path: Path to trained model
        batch_size: Number of samples to process at once

    Returns:
        dict with run metadata (count, duration, etc.)
    """
    from src.predict import SentimentClassifier

    start_time = datetime.now()

    # Load model once
    model = SentimentClassifier.load(model_path)

    # Process in chunks to manage memory
    predictions = []
    total_processed = 0

    for chunk in pd.read_csv(input_path, chunksize=batch_size):
        chunk_preds = [
            model.predict(text) for text in chunk["text"]
        ]
        predictions.extend(chunk_preds)
        total_processed += len(chunk)

        # Log progress
        print(f"Processed {total_processed} samples...")

    # Write results
    results_df = pd.DataFrame({
        "prediction": predictions,
        "processed_at": datetime.now().isoformat(),
    })
    results_df.to_parquet(output_path, index=False)

    duration = (datetime.now() - start_time).total_seconds()

    return {
        "total_processed": total_processed,
        "duration_seconds": duration,
        "throughput": total_processed / duration,
    }
```

### Pros and Cons

| Pros | Cons |
|------|------|
| High throughput (optimize for bulk) | Stale predictions |
| Cost-efficient (scale to zero between runs) | Cannot handle new inputs |
| Simple failure handling (retry entire job) | Storage costs for precomputed results |
| Easy to test (deterministic inputs) | Complex orchestration for dependencies |
| GPU batching maximizes utilization | Cold start when data patterns change |

---

## Part 3: Real-Time Inference

### When to Use Real-Time

Use real-time inference when:
- Input is unknown until request time
- Low latency is critical (< 100ms)
- Predictions influence immediate user actions
- Input space is too large to precompute

### Architecture

```
                                    ┌─────────────────┐
                                    │  Load Balancer  │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
    ┌─────────▼─────────┐        ┌───────────▼───────────┐        ┌─────────▼─────────┐
    │  Model Server 1   │        │   Model Server 2      │        │  Model Server 3   │
    │  (GPU/CPU)        │        │   (GPU/CPU)           │        │  (GPU/CPU)        │
    └───────────────────┘        └───────────────────────┘        └───────────────────┘
```

### Real-World Examples

**Fraud Detection (Stripe, PayPal)**
- Every transaction scored in < 50ms
- Block or flag before charge completes
- Cannot precompute: infinite input combinations

**Content Moderation (Social Media)**
- Score posts/comments before publishing
- Block harmful content immediately
- Latency budget: 100-500ms

**Search Ranking (Google, Amazon)**
- Re-rank results based on user context
- Personalize in real-time
- Latency directly impacts revenue

**Our Sentiment Classifier**
- Analyze customer feedback as it arrives
- Route urgent negative feedback to support
- Enable real-time dashboard updates

### Real-Time Inference Code Pattern

```python
"""Real-time inference with FastAPI."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from functools import lru_cache
import time

app = FastAPI(title="Sentiment Classifier API")


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    latency_ms: float


@lru_cache(maxsize=1)
def get_model():
    """Load model once, cache for all requests."""
    from src.predict import SentimentClassifier
    return SentimentClassifier.load("models/sentiment_model.pkl")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict sentiment for input text."""
    start = time.perf_counter()

    model = get_model()

    sentiment = model.predict(request.text)
    probabilities = model.predict_proba(request.text)
    confidence = max(probabilities.values())

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        sentiment=sentiment,
        confidence=confidence,
        latency_ms=latency_ms,
    )


@app.get("/health")
def health():
    """Health check endpoint."""
    # Verify model is loaded and functional
    model = get_model()
    _ = model.predict("test")
    return {"status": "healthy"}
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Fresh predictions on any input | Higher infrastructure cost (always-on) |
| Immediate response to new data | Latency constraints limit model complexity |
| No storage of precomputed results | Complex scaling under load |
| Simpler data pipeline (no batch jobs) | Every request is a potential failure |
| Handles infinite input combinations | Cold start on first request |

---

## Part 4: Latency vs Throughput Tradeoffs

These are inversely related. Optimizing for one often hurts the other.

### The Tradeoff Explained

```
Throughput (requests/sec)
    ▲
    │
    │    ╭──────────────────╮
    │   ╱                    ╲
    │  ╱                      ╲
    │ ╱  Batch: High throughput ╲
    │╱   (process 10K together)  ╲
    ├────────────────────────────────▶ Latency (ms)
    │
    │  Real-time: Low latency
    │  (respond in 50ms)
```

### Optimization Strategies

| Goal | Strategy | Tradeoff |
|------|----------|----------|
| Lower latency | Smaller models | Lower accuracy |
| Lower latency | Model distillation | Training complexity |
| Lower latency | Caching | Memory cost, staleness |
| Lower latency | Edge deployment | Operational complexity |
| Higher throughput | Request batching | Higher latency |
| Higher throughput | Async processing | Implementation complexity |
| Higher throughput | GPU utilization | Infrastructure cost |

### Latency Budgets

Break down your latency target:

```
Total latency budget: 100ms
├── Network (client → server): 20ms
├── Load balancer: 2ms
├── Request parsing: 1ms
├── Feature extraction: 15ms
├── Model inference: 40ms  ← Your model gets this much
├── Response serialization: 2ms
└── Network (server → client): 20ms
```

If your model takes 80ms, you've already failed before adding network latency.

### Measuring What Matters

```python
"""Latency measurement patterns."""
import time
from dataclasses import dataclass
from typing import List
import statistics


@dataclass
class LatencyMetrics:
    p50: float  # Median
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    mean: float
    max: float


def measure_latency(
    predict_fn,
    test_inputs: List[str],
    warmup: int = 10,
) -> LatencyMetrics:
    """Measure prediction latency distribution.

    Args:
        predict_fn: Function that takes text, returns prediction
        test_inputs: List of test texts
        warmup: Number of warmup calls (not measured)
    """
    # Warmup (important for JIT compilation, caching)
    for text in test_inputs[:warmup]:
        predict_fn(text)

    # Measure
    latencies = []
    for text in test_inputs:
        start = time.perf_counter()
        predict_fn(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    n = len(latencies)

    return LatencyMetrics(
        p50=latencies[n // 2],
        p95=latencies[int(n * 0.95)],
        p99=latencies[int(n * 0.99)],
        mean=statistics.mean(latencies),
        max=max(latencies),
    )
```

**Report p95 and p99, not mean.** Mean hides tail latency. If your p99 is 500ms but mean is 50ms, 1% of users wait 10x longer.

---

## Part 5: Hybrid Approaches

Real systems often combine batch and real-time.

### Pattern 1: Precompute + Fallback

```
Request arrives
    │
    ▼
┌─────────────────────┐
│ Check cache for     │
│ precomputed result  │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
   Found      Not Found
     │           │
     ▼           ▼
  Return     Real-time
  cached     inference
  result        │
                ▼
            Cache result
            for next time
```

**Example: Product Recommendations**
- Precompute recommendations for active users (batch)
- New users or cold items: real-time fallback
- Cache real-time results for future requests

### Pattern 2: Feature Store

```
┌────────────────────────────────────────────────────────────────┐
│                        Feature Store                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ User        │  │ Product     │  │ Real-time   │            │
│  │ Features    │  │ Features    │  │ Features    │            │
│  │ (batch)     │  │ (batch)     │  │ (streaming) │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└────────────────────────────────────────────────────────────────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Model Server  │
                    │ (real-time)   │
                    └───────────────┘
```

- Batch pipeline computes slow features daily
- Streaming pipeline updates fast features in real-time
- Model server combines both at inference time

### Pattern 3: Prediction Caching

```python
"""Prediction caching for common inputs."""
from functools import lru_cache
import hashlib


class CachedPredictor:
    def __init__(self, model, cache_size: int = 10000):
        self.model = model
        self.cache_size = cache_size
        self._cache = {}

    def _normalize(self, text: str) -> str:
        """Normalize text for cache key."""
        return text.lower().strip()

    def _cache_key(self, text: str) -> str:
        """Create cache key from text."""
        normalized = self._normalize(text)
        return hashlib.md5(normalized.encode()).hexdigest()

    def predict(self, text: str) -> str:
        """Predict with caching."""
        key = self._cache_key(text)

        if key in self._cache:
            return self._cache[key]

        prediction = self.model.predict(text)

        # Simple LRU: evict oldest if at capacity
        if len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = prediction
        return prediction
```

**When caching works:**
- Repeated queries are common (search, FAQ)
- Inputs can be normalized (case, whitespace)
- Staleness is acceptable

**When caching fails:**
- Unique inputs (personalized content)
- Time-sensitive predictions (stock prices)
- Large input space with uniform distribution

---

## Part 6: Serving Patterns

### REST API

The most common pattern. HTTP + JSON.

```python
# Client
import requests

response = requests.post(
    "https://api.example.com/predict",
    json={"text": "Great product!"},
)
prediction = response.json()["sentiment"]
```

| Pros | Cons |
|------|------|
| Universal compatibility | HTTP overhead (~10ms) |
| Easy debugging (curl, browser) | JSON serialization cost |
| Extensive tooling | Not ideal for streaming |
| Familiar to all developers | Verbose for batch requests |

### gRPC

Binary protocol with code generation. 2-10x faster than REST for high-throughput.

```protobuf
// sentiment.proto
service SentimentService {
    rpc Predict(PredictRequest) returns (PredictResponse);
    rpc PredictBatch(PredictBatchRequest) returns (PredictBatchResponse);
    rpc PredictStream(stream PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
    string text = 1;
}

message PredictResponse {
    string sentiment = 1;
    float confidence = 2;
}
```

| Pros | Cons |
|------|------|
| Fast (binary, HTTP/2) | Requires code generation |
| Strong typing | Harder to debug |
| Bi-directional streaming | Less universal tooling |
| Built-in load balancing | Browser support limited |

### Streaming

For continuous data or long-running predictions.

```python
"""Server-Sent Events for streaming predictions."""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()


async def stream_predictions(texts: list[str]):
    """Stream predictions as they complete."""
    model = get_model()

    for i, text in enumerate(texts):
        prediction = model.predict(text)

        yield json.dumps({
            "index": i,
            "text": text,
            "prediction": prediction,
        }) + "\n"

        await asyncio.sleep(0)  # Yield control


@app.post("/predict/stream")
async def predict_stream(request: BatchRequest):
    """Stream predictions for batch request."""
    return StreamingResponse(
        stream_predictions(request.texts),
        media_type="application/x-ndjson",
    )
```

| Use Case | Pattern |
|----------|---------|
| Large batch with progress | Streaming |
| Real-time audio/video | WebSocket |
| Chat/conversational AI | Server-Sent Events |
| One-shot prediction | REST/gRPC |

### Embedded Inference

Model runs in the client application. No network call.

```python
"""Embedded inference for edge deployment."""
# Export model to portable format
import onnxruntime as ort

# Client-side (mobile, browser, IoT)
session = ort.InferenceSession("model.onnx")
inputs = {"text": ["Great product!"]}
outputs = session.run(None, inputs)
```

| Pros | Cons |
|------|------|
| Zero network latency | Model update requires app update |
| Works offline | Limited compute on device |
| Privacy (data stays local) | Larger app size |
| Scales with users (no server) | Platform-specific optimization |

---

## Part 7: Infrastructure Options

### Comparison Matrix

| Option | Latency | Scale | Cost Model | Complexity | Best For |
|--------|---------|-------|------------|------------|----------|
| **Containers (K8s)** | Low | Manual/Auto | Always-on | High | High-traffic production |
| **Serverless (Lambda)** | Medium | Automatic | Per-request | Low | Sporadic traffic |
| **Managed ML (SageMaker)** | Low | Automatic | Endpoint hours | Medium | AWS-native teams |
| **Edge (CloudFlare Workers)** | Very Low | Global | Per-request | Medium | Latency-critical |

### Container Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-classifier
  template:
    spec:
      containers:
        - name: model-server
          image: sentiment-classifier:v1.2.3
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
```

### Serverless Deployment

```python
"""AWS Lambda handler for sentiment prediction."""
import json
import boto3

# Load model at cold start (outside handler)
model = None

def load_model():
    global model
    if model is None:
        from src.predict import SentimentClassifier
        model = SentimentClassifier.load("/opt/model/sentiment.pkl")
    return model


def handler(event, context):
    """Lambda handler for prediction requests."""
    body = json.loads(event["body"])
    text = body["text"]

    classifier = load_model()
    prediction = classifier.predict(text)

    return {
        "statusCode": 200,
        "body": json.dumps({"sentiment": prediction}),
    }
```

**Cold start warning:** First request after idle period takes 1-10 seconds. Mitigate with:
- Provisioned concurrency (keeps instances warm)
- Smaller model/dependencies
- Ping endpoint periodically

---

## Part 8: Decision Framework

### Questions to Ask

Answer these before choosing a serving strategy:

| Question | Batch | Real-Time | Hybrid |
|----------|-------|-----------|--------|
| Are inputs known in advance? | Yes | No | Some |
| Is latency critical (< 100ms)? | No | Yes | Depends |
| Is traffic predictable? | Yes | Variable | Variable |
| Can predictions be cached? | Yes | No | Partially |
| What's the cost tolerance? | Low | Higher | Medium |
| How often do inputs change? | Rarely | Constantly | Mixed |

### Decision Tree

```
Is the input known before the user requests it?
├── YES: Can you tolerate stale predictions (hours/days old)?
│   ├── YES → BATCH
│   └── NO → HYBRID (batch + real-time refresh)
└── NO: Is latency critical (< 100ms)?
    ├── YES: Is traffic predictable and high?
    │   ├── YES → CONTAINERS with autoscaling
    │   └── NO → SERVERLESS with provisioned concurrency
    └── NO: Consider BATCH with queue-based processing
```

---

## Exercise: Sentiment Classifier Serving Decision

Apply the decision framework to your sentiment classifier for three scenarios:

### Scenario A: Product Review Dashboard

**Context:** Marketing team wants daily sentiment summary of product reviews from the previous day.

**Questions to answer:**
1. Are inputs known in advance?
2. What's the latency tolerance?
3. What's the expected volume?
4. Recommended serving strategy?

<details>
<summary>Analysis</summary>

1. **Inputs known?** Yes - yesterday's reviews are collected
2. **Latency tolerance?** High - dashboard updated once daily
3. **Volume?** 1,000-10,000 reviews per day
4. **Strategy:** **Batch inference**
   - Run nightly job to process all reviews
   - Store aggregated results in database
   - Dashboard queries precomputed data

**Architecture:**
```
Daily Reviews → Batch Job (1am) → Results DB → Dashboard
```

</details>

### Scenario B: Customer Support Routing

**Context:** Route incoming support tickets to appropriate teams based on sentiment. Angry customers go to senior agents immediately.

**Questions to answer:**
1. Are inputs known in advance?
2. What's the latency tolerance?
3. What's the expected volume?
4. Recommended serving strategy?

<details>
<summary>Analysis</summary>

1. **Inputs known?** No - tickets arrive unpredictably
2. **Latency tolerance?** Low - need to route before human sees it
3. **Volume?** 100-1,000 tickets per day, bursty
4. **Strategy:** **Real-time with serverless**
   - AWS Lambda or Cloud Functions
   - Trigger on ticket creation
   - Route based on prediction

**Architecture:**
```
Ticket Created → Lambda → Predict → Route to Queue
```

</details>

### Scenario C: Social Media Monitoring

**Context:** Monitor brand mentions across Twitter/Reddit. Alert on negative spikes. Real-time dashboard for current sentiment.

**Questions to answer:**
1. Are inputs known in advance?
2. What's the latency tolerance?
3. What's the expected volume?
4. Recommended serving strategy?

<details>
<summary>Analysis</summary>

1. **Inputs known?** No - streaming from social APIs
2. **Latency tolerance?** Medium (1-5 minutes acceptable)
3. **Volume?** Highly variable, 100-100,000/day
4. **Strategy:** **Hybrid with streaming**
   - Stream ingestion (Kafka/Kinesis)
   - Micro-batch processing (1-minute windows)
   - Cache recent results for dashboard
   - Batch for historical analysis

**Architecture:**
```
Social Streams → Kafka → Spark Streaming (1min) → Redis Cache → Dashboard
                                               → S3 → Daily Batch → Analytics
```

</details>

---

## Key Takeaways

1. **Choose batch when inputs are known and latency tolerance is high.** Lower cost, simpler architecture, higher throughput.

2. **Choose real-time when inputs are unknown and latency is critical.** Higher cost, more complex, but necessary for interactive use cases.

3. **Hybrid approaches are common in production.** Precompute what you can, fall back to real-time for the rest.

4. **Measure p95/p99 latency, not mean.** Tail latency affects user experience more than average.

5. **Your model latency budget is smaller than you think.** After network, parsing, and serialization, you may have 40ms for inference.

6. **Match infrastructure to traffic patterns.** Containers for steady high traffic, serverless for sporadic traffic.

7. **Caching can convert real-time to batch.** If inputs repeat, cache predictions aggressively.

---

## Next Steps

You've learned how to choose a serving strategy. Now implement it.

Run `/start-3-2` to begin **Lesson 3.2: Building a Prediction API with FastAPI**. You'll build a production-ready REST API for your sentiment classifier with:
- Request validation
- Error handling
- Health checks
- Structured logging
- Docker packaging
