# Lesson 4.2: Monitoring & Observability

## Learning Objectives

By the end of this lesson, students will:
1. Distinguish system metrics from model metrics
2. Implement logging and alerting
3. Build prediction monitoring dashboard

## Duration: 45 minutes

---

## Part 1: The Three Pillars of Observability

Production ML systems fail in ways that unit tests never catch. A model with 95% test accuracy can silently degrade to 60% in production. Without observability, you won't know until users complain.

### Monitoring Stack Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Production Monitoring Flow                           │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           YOUR ML APPLICATION                            │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
  │  │   FastAPI   │  │   Model     │  │  Metrics    │  │  Structured     │ │
  │  │   Server    │──│  Inference  │──│  /metrics   │  │  Logs (JSON)    │ │
  │  └─────────────┘  └─────────────┘  └──────┬──────┘  └────────┬────────┘ │
  └────────────────────────────────────────────┼─────────────────┼──────────┘
                                               │                 │
            ┌──────────────────────────────────┘                 │
            │                                                    │
            ▼                                                    ▼
  ┌─────────────────────┐                           ┌─────────────────────┐
  │     PROMETHEUS      │                           │   LOG AGGREGATOR    │
  │  ┌───────────────┐  │                           │  ┌───────────────┐  │
  │  │ Scrapes every │  │                           │  │ ELK / Loki /  │  │
  │  │ 15 seconds    │  │                           │  │ CloudWatch    │  │
  │  └───────────────┘  │                           │  └───────────────┘  │
  │  Time-series DB     │                           │  Searchable logs    │
  └──────────┬──────────┘                           └──────────┬──────────┘
             │                                                  │
             ▼                                                  ▼
  ┌─────────────────────┐                           ┌─────────────────────┐
  │      GRAFANA        │                           │    LOG DASHBOARD    │
  │  ┌───────────────┐  │                           │  ┌───────────────┐  │
  │  │ Dashboards    │  │                           │  │ Kibana /      │  │
  │  │ Visualizations│  │                           │  │ Grafana Loki  │  │
  │  └───────────────┘  │                           │  └───────────────┘  │
  └──────────┬──────────┘                           └─────────────────────┘
             │
             ▼
  ┌─────────────────────┐        ┌─────────────────────┐
  │   ALERTMANAGER      │───────>│   NOTIFICATIONS     │
  │  ┌───────────────┐  │        │  ┌───────────────┐  │
  │  │ Alert rules   │  │        │  │ Slack         │  │
  │  │ Thresholds    │  │        │  │ PagerDuty     │  │
  │  └───────────────┘  │        │  │ Email         │  │
  └─────────────────────┘        │  └───────────────┘  │
                                 └─────────────────────┘
```

### Logs, Metrics, Traces

| Pillar | What It Captures | Example |
|--------|------------------|---------|
| **Logs** | Discrete events with context | `ERROR: Prediction failed for request_id=abc123` |
| **Metrics** | Numeric measurements over time | `prediction_latency_p99 = 145ms` |
| **Traces** | Request flow across services | `request -> API -> model -> cache -> response (234ms)` |

Each serves a different purpose:

- **Logs**: Answer "what happened?" for specific events
- **Metrics**: Answer "how is the system performing?" over time
- **Traces**: Answer "where did time go?" for slow requests

### ML-Specific Observability

Standard web observability covers system health. ML systems need additional monitoring:

```
System Metrics          Model Metrics
---------------         ---------------
Latency                 Prediction distribution
Error rate              Confidence scores
Throughput              Input feature drift
CPU/Memory              Output distribution shift
```

A model can be fast and available (system metrics green) while giving garbage predictions (model metrics red). You need both.

---

## Part 2: System Metrics for ML APIs

### The Four Golden Signals

Google's SRE book defines four signals for any service:

| Signal | What to Measure | Threshold |
|--------|-----------------|-----------|
| **Latency** | Time to serve a request | p99 < 200ms |
| **Traffic** | Requests per second | Capacity planning |
| **Errors** | Failed requests / total | < 1% |
| **Saturation** | Resource utilization | CPU < 80% |

For ML APIs, add model-specific signals:

| Signal | What to Measure | Why |
|--------|-----------------|-----|
| **Model latency** | Inference time only | Separate from I/O overhead |
| **Batch size** | Requests batched together | Throughput optimization |
| **Feature extraction time** | Preprocessing duration | Often the bottleneck |

### Prometheus Metrics Types

Prometheus uses four metric types:

```python
from prometheus_client import Counter, Histogram, Gauge, Info

# Counter: Only goes up (requests, errors)
predictions_total = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_version", "predicted_label"]
)

# Histogram: Distribution of values (latency, confidence)
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time to make prediction",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Gauge: Current value (queue size, active connections)
model_loaded = Gauge(
    "model_loaded",
    "Whether model is loaded",
    ["model_version"]
)

# Info: Static metadata
model_info = Info(
    "model",
    "Model metadata"
)
```

---

## Part 3: Model Metrics

### Prediction Distribution Monitoring

The most important model metric: **Are predictions shifting?**

```python
from prometheus_client import Counter, Histogram

# Track prediction distribution
prediction_distribution = Counter(
    "prediction_distribution_total",
    "Predictions by label",
    ["label"]
)

# Track confidence scores
prediction_confidence = Histogram(
    "prediction_confidence",
    "Model confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)
```

Why this matters:

| Scenario | System Metrics | Model Metrics |
|----------|---------------|---------------|
| Model predicting all "positive" | Green | Distribution skewed |
| Low confidence predictions | Green | Confidence histogram shifts left |
| Input drift | Green | Feature distributions change |

### Key Model Metrics to Track

```python
# 1. Prediction counts by label
predictions_total = Counter(
    "predictions_total",
    "Total predictions",
    ["model_version", "label"]
)

# 2. Confidence distribution
prediction_confidence = Histogram(
    "prediction_confidence",
    "Prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# 3. Input text length (proxy for input drift)
input_length = Histogram(
    "input_text_length",
    "Input text character count",
    buckets=[10, 50, 100, 250, 500, 1000, 2000]
)

# 4. Model version info
model_info = Info("model", "Model metadata")
model_info.info({
    "version": "1.2.0",
    "training_date": "2024-01-15",
    "accuracy": "0.89"
})
```

---

## Part 4: Instrumenting FastAPI

### Complete Metrics Implementation

```python
# src/metrics.py
"""Prometheus metrics for ML API."""
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable

# System metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Model metrics
predictions_total = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_version", "label"]
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Model inference time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)

prediction_confidence = Histogram(
    "prediction_confidence",
    "Prediction confidence score",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

input_text_length = Histogram(
    "input_text_length",
    "Input text character count",
    buckets=[10, 50, 100, 250, 500, 1000, 2000]
)

# Model state
model_loaded = Gauge(
    "model_loaded",
    "Whether model is loaded (1=yes, 0=no)"
)

model_info = Info("model", "Model metadata")


def track_prediction(model_version: str):
    """Decorator to track prediction metrics."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(text: str, *args, **kwargs):
            # Track input
            input_text_length.observe(len(text))

            # Time the prediction
            start = time.perf_counter()
            result = func(text, *args, **kwargs)
            duration = time.perf_counter() - start

            # Track outputs
            prediction_latency_seconds.observe(duration)
            predictions_total.labels(
                model_version=model_version,
                label=result["label"]
            ).inc()
            prediction_confidence.observe(result["confidence"])

            return result
        return wrapper
    return decorator
```

### Integrating with FastAPI

```python
# src/app.py
"""FastAPI application with Prometheus metrics."""
from fastapi import FastAPI, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
import uuid

from src.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    model_loaded,
    model_info,
    track_prediction,
)
from src.model import SentimentClassifier

app = FastAPI(title="Sentiment API")

# Load model on startup
classifier: SentimentClassifier = None
MODEL_VERSION = "1.2.0"


@app.on_event("startup")
async def startup():
    global classifier
    classifier = SentimentClassifier.load("models/model.pkl")
    model_loaded.set(1)
    model_info.info({
        "version": MODEL_VERSION,
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track HTTP metrics for all requests."""
    start = time.perf_counter()

    # Add request ID for tracing
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    response = await call_next(request)

    duration = time.perf_counter() - start

    # Record metrics
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    http_request_duration_seconds.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    # Add request ID to response for tracing
    response.headers["X-Request-ID"] = request_id

    return response


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
async def predict(request: Request):
    """Make a prediction."""
    body = await request.json()
    text = body["text"]
    request_id = request.headers.get("X-Request-ID", "unknown")

    # Prediction with metrics tracking
    start = time.perf_counter()
    result = classifier.predict_with_confidence(text)
    duration = time.perf_counter() - start

    # Log structured output
    import structlog
    logger = structlog.get_logger()
    logger.info(
        "prediction_complete",
        request_id=request_id,
        input_length=len(text),
        label=result["label"],
        confidence=result["confidence"],
        latency_ms=duration * 1000
    )

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "model_version": MODEL_VERSION,
        "request_id": request_id
    }
```

---

## Part 5: Structured Logging

### Why Structured Logs?

Unstructured logs are hard to query:

```
2024-01-15 10:23:45 INFO Prediction complete for user request
2024-01-15 10:23:46 ERROR Failed to process input
```

Structured logs are machine-parseable:

```json
{"timestamp": "2024-01-15T10:23:45Z", "level": "info", "event": "prediction_complete", "request_id": "abc123", "label": "positive", "confidence": 0.92, "latency_ms": 45.2}
{"timestamp": "2024-01-15T10:23:46Z", "level": "error", "event": "prediction_failed", "request_id": "def456", "error": "EmptyInputError", "input_length": 0}
```

### Implementing with structlog

```python
# src/logging_config.py
"""Structured logging configuration."""
import structlog
import logging
import sys


def configure_logging(json_output: bool = True):
    """Configure structured logging.

    Args:
        json_output: If True, output JSON. If False, output human-readable.
    """
    # Shared processors for all output formats
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: Human-readable output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )


def get_logger(name: str = None):
    """Get a configured logger."""
    return structlog.get_logger(name)
```

### Logging with Context

```python
# src/predict.py
from src.logging_config import get_logger

logger = get_logger(__name__)


class SentimentClassifier:
    def predict_with_logging(self, text: str, request_id: str):
        """Make prediction with structured logging."""

        # Bind context that follows through all log calls
        log = logger.bind(
            request_id=request_id,
            input_length=len(text)
        )

        log.info("prediction_started")

        try:
            # Preprocess
            cleaned = self.preprocess(text)
            log.debug("preprocessing_complete", cleaned_length=len(cleaned))

            # Feature extraction
            features = self.extract_features(cleaned)
            log.debug("features_extracted", feature_count=len(features))

            # Inference
            start = time.perf_counter()
            label, confidence = self.model.predict(features)
            latency_ms = (time.perf_counter() - start) * 1000

            log.info(
                "prediction_complete",
                label=label,
                confidence=confidence,
                latency_ms=latency_ms
            )

            return {"label": label, "confidence": confidence}

        except ValueError as e:
            log.warning("prediction_failed", error=str(e), error_type="ValueError")
            raise
        except Exception as e:
            log.error("prediction_error", error=str(e), error_type=type(e).__name__)
            raise
```

### Request ID Propagation

Request IDs enable tracing across services:

```python
# src/middleware.py
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from src.logging_config import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests for tracing."""

    async def dispatch(self, request: Request, call_next):
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in request state for access in handlers
        request.state.request_id = request_id

        # Bind to logger context
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            logger.info(
                "request_started",
                method=request.method,
                path=request.url.path
            )

            response = await call_next(request)

            logger.info(
                "request_complete",
                status_code=response.status_code
            )

        response.headers["X-Request-ID"] = request_id
        return response
```

---

## Part 6: Alerting Rules

### Prometheus Alerting Rules

Create `prometheus/alerts.yml`:

```yaml
groups:
  - name: ml-api-alerts
    rules:
      # High latency alert
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency (p99 > 200ms)"
          description: "99th percentile latency is {{ $value | humanizeDuration }}"

      # Error rate alert
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          /
          sum(rate(http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate (> 1%)"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Model not loaded
      - alert: ModelNotLoaded
        expr: model_loaded == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model not loaded"
          description: "The ML model failed to load or was unloaded"

      # Low confidence predictions
      - alert: LowConfidencePredictions
        expr: |
          histogram_quantile(0.5, rate(prediction_confidence_bucket[1h])) < 0.7
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Median prediction confidence below 70%"
          description: "Model may be encountering out-of-distribution inputs"

      # Prediction distribution shift
      - alert: PredictionDistributionShift
        expr: |
          abs(
            sum(rate(predictions_total{label="positive"}[1h]))
            /
            sum(rate(predictions_total[1h]))
            - 0.33
          ) > 0.15
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Prediction distribution shift detected"
          description: "Positive prediction ratio deviated >15% from baseline"

      # No predictions (traffic drop)
      - alert: NoPredictions
        expr: sum(rate(predictions_total[5m])) == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "No predictions in last 10 minutes"
          description: "API may be down or no traffic is reaching it"
```

### When to Page vs. When to Ticket

| Condition | Action | Why |
|-----------|--------|-----|
| Service down | Page | Immediate revenue/user impact |
| Error rate > 5% | Page | Active incident |
| Error rate 1-5% | Ticket | Degraded but functional |
| High latency | Ticket | UX impact, not critical |
| Low confidence | Ticket | Investigate during business hours |
| Distribution shift | Ticket | May be legitimate traffic change |

```yaml
# Example severity routing
routes:
  - match:
      severity: critical
    receiver: pagerduty-oncall

  - match:
      severity: warning
    receiver: slack-ml-alerts
    group_wait: 30m  # Batch warnings together
```

---

## Part 7: Dashboard Design

### Essential Dashboard Panels

A production ML dashboard should show:

**Row 1: Service Health (System)**
```
+------------------+------------------+------------------+
|  Request Rate    |  Error Rate      |  Latency p50/p99 |
|  (req/sec)       |  (%)             |  (ms)            |
+------------------+------------------+------------------+
```

**Row 2: Model Performance**
```
+------------------+------------------+------------------+
|  Predictions/sec |  Confidence      |  Model Version   |
|  by Label        |  Distribution    |  (current)       |
+------------------+------------------+------------------+
```

**Row 3: Drift Detection**
```
+------------------+------------------+------------------+
|  Label           |  Input Length    |  Confidence      |
|  Distribution    |  Distribution    |  Over Time       |
|  Over Time       |  Over Time       |                  |
+------------------+------------------+------------------+
```

**Row 4: Resources**
```
+------------------+------------------+------------------+
|  CPU Usage       |  Memory Usage    |  Active          |
|  (%)             |  (MB)            |  Connections     |
+------------------+------------------+------------------+
```

### Grafana Dashboard JSON

```json
{
  "title": "ML API - Sentiment Classifier",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [
        {
          "expr": "sum(rate(http_requests_total[1m]))",
          "legendFormat": "requests/sec"
        }
      ]
    },
    {
      "title": "Error Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100"
        }
      ],
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 1},
          {"color": "red", "value": 5}
        ]
      }
    },
    {
      "title": "Prediction Latency",
      "type": "graph",
      "targets": [
        {
          "expr": "histogram_quantile(0.5, rate(prediction_latency_seconds_bucket[5m]))",
          "legendFormat": "p50"
        },
        {
          "expr": "histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m]))",
          "legendFormat": "p99"
        }
      ]
    },
    {
      "title": "Predictions by Label",
      "type": "graph",
      "targets": [
        {
          "expr": "sum(rate(predictions_total[5m])) by (label)",
          "legendFormat": "{{label}}"
        }
      ]
    },
    {
      "title": "Confidence Distribution",
      "type": "heatmap",
      "targets": [
        {
          "expr": "sum(rate(prediction_confidence_bucket[5m])) by (le)"
        }
      ]
    },
    {
      "title": "Label Distribution Over Time",
      "type": "graph",
      "targets": [
        {
          "expr": "sum(rate(predictions_total{label=\"positive\"}[1h])) / sum(rate(predictions_total[1h]))",
          "legendFormat": "positive %"
        },
        {
          "expr": "sum(rate(predictions_total{label=\"negative\"}[1h])) / sum(rate(predictions_total[1h]))",
          "legendFormat": "negative %"
        }
      ]
    }
  ]
}
```

---

## Part 8: Cloud-Native Monitoring

### Google Cloud Monitoring

```python
# src/cloud_monitoring.py
"""Google Cloud Monitoring integration."""
from google.cloud import monitoring_v3
import time


class CloudMetrics:
    def __init__(self, project_id: str):
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"

    def record_prediction(
        self,
        label: str,
        confidence: float,
        latency_seconds: float
    ):
        """Record prediction metrics to Cloud Monitoring."""
        now = time.time()

        # Create time series for prediction count
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/ml/predictions_total"
        series.metric.labels["label"] = label
        series.resource.type = "global"

        point = monitoring_v3.Point()
        point.value.int64_value = 1
        point.interval.end_time.seconds = int(now)
        series.points = [point]

        self.client.create_time_series(
            name=self.project_name,
            time_series=[series]
        )
```

### AWS CloudWatch

```python
# src/cloudwatch_metrics.py
"""AWS CloudWatch integration."""
import boto3
from datetime import datetime


class CloudWatchMetrics:
    def __init__(self, namespace: str = "MLApi/Sentiment"):
        self.client = boto3.client("cloudwatch")
        self.namespace = namespace

    def record_prediction(
        self,
        label: str,
        confidence: float,
        latency_seconds: float
    ):
        """Record prediction metrics to CloudWatch."""
        self.client.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    "MetricName": "PredictionCount",
                    "Dimensions": [
                        {"Name": "Label", "Value": label}
                    ],
                    "Value": 1,
                    "Unit": "Count"
                },
                {
                    "MetricName": "PredictionLatency",
                    "Value": latency_seconds * 1000,
                    "Unit": "Milliseconds"
                },
                {
                    "MetricName": "PredictionConfidence",
                    "Value": confidence,
                    "Unit": "None"
                }
            ]
        )
```

---

## Exercises

### Exercise 4.2.1: Add Prometheus Metrics

Add these metrics to your FastAPI application:

1. `predictions_total` - Counter with labels for `model_version` and `label`
2. `prediction_latency_seconds` - Histogram with appropriate buckets
3. `prediction_confidence` - Histogram for confidence score distribution
4. `model_info` - Info metric with model version and training date

Run your API, make some predictions, and verify metrics appear at `/metrics`.

### Exercise 4.2.2: Implement Structured Logging

Replace print statements with structured logging:

1. Install structlog: `pip install structlog`
2. Configure JSON output for production
3. Add request IDs to all prediction logs
4. Log prediction events with label, confidence, and latency

Make predictions and verify logs are JSON-formatted with request IDs.

### Exercise 4.2.3: Create Alert Rules

Write Prometheus alert rules for:

1. **HighLatency**: p99 latency > 500ms for 5 minutes
2. **LowConfidence**: Median confidence < 0.6 for 1 hour
3. **SkewedPredictions**: Any label > 70% of predictions for 30 minutes
4. **ServiceDown**: No requests for 5 minutes

Test by temporarily changing thresholds to trigger alerts.

### Exercise 4.2.4: Design Your Dashboard

Sketch a dashboard layout (on paper or in a doc) that shows:

1. At-a-glance service health (green/yellow/red)
2. Prediction volume and distribution
3. Latency trends
4. Drift indicators

Consider: What would you look at first during an incident?

---

## Common Pitfalls

### 1. Metric Cardinality Explosion

```python
# BAD: Unbounded label values
predictions.labels(user_id=user_id).inc()  # Millions of unique values!

# GOOD: Bounded label values
predictions.labels(label=predicted_label).inc()  # 3 possible values
```

High cardinality labels cause Prometheus to run out of memory.

### 2. Missing Request IDs

```python
# BAD: Can't trace request across logs
logger.info("Prediction failed")

# GOOD: Request ID enables tracing
logger.info("Prediction failed", request_id=request_id)
```

Without request IDs, correlating logs from a single request is impossible.

### 3. Alerting on Every Blip

```yaml
# BAD: Alerts on momentary spikes
- alert: HighLatency
  expr: prediction_latency_seconds > 0.1
  for: 0s  # Instant alert

# GOOD: Requires sustained condition
- alert: HighLatency
  expr: histogram_quantile(0.99, rate(...[5m])) > 0.2
  for: 5m  # Must persist for 5 minutes
```

Alert fatigue kills on-call engineers. Use `for:` duration.

### 4. Ignoring Model Metrics

Monitoring only system metrics misses model degradation:

| Symptom | System Metrics | Model Metrics |
|---------|---------------|---------------|
| Model returning all same label | Green | Distribution skewed |
| Model uncertain on all inputs | Green | Confidence dropping |
| Training data outdated | Green | Drift detected |

Always monitor both.

---

## Key Takeaways

1. **Three pillars**: Logs for events, metrics for aggregates, traces for request flows. You need all three for complete observability.

2. **System vs. model metrics**: Fast API with garbage predictions is still broken. Monitor prediction distribution, confidence, and drift alongside latency and errors.

3. **Structured logging enables debugging**: JSON logs with request IDs let you trace a single request across services and correlate with metrics.

4. **Alert on symptoms, not causes**: Alert on user-facing impact (latency, errors) rather than internal metrics (CPU usage) unless they predict problems.

5. **Page for incidents, ticket for anomalies**: Not every alert needs to wake someone up. Reserve pages for active user impact.

6. **Dashboards answer questions**: Design for incident response. What's broken? Since when? How bad? What changed?

---

## Next Steps

Your API is now observable: metrics exposed, logs structured, alerts configured. But how do you detect when the model's predictions are degrading due to changing input data?

Run `/start-4-3` to learn about data drift detection and model retraining triggers.
