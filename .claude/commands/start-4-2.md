# Lesson 4.2: Monitoring & Observability

Read the lesson content from `lesson-modules/4-production/4-2-monitoring.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"Your model is deployed. Users are calling it. Is it working? How would you know if it isn't?"

### 2. Socratic Question
Ask: "If your model starts returning wrong predictions, how long until you notice?"

Expected: Without monitoring, maybe never—or when users complain. Guide them to understand observability is about reducing that detection time.

### 3. The Three Pillars of Observability (10 min)
Cover each pillar:

**Metrics**
- Numeric measurements over time
- Request count, latency, error rate
- Model-specific: prediction distribution, confidence scores

**Logs**
- Discrete events with context
- Request details, errors, model inputs/outputs
- Structured logging for searchability

**Traces**
- Request flow through systems
- Identify bottlenecks
- Distributed system debugging

For ML, add a fourth:
**Model Performance Metrics**
- Accuracy over time (if ground truth available)
- Prediction distribution shifts
- Feature value distributions

### 4. Add Logging to the API (15 min)
Update `api/main.py`:
```python
import logging
import time
from fastapi import FastAPI, Request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"method={request.method} path={request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )
    return response

@app.post("/predict")
def predict(request: PredictionRequest):
    logger.info(f"Prediction request: text_length={len(request.text)}")

    prediction = model.predict([request.text])[0]
    confidence = model.predict_proba([request.text]).max()

    logger.info(f"Prediction result: sentiment={prediction} confidence={confidence:.3f}")

    return PredictionResponse(...)
```

### 5. Add Metrics with Prometheus (15 min)
Install and configure:
```bash
pip install prometheus-client
```

Add to `api/main.py`:
```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['sentiment']
)
REQUEST_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency'
)
CONFIDENCE_HISTOGRAM = Histogram(
    'prediction_confidence',
    'Model confidence distribution',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()

@app.post("/predict")
def predict(request: PredictionRequest):
    with REQUEST_LATENCY.time():
        prediction = model.predict([request.text])[0]
        confidence = model.predict_proba([request.text]).max()

    REQUEST_COUNT.labels(sentiment=prediction).inc()
    CONFIDENCE_HISTOGRAM.observe(confidence)

    return PredictionResponse(...)
```

### 6. Alerting Strategy (10 min)
Discuss what to alert on:

**Immediate alerts (page someone)**
- Error rate > 5%
- Latency p99 > 2 seconds
- Service unreachable

**Warning alerts (investigate soon)**
- Confidence scores dropping
- Prediction distribution changing
- Request volume anomalies

**Informational (dashboard only)**
- Request counts
- Model version in use

Ask: "What's the cost of alerting too much vs too little?"

### 7. Quick Dashboard (10 min)
Show basic visualization options:
- Cloud provider dashboards (Cloud Run metrics, CloudWatch)
- Grafana for custom dashboards
- Simple logging aggregation (CloudWatch Logs, Stackdriver)

If time, set up a basic Cloud Run dashboard showing:
- Request count
- Latency percentiles
- Error rate

### 8. Wrap Up
- Observability is how you know your model is working
- Log everything, but structure it for search
- Metrics enable alerting and dashboards
- Preview: Lesson 4.3 uses these signals to detect model drift
- Next: `/start-4-3`

## Teaching Notes
- Monitoring is often neglected—emphasize it's not optional
- Start simple: logs first, then metrics
- Connect to their experience with broken production systems
- Don't over-engineer: match monitoring to system complexity
