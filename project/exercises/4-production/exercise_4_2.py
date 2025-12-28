"""
Exercise 4.2: Monitoring & Observability
Difficulty: *** (Production-level debugging)
Topic: Prometheus metrics, memory leaks, latency tracking, alerting rules

Instructions:
Production monitoring is critical for ML systems. A model can be "up" (returning 200s)
while silently giving garbage predictions. This exercise covers real monitoring bugs
and implementation patterns.

Part 1: Fix a Prometheus metrics memory leak (DEBUG)
Part 2: Implement latency tracking decorator (WRITE & VERIFY)
Part 3: Create AlertRule class for monitoring thresholds (WRITE & VERIFY)

Hints available: Type /hint 1, /hint 2, /hint 3 for progressive help
"""

import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Callable, Any
from collections import defaultdict


# =============================================================================
# PART 1: DEBUG - Fix Prometheus Memory Leak
# =============================================================================
#
# INCIDENT REPORT:
# """
# Service: sentiment-api-prod
# Severity: P1 (Production Impact)
# Duration: 6 hours before detection
#
# Symptoms:
# - Memory usage grew from 512MB to 8GB over 6 hours
# - OOMKilled by Kubernetes
# - Service restarted, then same pattern repeated
#
# Timeline:
# - 09:00: Deploy new version with "improved metrics"
# - 09:15: Memory at 600MB (normal)
# - 12:00: Memory at 2.1GB (concerning)
# - 15:00: Memory at 6.4GB (critical)
# - 15:23: OOMKilled, pod restarted
#
# Investigation:
# - Heap dump showed millions of Prometheus metric instances
# - Each unique (user_id, request_id) combination created new time series
#
# Root cause: Unbounded label cardinality in Prometheus metrics
# """
#
# Below is the code that caused the incident. Your task is to identify and fix it.

class BrokenMetricsCollector:
    """
    This metrics collector has a critical bug that causes memory leaks.

    DO NOT USE THIS IN PRODUCTION - it's here to demonstrate the bug.
    """

    def __init__(self):
        # Simulating prometheus_client.Counter behavior
        self._counters: dict[tuple, int] = defaultdict(int)
        self._histograms: dict[tuple, list] = defaultdict(list)

    def record_prediction(
        self,
        user_id: str,
        request_id: str,
        model_version: str,
        predicted_label: str,
        confidence: float,
        latency_ms: float,
    ):
        """Record metrics for a prediction. THIS CODE HAS A BUG."""
        # BUG: Using user_id and request_id as labels!
        # Each unique combination creates a new time series
        # With 1000 users making 100 requests each = 100,000 time series
        # Prometheus keeps all time series in memory = OOM

        # Counter: predictions by user, request, model, label
        counter_key = (user_id, request_id, model_version, predicted_label)
        self._counters[counter_key] += 1

        # Histogram: latency by user
        histogram_key = (user_id, model_version)
        self._histograms[histogram_key].append(latency_ms)

        # This code "works" but memory grows unboundedly!

    def get_memory_usage(self) -> dict:
        """Return current metrics memory state (for debugging)."""
        return {
            "counter_series": len(self._counters),
            "histogram_series": len(self._histograms),
            "total_series": len(self._counters) + len(self._histograms),
        }


def demonstrate_memory_leak():
    """Show how the memory leak manifests."""
    collector = BrokenMetricsCollector()

    # Simulate 1 hour of production traffic
    # 100 users, 50 requests per user
    import random
    import uuid

    for _ in range(100):  # 100 users
        user_id = f"user_{uuid.uuid4()}"
        for _ in range(50):  # 50 requests per user
            collector.record_prediction(
                user_id=user_id,
                request_id=str(uuid.uuid4()),
                model_version="1.0.0",
                predicted_label=random.choice(["positive", "negative", "neutral"]),
                confidence=random.random(),
                latency_ms=random.uniform(10, 100),
            )

    stats = collector.get_memory_usage()
    print(f"After 1 hour simulation:")
    print(f"  Counter series: {stats['counter_series']:,}")
    print(f"  Histogram series: {stats['histogram_series']:,}")
    print(f"  Total series: {stats['total_series']:,}")
    print(f"  (In production, each series uses ~1-3KB of memory)")


class FixedMetricsCollector:
    """
    YOUR TASK: Implement a fixed version that doesn't leak memory.

    Requirements:
    1. Track predictions by model_version and predicted_label (bounded!)
    2. Track latency distribution (don't store per-user)
    3. DO NOT use user_id or request_id as labels
    4. Still record useful metrics for debugging

    Bonus: Consider what information IS useful to track:
    - Total predictions by label (helps detect drift)
    - Latency percentiles (helps detect performance issues)
    - Error counts (helps detect bugs)
    """

    def __init__(self):
        # TODO: Initialize your metrics storage
        # Remember: labels must be BOUNDED (known, small set of values)
        self._counters: dict[tuple, int] = defaultdict(int)
        self._latencies: list[float] = []

    def record_prediction(
        self,
        user_id: str,  # DO NOT use as label!
        request_id: str,  # DO NOT use as label!
        model_version: str,  # OK to use (bounded)
        predicted_label: str,  # OK to use (bounded)
        confidence: float,
        latency_ms: float,
    ):
        """
        Record metrics for a prediction.

        Args:
            user_id: Unique user identifier (log it, don't label with it)
            request_id: Unique request ID (log it, don't label with it)
            model_version: Model version string (OK as label - bounded)
            predicted_label: Predicted class (OK as label - bounded)
            confidence: Prediction confidence score
            latency_ms: Request latency in milliseconds
        """
        # TODO: Implement proper metrics recording
        # - Use only bounded labels
        # - Consider using a circular buffer for latencies
        # - Log user_id and request_id for debugging, but don't metric them
        pass

    def get_memory_usage(self) -> dict:
        """Return current metrics memory state."""
        return {
            "counter_series": len(self._counters),
            "latency_samples": len(self._latencies),
        }


# =============================================================================
# PART 2: WRITE & VERIFY - Latency Tracking Decorator
# =============================================================================
#
# Create a decorator that automatically tracks function execution time
# and records it as a metric. This is a common pattern for instrumenting ML APIs.


class MetricsRegistry:
    """Simple in-memory metrics registry for the exercise."""

    def __init__(self):
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)

    def record_histogram(self, name: str, value: float):
        """Record a value to a histogram metric."""
        self.histograms[name].append(value)

    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter metric."""
        self.counters[name] += amount

    def get_percentile(self, name: str, percentile: float) -> float | None:
        """Get a percentile value from a histogram."""
        values = self.histograms.get(name, [])
        if not values:
            return None
        import numpy as np
        return float(np.percentile(values, percentile))


# Global registry for the exercise
_registry = MetricsRegistry()


def track_prediction_latency(
    metric_name: str = "prediction_latency_seconds",
    registry: MetricsRegistry | None = None,
) -> Callable:
    """
    Decorator that tracks function execution time.

    Usage:
        @track_prediction_latency()
        def predict(text: str) -> dict:
            # ... model inference ...
            return {"label": "positive", "confidence": 0.95}

        # After calling predict(), the latency is automatically recorded

    Args:
        metric_name: Name for the latency histogram metric
        registry: MetricsRegistry to use (defaults to global)

    Returns:
        Decorator function

    The decorator should:
    1. Record the start time before calling the function
    2. Call the original function
    3. Record the elapsed time to the histogram
    4. Return the original function's result
    5. Handle exceptions (still record latency, then re-raise)
    """
    if registry is None:
        registry = _registry

    def decorator(func: Callable) -> Callable:
        # TODO: Implement the decorator
        # Remember to use functools.wraps to preserve function metadata!

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # TODO: Record timing and call the function
            # Don't forget to handle exceptions!
            pass

        return wrapper

    return decorator


# =============================================================================
# PART 3: WRITE & VERIFY - AlertRule Class
# =============================================================================
#
# Define monitoring thresholds that trigger alerts when exceeded.
# This is how production ML systems know when something is wrong.


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(Enum):
    """Current state of an alert."""
    OK = "ok"
    PENDING = "pending"
    FIRING = "firing"


@dataclass
class AlertRule:
    """
    Defines a monitoring alert rule.

    An alert fires when a metric exceeds a threshold for a duration.

    Attributes:
        name: Human-readable alert name
        description: What this alert means
        metric_name: The metric to monitor
        threshold: Value that triggers the alert
        comparison: How to compare ("gt", "lt", "gte", "lte")
        for_seconds: How long threshold must be exceeded before firing
        severity: Alert severity level

    Example:
        # Alert if p99 latency > 200ms for 5 minutes
        rule = AlertRule(
            name="HighLatency",
            description="99th percentile latency is too high",
            metric_name="prediction_latency_p99",
            threshold=0.2,  # 200ms in seconds
            comparison="gt",
            for_seconds=300,  # 5 minutes
            severity=AlertSeverity.WARNING,
        )
    """
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "gte", "lte"
    for_seconds: int
    severity: AlertSeverity

    # Internal state for tracking
    _first_breach_time: float | None = None
    _state: AlertState = AlertState.OK

    def evaluate(self, current_value: float, current_time: float | None = None) -> AlertState:
        """
        Evaluate the rule against a current metric value.

        Args:
            current_value: The current value of the metric
            current_time: Current timestamp (defaults to time.time())

        Returns:
            AlertState indicating current alert status:
            - OK: Metric is within acceptable range
            - PENDING: Threshold breached but not long enough
            - FIRING: Threshold breached for required duration

        The alert should:
        1. Check if current_value breaches threshold (based on comparison)
        2. If breached, track when breach started
        3. If breached for >= for_seconds, return FIRING
        4. If breach ends, reset tracking and return OK
        """
        if current_time is None:
            current_time = time.time()

        # TODO: Implement the evaluation logic
        # Consider:
        # - How to compare based on self.comparison
        # - When to set/reset _first_breach_time
        # - When to transition between states

        return AlertState.OK  # Placeholder


def create_standard_ml_alerts() -> list[AlertRule]:
    """
    Create a standard set of alerts for an ML API.

    Returns:
        List of AlertRule objects covering common failure modes:
        1. HighLatency: p99 latency > 500ms for 5 minutes
        2. HighErrorRate: error rate > 5% for 2 minutes
        3. LowConfidence: median confidence < 0.6 for 30 minutes
        4. PredictionSkew: any class > 80% of predictions for 1 hour
        5. NoTraffic: 0 requests for 10 minutes
    """
    # TODO: Create and return the list of AlertRule objects
    alerts = []

    # Example:
    # alerts.append(AlertRule(
    #     name="HighLatency",
    #     description="P99 prediction latency exceeds 500ms",
    #     metric_name="prediction_latency_p99_seconds",
    #     threshold=0.5,
    #     comparison="gt",
    #     for_seconds=300,
    #     severity=AlertSeverity.WARNING,
    # ))

    return alerts


# =============================================================================
# HINTS (Don't peek until you've tried!)
# =============================================================================

"""
=== HINT 1 (Part 1 - Memory Leak) ===

The key insight is LABEL CARDINALITY. In Prometheus:
- Each unique combination of labels creates a new "time series"
- Time series are stored in memory
- user_id and request_id are HIGH CARDINALITY (millions of values)

Bad (unbounded):
  predictions_total{user_id="abc123", request_id="xyz789"}
  predictions_total{user_id="abc124", request_id="xyz790"}
  ... millions of series ...

Good (bounded):
  predictions_total{model_version="1.0.0", label="positive"}
  predictions_total{model_version="1.0.0", label="negative"}
  predictions_total{model_version="1.0.0", label="neutral"}
  ... only ~10 series ...

For the fixed implementation:
```python
def record_prediction(self, user_id, request_id, model_version, predicted_label, confidence, latency_ms):
    # GOOD: Only use bounded labels
    counter_key = (model_version, predicted_label)
    self._counters[counter_key] += 1

    # GOOD: Single latency distribution, not per-user
    # Use a circular buffer to limit memory
    self._latencies.append(latency_ms)
    if len(self._latencies) > 10000:  # Keep last 10k samples
        self._latencies = self._latencies[-10000:]

    # For debugging: LOG the request_id, don't METRIC it
    # logger.info("prediction", request_id=request_id, user_id=user_id)
```


=== HINT 2 (Part 2 - Latency Decorator) ===

The decorator pattern with timing:

```python
def track_prediction_latency(metric_name="prediction_latency_seconds", registry=None):
    if registry is None:
        registry = _registry

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Always record latency, even on exception
                elapsed = time.perf_counter() - start_time
                registry.record_histogram(metric_name, elapsed)

        return wrapper
    return decorator
```

Key points:
- Use time.perf_counter() for accurate timing (not time.time())
- Use try/finally to always record latency
- Use @wraps to preserve function metadata
- Re-raise exceptions after recording


=== HINT 3 (Part 3 - AlertRule) ===

The evaluate method needs to handle state transitions:

```python
def evaluate(self, current_value: float, current_time: float = None) -> AlertState:
    if current_time is None:
        current_time = time.time()

    # Check if threshold is breached
    is_breached = False
    if self.comparison == "gt":
        is_breached = current_value > self.threshold
    elif self.comparison == "lt":
        is_breached = current_value < self.threshold
    elif self.comparison == "gte":
        is_breached = current_value >= self.threshold
    elif self.comparison == "lte":
        is_breached = current_value <= self.threshold

    if is_breached:
        # Start tracking breach time if not already
        if self._first_breach_time is None:
            self._first_breach_time = current_time
            self._state = AlertState.PENDING
            return AlertState.PENDING

        # Check if breach duration exceeded
        breach_duration = current_time - self._first_breach_time
        if breach_duration >= self.for_seconds:
            self._state = AlertState.FIRING
            return AlertState.FIRING
        else:
            self._state = AlertState.PENDING
            return AlertState.PENDING
    else:
        # Not breached - reset
        self._first_breach_time = None
        self._state = AlertState.OK
        return AlertState.OK
```

For create_standard_ml_alerts:

```python
def create_standard_ml_alerts() -> list[AlertRule]:
    return [
        AlertRule(
            name="HighLatency",
            description="P99 prediction latency exceeds 500ms",
            metric_name="prediction_latency_p99_seconds",
            threshold=0.5,
            comparison="gt",
            for_seconds=300,
            severity=AlertSeverity.WARNING,
        ),
        AlertRule(
            name="HighErrorRate",
            description="Error rate exceeds 5%",
            metric_name="error_rate",
            threshold=0.05,
            comparison="gt",
            for_seconds=120,
            severity=AlertSeverity.CRITICAL,
        ),
        # ... etc
    ]
```
"""
