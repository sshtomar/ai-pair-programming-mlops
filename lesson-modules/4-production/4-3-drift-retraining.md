# Lesson 4.3: Model Drift & Retraining

**Duration:** 45 minutes

**Prerequisites:** Lesson 4.2 (Production Monitoring)

## Learning Objectives

By the end of this lesson, you will:
1. Distinguish between data drift, concept drift, and label drift
2. Implement statistical drift detection for your sentiment classifier
3. Design retraining triggers and automated workflows

---

## Why Models Degrade

Your model was trained on historical data. Production data evolves. This mismatch is **drift**.

Consider your sentiment classifier:
- New slang emerges ("slay", "mid", "goated")
- Global events shift sentiment patterns (pandemic, economic crisis)
- Your user base changes (new demographics, markets)
- Upstream systems change (different text preprocessing, new data sources)

A model that worked well at deployment will silently degrade without drift detection.

---

## The Three Types of Drift

### 1. Data Drift (Covariate Shift)

**What:** Input distribution P(X) changes, but P(Y|X) stays the same.

**Example:** Your classifier trained on formal product reviews now receives casual social media text. The relationship between words and sentiment hasn't changed—but the vocabulary has shifted.

```
Training data:  "The product quality is exceptional"
Production data: "this thing slaps fr fr no cap"
```

The sentiment-to-text relationship is preserved (positive is positive), but your model may not recognize the new vocabulary.

### 2. Concept Drift

**What:** The relationship P(Y|X) changes, even if P(X) stays the same.

**Example:** The word "sick" traditionally meant negative (illness). In some contexts, it now means positive (impressive). Same input distribution, different meaning.

```python
# Before: P("sick" → negative) = 0.9
# After:  P("sick" → negative) = 0.4
```

Concept drift is harder to detect because you need ground truth labels.

### 3. Label Drift (Prior Probability Shift)

**What:** Output distribution P(Y) changes.

**Example:** During a product crisis, negative reviews spike from 20% to 60%. Your model, calibrated for the original distribution, may be overconfident on positive predictions.

```
Training:   60% positive, 40% negative
Production: 30% positive, 70% negative
```

### Quick Reference

| Drift Type | What Changes | Detection Difficulty | Requires Labels |
|------------|--------------|---------------------|-----------------|
| Data Drift | P(X) | Easy | No |
| Concept Drift | P(Y\|X) | Hard | Yes |
| Label Drift | P(Y) | Medium | Yes |

### Visual: Detecting Distribution Shift

```
    Training Distribution              Production Distribution
    ═════════════════════              ══════════════════════

            ▁▂▄▆█▆▄▂▁                        ▁▂▃▅▇█▇▅▃▂▁
          ╱            ╲                          ╱    ╲
         ╱              ╲                        ╱      ╲
    ────╱────────────────╲────          ────────╱────────╲────────
        │       ↑        │                      │    ↑   │
        │    center      │                      │  SHIFT │
        │                │                      │        │

    ┌───────────────────────────────────────────────────────────────┐
    │  ⚠️  DRIFT DETECTED                                           │
    │                                                               │
    │  • KS statistic: 0.23 (threshold: 0.10)                       │
    │  • PSI score: 0.31 (threshold: 0.20)                          │
    │                                                               │
    │  Action: Trigger retraining pipeline                          │
    └───────────────────────────────────────────────────────────────┘
```

---

## Detection Methods

### Statistical Tests for Numerical Features

**Kolmogorov-Smirnov (KS) Test**

Compares two distributions by measuring the maximum distance between their cumulative distribution functions.

```python
from scipy import stats
import numpy as np

def detect_drift_ks(reference: np.ndarray,
                    current: np.ndarray,
                    threshold: float = 0.05) -> dict:
    """
    Detect drift using two-sample KS test.

    Args:
        reference: Feature values from training/reference period
        current: Feature values from current production window
        threshold: p-value threshold for significance

    Returns:
        dict with statistic, p_value, and drift_detected
    """
    statistic, p_value = stats.ks_2samp(reference, current)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < threshold
    }

# Example: Check if text length distribution shifted
reference_lengths = np.array([45, 52, 38, 61, 55, 42, 48, 50, 44, 47])
current_lengths = np.array([23, 18, 31, 25, 19, 28, 22, 20, 26, 24])

result = detect_drift_ks(reference_lengths, current_lengths)
print(f"KS Statistic: {result['statistic']:.3f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Drift detected: {result['drift_detected']}")
```

**Population Stability Index (PSI)**

Measures how much a distribution has shifted. Commonly used in finance for model monitoring.

```python
import numpy as np

def calculate_psi(reference: np.ndarray,
                  current: np.ndarray,
                  bins: int = 10) -> float:
    """
    Calculate Population Stability Index.

    PSI < 0.1: No significant shift
    PSI 0.1-0.25: Moderate shift (investigate)
    PSI > 0.25: Significant shift (action required)
    """
    # Create bins from reference distribution
    _, bin_edges = np.histogram(reference, bins=bins)

    # Calculate proportions in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions (add small epsilon to avoid log(0))
    epsilon = 1e-10
    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * bins)
    curr_pct = (curr_counts + epsilon) / (len(current) + epsilon * bins)

    # PSI formula: sum((curr - ref) * ln(curr/ref))
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

    return psi

# Example
psi = calculate_psi(reference_lengths, current_lengths)
print(f"PSI: {psi:.3f}")
if psi < 0.1:
    print("No significant drift")
elif psi < 0.25:
    print("Moderate drift - investigate")
else:
    print("Significant drift - action required")
```

### Drift Detection for Text Data

Text requires special handling. You can't directly apply KS tests to raw text.

**Approach 1: Vocabulary Shift**

Track out-of-vocabulary (OOV) rate—words not seen during training.

```python
from typing import Set
from collections import Counter

class VocabularyDriftDetector:
    def __init__(self, training_vocab: Set[str], oov_threshold: float = 0.15):
        """
        Args:
            training_vocab: Set of words seen during training
            oov_threshold: Alert if OOV rate exceeds this
        """
        self.training_vocab = training_vocab
        self.oov_threshold = oov_threshold

    def check_drift(self, texts: list[str]) -> dict:
        """Check vocabulary drift in a batch of texts."""
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())

        word_counts = Counter(all_words)
        total_words = sum(word_counts.values())

        oov_words = {w: c for w, c in word_counts.items()
                     if w not in self.training_vocab}
        oov_count = sum(oov_words.values())
        oov_rate = oov_count / total_words if total_words > 0 else 0

        # Top OOV words for investigation
        top_oov = sorted(oov_words.items(), key=lambda x: -x[1])[:10]

        return {
            "oov_rate": oov_rate,
            "drift_detected": oov_rate > self.oov_threshold,
            "top_oov_words": top_oov,
            "total_unique_oov": len(oov_words)
        }

# Example usage
training_vocab = {"good", "bad", "great", "terrible", "product", "quality", "love", "hate"}
detector = VocabularyDriftDetector(training_vocab, oov_threshold=0.2)

production_texts = [
    "this product is absolutely goated",
    "mid quality tbh not worth it",
    "lowkey slaps would recommend"
]

result = detector.check_drift(production_texts)
print(f"OOV Rate: {result['oov_rate']:.1%}")
print(f"New words: {result['top_oov_words']}")
```

**Approach 2: Embedding Drift**

Compare embedding distributions between reference and current data.

```python
import numpy as np
from scipy.spatial.distance import cosine

class EmbeddingDriftDetector:
    def __init__(self, reference_embeddings: np.ndarray):
        """
        Args:
            reference_embeddings: (n_samples, embedding_dim) from training data
        """
        self.reference_centroid = np.mean(reference_embeddings, axis=0)
        self.reference_std = np.std(reference_embeddings, axis=0)

    def check_drift(self,
                    current_embeddings: np.ndarray,
                    cosine_threshold: float = 0.1,
                    std_threshold: float = 2.0) -> dict:
        """
        Detect drift via centroid shift and spread change.

        Args:
            current_embeddings: (n_samples, embedding_dim) from production
            cosine_threshold: Max acceptable cosine distance from reference centroid
            std_threshold: Max z-score for std deviation change
        """
        current_centroid = np.mean(current_embeddings, axis=0)
        current_std = np.std(current_embeddings, axis=0)

        # Centroid shift (cosine distance)
        centroid_distance = cosine(self.reference_centroid, current_centroid)

        # Spread change (how much std deviated)
        std_change = np.mean(np.abs(current_std - self.reference_std) / (self.reference_std + 1e-10))

        return {
            "centroid_distance": centroid_distance,
            "std_change": std_change,
            "drift_detected": (centroid_distance > cosine_threshold or
                              std_change > std_threshold)
        }
```

**Approach 3: Classifier Two-Sample Test**

Train a classifier to distinguish reference from current data. If it succeeds (AUC > 0.5), drift exists.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def classifier_drift_test(reference_embeddings: np.ndarray,
                          current_embeddings: np.ndarray,
                          auc_threshold: float = 0.6) -> dict:
    """
    Train classifier to distinguish reference vs current.
    High AUC = easy to distinguish = drift detected.
    """
    # Create labels: 0 = reference, 1 = current
    X = np.vstack([reference_embeddings, current_embeddings])
    y = np.array([0] * len(reference_embeddings) + [1] * len(current_embeddings))

    # Cross-validated AUC
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    mean_auc = scores.mean()

    return {
        "auc": mean_auc,
        "auc_std": scores.std(),
        "drift_detected": mean_auc > auc_threshold
    }
```

---

## Reference Windows: What to Compare Against

You need a baseline to detect drift. Two approaches:

### Fixed Reference Window

Compare against a static snapshot (e.g., training data distribution).

```python
class FixedReferenceMonitor:
    def __init__(self, training_data_stats: dict):
        """Stats computed once from training data."""
        self.reference = training_data_stats

    def check(self, current_stats: dict) -> dict:
        # Compare current against fixed reference
        pass
```

**Pros:** Detects total drift from training. Clear threshold.
**Cons:** Gradual drift accumulates. May over-alert as time passes.

### Sliding Reference Window

Compare recent data against slightly older data.

```python
class SlidingWindowMonitor:
    def __init__(self, window_size: int = 7, comparison_gap: int = 7):
        """
        Args:
            window_size: Days of data in each window
            comparison_gap: Days between windows

        Example: window_size=7, gap=7 compares last week vs week before
        """
        self.window_size = window_size
        self.comparison_gap = comparison_gap
        self.data_buffer = []

    def add_data(self, data_point: dict):
        self.data_buffer.append(data_point)
        # Keep only what we need
        max_needed = self.window_size + self.comparison_gap
        if len(self.data_buffer) > max_needed:
            self.data_buffer = self.data_buffer[-max_needed:]

    def check(self) -> dict | None:
        if len(self.data_buffer) < self.window_size + self.comparison_gap:
            return None  # Not enough data yet

        current_window = self.data_buffer[-self.window_size:]
        reference_window = self.data_buffer[-(self.window_size + self.comparison_gap):-self.comparison_gap]

        # Compare windows
        pass
```

**Pros:** Detects sudden shifts. Adapts to gradual change.
**Cons:** May miss slow drift. Requires defining "normal" rate of change.

### Recommended: Use Both

```python
class DualWindowMonitor:
    """Monitor against both training baseline and recent history."""

    def __init__(self, training_stats: dict, window_size: int = 7):
        self.training_baseline = training_stats
        self.sliding_monitor = SlidingWindowMonitor(window_size)

    def check(self, current_stats: dict) -> dict:
        return {
            "vs_training": self.compare(current_stats, self.training_baseline),
            "vs_recent": self.sliding_monitor.check()
        }
```

---

## When to Retrain

### Scheduled Retraining

Retrain on a fixed schedule regardless of drift detection.

```yaml
# Example: GitHub Actions scheduled retraining
name: Weekly Model Retraining
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2 AM

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - name: Fetch latest data
        run: python scripts/fetch_training_data.py --days 90

      - name: Retrain model
        run: python train.py --config config/production.yaml

      - name: Evaluate
        run: python evaluate.py --model latest --test-set holdout

      - name: Register if improved
        run: python scripts/conditional_register.py
```

**When to use scheduled retraining:**
- Data changes predictably (daily/weekly patterns)
- Labeling pipeline produces regular batches
- Compliance requires periodic model updates
- Low-stakes application where drift detection overhead isn't worth it

### Triggered Retraining

Retrain when drift exceeds thresholds.

```python
from dataclasses import dataclass
from enum import Enum

class RetrainingAction(Enum):
    NONE = "none"
    INVESTIGATE = "investigate"
    RETRAIN = "retrain"
    EMERGENCY = "emergency"

@dataclass
class DriftThresholds:
    """Thresholds for different actions."""
    investigate_psi: float = 0.1
    retrain_psi: float = 0.25
    emergency_psi: float = 0.5
    investigate_accuracy_drop: float = 0.02
    retrain_accuracy_drop: float = 0.05
    emergency_accuracy_drop: float = 0.10

def determine_action(drift_metrics: dict,
                     thresholds: DriftThresholds) -> RetrainingAction:
    """
    Decide action based on drift severity.

    Args:
        drift_metrics: Dict with 'psi', 'accuracy_drop', etc.
        thresholds: Action thresholds

    Returns:
        Recommended action
    """
    psi = drift_metrics.get("psi", 0)
    accuracy_drop = drift_metrics.get("accuracy_drop", 0)

    # Emergency: Major degradation
    if psi > thresholds.emergency_psi or accuracy_drop > thresholds.emergency_accuracy_drop:
        return RetrainingAction.EMERGENCY

    # Retrain: Significant drift
    if psi > thresholds.retrain_psi or accuracy_drop > thresholds.retrain_accuracy_drop:
        return RetrainingAction.RETRAIN

    # Investigate: Early warning
    if psi > thresholds.investigate_psi or accuracy_drop > thresholds.investigate_accuracy_drop:
        return RetrainingAction.INVESTIGATE

    return RetrainingAction.NONE
```

### Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    DRIFT DETECTED                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Do you have ground truth     │
            │  labels for recent data?      │
            └───────────────────────────────┘
                    │               │
                   Yes              No
                    │               │
                    ▼               ▼
        ┌───────────────┐   ┌───────────────────┐
        │ Measure actual│   │ Only data drift   │
        │ performance   │   │ detected          │
        └───────────────┘   └───────────────────┘
                │                   │
                ▼                   ▼
    ┌─────────────────────┐  ┌────────────────────────┐
    │ Performance dropped │  │ Start labeling sample  │
    │ significantly?      │  │ to measure real impact │
    └─────────────────────┘  └────────────────────────┘
        │           │               │
       Yes          No              ▼
        │           │        ┌────────────────────┐
        ▼           ▼        │ Meanwhile: monitor │
    ┌───────┐   ┌───────┐    │ closely, don't     │
    │RETRAIN│   │MONITOR│    │ auto-retrain       │
    └───────┘   └───────┘    └────────────────────┘
```

---

## Safe Deployment of Retrained Models

### Shadow Mode

Run new model in parallel without serving its predictions.

```python
from typing import Any
import logging

logger = logging.getLogger(__name__)

class ShadowDeployment:
    """Run new model alongside production for validation."""

    def __init__(self, production_model: Any, shadow_model: Any):
        self.production = production_model
        self.shadow = shadow_model
        self.comparison_log = []

    def predict(self, input_data: dict) -> dict:
        # Production prediction (what user sees)
        prod_result = self.production.predict(input_data)

        # Shadow prediction (logged only)
        try:
            shadow_result = self.shadow.predict(input_data)

            # Log comparison
            self.comparison_log.append({
                "input_hash": hash(str(input_data)),
                "production": prod_result,
                "shadow": shadow_result,
                "agreement": prod_result["label"] == shadow_result["label"]
            })

            # Alert on major disagreements
            if len(self.comparison_log) >= 100:
                self._analyze_shadow_performance()

        except Exception as e:
            logger.error(f"Shadow model failed: {e}")

        # Always return production result
        return prod_result

    def _analyze_shadow_performance(self):
        agreement_rate = sum(1 for x in self.comparison_log if x["agreement"]) / len(self.comparison_log)
        logger.info(f"Shadow agreement rate: {agreement_rate:.1%}")

        if agreement_rate < 0.8:
            logger.warning("Shadow model differs significantly from production")
```

### Canary Deployment

Gradually route traffic to new model.

```python
import random
from datetime import datetime

class CanaryRouter:
    """Route percentage of traffic to canary model."""

    def __init__(self,
                 production_model: Any,
                 canary_model: Any,
                 canary_percentage: float = 5.0):
        self.production = production_model
        self.canary = canary_model
        self.canary_percentage = canary_percentage
        self.canary_metrics = {"requests": 0, "errors": 0, "latencies": []}
        self.production_metrics = {"requests": 0, "errors": 0, "latencies": []}

    def predict(self, input_data: dict) -> dict:
        use_canary = random.random() * 100 < self.canary_percentage

        start = datetime.now()
        try:
            if use_canary:
                result = self.canary.predict(input_data)
                result["model_version"] = "canary"
                self.canary_metrics["requests"] += 1
                self.canary_metrics["latencies"].append((datetime.now() - start).total_seconds())
            else:
                result = self.production.predict(input_data)
                result["model_version"] = "production"
                self.production_metrics["requests"] += 1
                self.production_metrics["latencies"].append((datetime.now() - start).total_seconds())

            return result

        except Exception as e:
            if use_canary:
                self.canary_metrics["errors"] += 1
                # Fallback to production on canary failure
                return self.production.predict(input_data)
            raise

    def should_promote_canary(self,
                              min_requests: int = 1000,
                              max_error_rate: float = 0.01,
                              max_latency_increase: float = 0.2) -> bool:
        """Check if canary is ready for full promotion."""
        if self.canary_metrics["requests"] < min_requests:
            return False  # Not enough data

        canary_error_rate = self.canary_metrics["errors"] / self.canary_metrics["requests"]
        if canary_error_rate > max_error_rate:
            return False

        import numpy as np
        canary_p50 = np.percentile(self.canary_metrics["latencies"], 50)
        prod_p50 = np.percentile(self.production_metrics["latencies"], 50)

        if canary_p50 > prod_p50 * (1 + max_latency_increase):
            return False

        return True
```

### A/B Testing

For models where business metrics matter, run controlled experiments.

```python
import hashlib
from typing import Literal

class ABTestRouter:
    """Deterministic A/B assignment for model comparison."""

    def __init__(self,
                 model_a: Any,
                 model_b: Any,
                 experiment_id: str,
                 split_ratio: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.experiment_id = experiment_id
        self.split_ratio = split_ratio

    def get_variant(self, user_id: str) -> Literal["A", "B"]:
        """Deterministic assignment based on user_id."""
        hash_input = f"{self.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000
        return "A" if normalized < self.split_ratio else "B"

    def predict(self, input_data: dict, user_id: str) -> dict:
        variant = self.get_variant(user_id)

        if variant == "A":
            result = self.model_a.predict(input_data)
        else:
            result = self.model_b.predict(input_data)

        result["experiment_variant"] = variant
        return result
```

---

## Automated Retraining Pipeline

Complete pipeline for drift-triggered retraining:

```python
# scripts/retraining_pipeline.py

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrainingConfig:
    min_samples: int = 10000
    min_days_since_last_training: int = 7
    psi_threshold: float = 0.25
    accuracy_drop_threshold: float = 0.05
    test_set_path: str = "data/test_holdout.csv"
    min_improvement: float = 0.01

class RetrainingPipeline:
    def __init__(self, config: RetrainingConfig):
        self.config = config

    def should_retrain(self, drift_report: dict, last_training: datetime) -> tuple[bool, str]:
        """Determine if retraining should proceed."""

        # Check cooldown
        days_since_training = (datetime.now() - last_training).days
        if days_since_training < self.config.min_days_since_last_training:
            return False, f"Cooldown: only {days_since_training} days since last training"

        # Check data availability
        available_samples = drift_report.get("new_labeled_samples", 0)
        if available_samples < self.config.min_samples:
            return False, f"Insufficient data: {available_samples} < {self.config.min_samples}"

        # Check drift severity
        psi = drift_report.get("psi", 0)
        accuracy_drop = drift_report.get("accuracy_drop", 0)

        if psi > self.config.psi_threshold:
            return True, f"PSI threshold exceeded: {psi:.3f}"

        if accuracy_drop > self.config.accuracy_drop_threshold:
            return True, f"Accuracy drop: {accuracy_drop:.1%}"

        return False, "No significant drift detected"

    def run(self, drift_report: dict, last_training: datetime) -> dict:
        """Execute retraining pipeline."""

        # Gate check
        should_proceed, reason = self.should_retrain(drift_report, last_training)
        if not should_proceed:
            logger.info(f"Skipping retraining: {reason}")
            return {"action": "skipped", "reason": reason}

        logger.info(f"Starting retraining: {reason}")

        # Step 1: Fetch and prepare data
        training_data = self._fetch_training_data()

        # Step 2: Train new model
        new_model = self._train_model(training_data)

        # Step 3: Evaluate on holdout test set
        current_metrics = self._evaluate_current_model()
        new_metrics = self._evaluate_model(new_model)

        improvement = new_metrics["accuracy"] - current_metrics["accuracy"]

        # Step 4: Decide on promotion
        if improvement < self.config.min_improvement:
            logger.warning(f"New model not better: {improvement:+.3f} accuracy change")
            return {
                "action": "rejected",
                "reason": f"Insufficient improvement: {improvement:+.3f}",
                "current_metrics": current_metrics,
                "new_metrics": new_metrics
            }

        # Step 5: Register and deploy
        model_version = self._register_model(new_model, new_metrics)
        self._deploy_to_shadow(model_version)

        return {
            "action": "deployed_shadow",
            "model_version": model_version,
            "improvement": improvement,
            "current_metrics": current_metrics,
            "new_metrics": new_metrics
        }

    def _fetch_training_data(self):
        """Fetch recent labeled data for retraining."""
        # Implementation: query data warehouse, apply preprocessing
        pass

    def _train_model(self, data):
        """Train new model with same config as production."""
        # Implementation: use same training script/config
        pass

    def _evaluate_current_model(self) -> dict:
        """Evaluate current production model on holdout set."""
        pass

    def _evaluate_model(self, model) -> dict:
        """Evaluate given model on holdout set."""
        pass

    def _register_model(self, model, metrics) -> str:
        """Register model in MLflow registry."""
        pass

    def _deploy_to_shadow(self, model_version: str):
        """Deploy new model in shadow mode."""
        pass
```

---

## Exercises

### Exercise 1: Implement Drift Detection (15 min)

Add drift detection to your monitoring system:

```python
# project/src/monitoring/drift.py

"""
TODO: Implement a DriftDetector class that:
1. Stores reference statistics from training data
2. Computes PSI for numerical features (text length, confidence)
3. Tracks vocabulary OOV rate
4. Returns drift report with recommendations
"""

class DriftDetector:
    def __init__(self, reference_data: dict):
        # Store reference distributions
        pass

    def compute_drift(self, current_batch: list[dict]) -> dict:
        # Compare current batch against reference
        # Return: {"psi": float, "oov_rate": float, "drift_detected": bool}
        pass
```

Test your implementation:

```bash
cd project
python -c "
from src.monitoring.drift import DriftDetector

# Simulate reference data (training distribution)
reference = {
    'text_lengths': [50, 55, 48, 62, 45, 58, 52, 49, 61, 54],
    'vocabulary': {'good', 'bad', 'product', 'quality', 'great', 'terrible'}
}

detector = DriftDetector(reference)

# Simulate production data with drift
drifted_batch = [
    {'text': 'this slaps fr', 'length': 13},
    {'text': 'mid tbh', 'length': 7},
    {'text': 'lowkey goated', 'length': 13}
]

report = detector.compute_drift(drifted_batch)
print(f'Drift report: {report}')
"
```

### Exercise 2: Retraining Trigger Logic (10 min)

Implement the decision logic for when to retrain:

```python
# project/src/monitoring/retrain_trigger.py

"""
TODO: Create a RetrainingTrigger that:
1. Checks drift thresholds
2. Checks cooldown period
3. Checks data availability
4. Returns action recommendation with reason
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class Action(Enum):
    NONE = "none"
    INVESTIGATE = "investigate"
    RETRAIN = "retrain"

@dataclass
class TriggerResult:
    action: Action
    reason: str
    urgency: str  # "low", "medium", "high"

def evaluate_retraining(
    drift_metrics: dict,
    last_training_date: datetime,
    available_samples: int
) -> TriggerResult:
    # Implement decision logic
    pass
```

### Exercise 3: Shadow Deployment (10 min)

Add shadow mode to your FastAPI service:

```python
# project/src/api/shadow.py

"""
TODO: Modify your /predict endpoint to:
1. Run predictions through both production and shadow model
2. Log disagreements
3. Return only production result to user
4. Track shadow model accuracy if ground truth becomes available
"""
```

---

## Key Takeaways

1. **Three types of drift** exist: data drift (inputs change), concept drift (input-output relationship changes), label drift (outputs change). Each requires different detection approaches.

2. **Detection methods** range from simple (PSI, KS test) to sophisticated (classifier two-sample test). For text, track vocabulary shift and embedding drift.

3. **Use dual reference windows**: compare against both training baseline (total drift) and recent history (sudden changes).

4. **Don't auto-retrain on data drift alone**. Data drift without performance degradation may not require action. Get ground truth before deciding.

5. **Safe deployment is essential**: use shadow mode to validate new models, canary deployments to limit blast radius, and A/B tests when business metrics matter.

6. **Retraining pipelines need gates**: cooldown periods, minimum data requirements, and improvement thresholds prevent churn.

---

## Common Pitfalls

| Pitfall | Why It's Bad | Better Approach |
|---------|--------------|-----------------|
| Retraining on every drift alert | Expensive, may cause instability | Use tiered thresholds, require performance drop |
| No reference window | Can't detect drift without baseline | Store training data statistics |
| Ignoring seasonality | Normal patterns trigger false alarms | Build seasonality into reference |
| Shadow mode without logging | Can't validate shadow performance | Log all predictions with timestamps |
| Same test set for years | May not represent current reality | Periodically refresh holdout set |

---

## Next Steps

You now understand how to detect when your model is degrading and when to take action. In the next lesson, you'll bring everything together with a capstone project that implements a complete MLOps pipeline.

**Continue to Lesson 4.4:** `/start-4-4` (Capstone: Full Pipeline)

---

## Quick Reference

```python
# PSI interpretation
PSI < 0.1   → No significant change
PSI 0.1-0.25 → Moderate change (monitor)
PSI > 0.25  → Significant change (action needed)

# Retraining decision
if performance_dropped and have_labels:
    retrain()
elif data_drift_only:
    get_labels_then_decide()
elif scheduled_retrain_due:
    retrain()
else:
    monitor()

# Safe deployment order
1. Shadow mode (validate)
2. Canary (5% traffic)
3. Gradual rollout (25% → 50% → 100%)
4. Rollback plan ready at each stage
```
