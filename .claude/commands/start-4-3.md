# Lesson 4.3: Model Drift & Retraining

Read the lesson content from `lesson-modules/4-production/4-3-drift-retraining.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"Your model was 95% accurate when you deployed it. Six months later, it's 70%. The code didn't change. What happened?"

### 2. Socratic Question
Ask: "Why would a model's performance degrade over time even if the code stays the same?"

Expected: Data changes, user behavior shifts, world changes (COVID, new products, etc.). Guide them to understand that models are trained on a snapshot of reality that becomes stale.

### 3. Types of Drift (10 min)
Cover the drift taxonomy:

**Data Drift (Covariate Shift)**
- Input distribution changes
- Example: New vocabulary in reviews, different demographics
- Detection: Monitor feature distributions

**Concept Drift**
- Relationship between inputs and outputs changes
- Example: "sick" used to mean ill, now means cool
- Detection: Monitor prediction accuracy (requires labels)

**Label Drift**
- Target distribution changes
- Example: More positive reviews after product improvement
- Detection: Monitor prediction distribution

Ask: "Which type is hardest to detect? Why?"

### 4. Detecting Drift (20 min)
Build a drift detection module:

```python
# drift/detector.py
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class DriftResult:
    is_drifted: bool
    p_value: float
    statistic: float

def detect_distribution_drift(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05
) -> DriftResult:
    """Kolmogorov-Smirnov test for distribution drift."""
    statistic, p_value = stats.ks_2samp(reference, current)
    return DriftResult(
        is_drifted=p_value < threshold,
        p_value=p_value,
        statistic=statistic
    )

def detect_prediction_drift(
    reference_predictions: list,
    current_predictions: list,
    threshold: float = 0.05
) -> DriftResult:
    """Chi-squared test for categorical prediction drift."""
    ref_counts = np.bincount([0 if p == 'negative' else 1 for p in reference_predictions])
    cur_counts = np.bincount([0 if p == 'negative' else 1 for p in current_predictions])

    statistic, p_value = stats.chisquare(cur_counts, ref_counts)
    return DriftResult(
        is_drifted=p_value < threshold,
        p_value=p_value,
        statistic=statistic
    )
```

### 5. Monitoring for Drift (10 min)
Add drift checking to the API:

```python
from collections import deque
from drift.detector import detect_prediction_drift

# Store recent predictions (in production, use a database)
prediction_window = deque(maxlen=1000)
reference_predictions = load_reference_predictions()

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.text])[0]
    prediction_window.append(prediction)
    return PredictionResponse(...)

@app.get("/drift-check")
def check_drift():
    if len(prediction_window) < 100:
        return {"status": "insufficient_data", "samples": len(prediction_window)}

    result = detect_prediction_drift(
        reference_predictions,
        list(prediction_window)
    )
    return {
        "status": "drifted" if result.is_drifted else "stable",
        "p_value": result.p_value,
        "samples_checked": len(prediction_window)
    }
```

### 6. Retraining Strategies (15 min)
Discuss when and how to retrain:

**Scheduled Retraining**
- Weekly/monthly on new data
- Simple, predictable
- May retrain unnecessarily

**Triggered Retraining**
- When drift detected
- When performance drops
- More efficient, more complex

**Continuous Training**
- Constantly incorporating new data
- Complex infrastructure
- Best for rapidly changing domains

Exercise: Design a retraining pipeline:
```
Drift Detected → Alert → Collect New Data →
  → Validate Data → Train New Model →
  → Validate Performance → Register Model →
  → Deploy (Canary) → Monitor → Full Rollout
```

### 7. The Retraining Pipeline (10 min)
Sketch the automation:

```yaml
# .github/workflows/retrain.yml (triggered manually or by alert)
name: Retrain Model

on:
  workflow_dispatch:
    inputs:
      reason:
        description: 'Reason for retraining'
        required: true

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - name: Pull latest data
        run: dvc pull

      - name: Train model
        run: python train.py

      - name: Validate model
        run: python scripts/validate_model.py

      - name: Register model
        run: python scripts/register_model.py

      - name: Deploy canary
        run: ./scripts/deploy_canary.sh
```

### 8. Wrap Up
- Models decay—drift is inevitable
- Detect drift before users notice
- Automate retraining but keep humans in the loop
- Preview: Lesson 4.4 brings everything together
- Next: `/start-4-4`

## Teaching Notes
- Drift is the most "ML-specific" operational challenge
- Without labels, drift detection is harder—discuss proxies
- Don't over-automate retraining—bad data can corrupt models
- Connect to earlier lessons: DVC and MLflow enable reproducible retraining
