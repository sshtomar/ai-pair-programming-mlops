# Lesson 2.3: Model Registry

**Duration:** 35 minutes

**Prerequisites:** Lesson 2.2 (Experiment Tracking with MLflow)

## Learning Objectives

By the end of this lesson, you will:
1. Understand model lifecycle stages and why they matter
2. Register models with proper versioning
3. Implement promotion workflows from staging to production

---

## Why a Model Registry?

Think about software releases. You don't deploy commits directly to production—you tag releases, promote through environments, and keep rollback capability. Models need the same discipline.

**The Model Registry solves three problems:**

1. **Governance**: Which model is in production? Who approved it? When?
2. **Rollback**: Production model failing? Switch back in seconds, no redeployment needed.
3. **Deployment Decoupling**: Deployment code always loads "the production model"—it never changes when you promote a new version.

Without a registry, your deployment code contains hardcoded run IDs or file paths. Every model update requires a code change. That's fragile.

---

## Model Lifecycle Stages

MLflow defines four lifecycle stages:

| Stage | Purpose |
|-------|---------|
| **None** | Just registered, not evaluated yet |
| **Staging** | Under evaluation, integration testing |
| **Production** | Serving live traffic |
| **Archived** | Retired, kept for audit/rollback |

```
Model Lifecycle: From Training to Production to Retirement
═══════════════════════════════════════════════════════════════════════════════

                         ┌─────────────────────────────────────────────────┐
                         │              MODEL REGISTRY                     │
                         │         "sentiment-classifier"                  │
                         └─────────────────────────────────────────────────┘
                                              │
   ┌──────────────────────────────────────────┼──────────────────────────────┐
   │                                          │                              │
   ▼                                          ▼                              ▼
┌──────────┐    promote     ┌──────────┐    promote     ┌──────────────┐
│   NONE   │ ─────────────► │ STAGING  │ ─────────────► │  PRODUCTION  │
│          │                │          │                │              │
│  v4 new  │                │  v3 test │                │   v2 live    │
│  ○ ○ ○   │                │  ◐ ◐ ◐   │                │   ● ● ●      │
└──────────┘                └──────────┘                └──────────────┘
     │                           │                              │
     │ Just registered           │ Integration tests            │ Serving traffic
     │ Not evaluated             │ Shadow testing               │ Monitored
     │                           │ Performance validation       │
     │                           │                              │
     │                           │                              ▼
     │                           │                      ┌──────────────┐
     │                           │   when replaced ───► │   ARCHIVED   │
     │                           │                      │              │
     │                           │                      │   v1 old     │
     │                           │                      │   ◌ ◌ ◌      │
     │                           │                      └──────────────┘
     │                           │                              │
     │                           │                              │ Kept for:
     │                           │                              │ - Rollback
     │                           │                              │ - Audit trail
     │                           │                              │ - Comparison
     │                           │                              │
     └───────────────────────────┴──────────────────────────────┘

  Typical Flow:
  ─────────────
  1. Train new model ──► Register as v4 (None)
  2. Run tests       ──► Promote v4 to Staging
  3. Validate        ──► Promote v4 to Production
  4. Auto-archive    ──► v2 moves to Archived

  Rollback (emergency):
  ─────────────────────
  Production failing? ──► Promote v1 from Archived back to Production
                          No code changes. No redeployment. Instant switch.
```

**Note on MLflow versions:** MLflow 2.9+ introduced model aliases as an alternative to stages. We'll cover both approaches—stages are still widely used and conceptually important.

---

## Registering Models

### From a Training Run

After logging a model during training, register it:

```python
import mlflow
from mlflow.tracking import MlflowClient

# During training - log the model
with mlflow.start_run() as run:
    # ... training code ...
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

# Register from the logged artifact
model_uri = f"runs:/{run_id}/model"
model_name = "sentiment-classifier"

# This creates version 1, 2, 3... automatically
result = mlflow.register_model(model_uri, model_name)
print(f"Registered version: {result.version}")
```

### Inline Registration During Training

Alternatively, register directly when logging:

```python
with mlflow.start_run():
    # ... training code ...
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="sentiment-classifier"  # Registers automatically
    )
```

---

## Programmatic Model Management

The `MlflowClient` provides full control over the registry:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List all registered models
for rm in client.search_registered_models():
    print(f"Model: {rm.name}")

# Get specific model details
model = client.get_registered_model("sentiment-classifier")
print(f"Latest versions: {model.latest_versions}")

# Get a specific version
version = client.get_model_version("sentiment-classifier", "3")
print(f"Version 3 stage: {version.current_stage}")
print(f"Run ID: {version.run_id}")
```

---

## Transitioning Between Stages

### Promoting a Model

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promote version 3 to Staging
client.transition_model_version_stage(
    name="sentiment-classifier",
    version="3",
    stage="Staging"
)

# After validation, promote to Production
client.transition_model_version_stage(
    name="sentiment-classifier",
    version="3",
    stage="Production",
    archive_existing_versions=True  # Auto-archive current production version
)
```

The `archive_existing_versions=True` parameter ensures only one version is in Production at a time—the old production version moves to Archived automatically.

### Adding Context to Transitions

Always document why a model was promoted:

```python
# Add description to model version
client.update_model_version(
    name="sentiment-classifier",
    version="3",
    description="Promoted after passing integration tests. F1: 0.89, latency p99: 45ms"
)

# Tag for additional metadata
client.set_model_version_tag(
    name="sentiment-classifier",
    version="3",
    key="approved_by",
    value="alice@company.com"
)
```

---

## Loading Models by Stage

This is where the registry shines. Your deployment code never changes:

```python
import mlflow

# Load whatever model is currently in Production
model = mlflow.pyfunc.load_model("models:/sentiment-classifier/Production")

# Load from Staging for testing
staging_model = mlflow.pyfunc.load_model("models:/sentiment-classifier/Staging")

# Load specific version (for debugging/comparison)
v2_model = mlflow.pyfunc.load_model("models:/sentiment-classifier/2")
```

The URI format: `models:/<model-name>/<stage-or-version>`

### Production Prediction Code

Here's what `predict.py` looks like with registry integration:

```python
# project/src/predict.py
import mlflow

class SentimentPredictor:
    def __init__(self, stage: str = "Production"):
        """Load model from registry by stage."""
        model_uri = f"models:/sentiment-classifier/{stage}"
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, texts: list[str]) -> list[dict]:
        """Run predictions on input texts."""
        predictions = self.model.predict(texts)
        return [
            {"text": text, "sentiment": pred}
            for text, pred in zip(texts, predictions)
        ]

# Usage - always loads current production model
predictor = SentimentPredictor()
results = predictor.predict(["Great product!", "Terrible experience"])

# For testing against staging
staging_predictor = SentimentPredictor(stage="Staging")
```

When you promote a new model to Production, the next `SentimentPredictor()` instantiation loads the new version. No code changes. No redeployment of the prediction service (just restart to pick up the new model).

---

## Model Aliases (MLflow 2.9+)

Newer MLflow versions offer aliases as a flexible alternative to the fixed stage system:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Set an alias (like a git tag)
client.set_registered_model_alias(
    name="sentiment-classifier",
    alias="champion",
    version="3"
)

# Set another alias
client.set_registered_model_alias(
    name="sentiment-classifier",
    alias="challenger",
    version="4"
)

# Load by alias
champion = mlflow.pyfunc.load_model("models:/sentiment-classifier@champion")
challenger = mlflow.pyfunc.load_model("models:/sentiment-classifier@challenger")
```

**Aliases vs Stages:**
- Stages: Fixed set (None/Staging/Production/Archived), one version per stage
- Aliases: Arbitrary names, more flexible, better for A/B testing scenarios

Choose based on your workflow. Stages work well for simple linear promotion. Aliases suit complex scenarios like canary deployments.

---

## Rollback Strategy

When production issues occur:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

def rollback_to_version(model_name: str, version: str):
    """Emergency rollback to a specific version."""
    # Demote current production
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    if current_prod:
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod[0].version,
            stage="Archived"
        )

    # Promote the rollback target
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"Rolled back to version {version}")

# Usage: Something's wrong, go back to v2
rollback_to_version("sentiment-classifier", "2")
```

No code deployment. No Docker image rebuild. Just update the registry and restart your prediction service.

---

## Exercise: Register and Promote Your Model

**Task:** Register your trained model and promote it through stages.

### Step 1: Ensure You Have a Trained Model

If you don't have a logged model from Lesson 2.2, train one:

```bash
cd project
python -m src.train
```

### Step 2: Create a Registry Script

Create `project/scripts/register_model.py`:

```python
"""Register and manage model versions in MLflow registry."""
import mlflow
from mlflow.tracking import MlflowClient

# Configure MLflow (use same tracking URI as training)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "sentiment-classifier"

def get_best_run() -> str:
    """Find the run with highest accuracy."""
    runs = mlflow.search_runs(
        experiment_names=["sentiment-analysis"],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    if runs.empty:
        raise ValueError("No runs found. Train a model first.")
    return runs.iloc[0].run_id

def register_best_model() -> str:
    """Register the best model, return version number."""
    run_id = get_best_run()
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {result.version}")
    return result.version

def promote_to_staging(version: str):
    """Move model version to Staging."""
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging"
    )
    print(f"Version {version} → Staging")

def promote_to_production(version: str):
    """Move model version to Production, archive existing."""
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Version {version} → Production")

def show_model_versions():
    """Display all versions and their stages."""
    print(f"\n{MODEL_NAME} versions:")
    print("-" * 50)
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        print(f"  v{v.version}: {v.current_stage:12} (run: {v.run_id[:8]})")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python register_model.py <command> [version]")
        print("Commands: register, staging, production, list")
        sys.exit(1)

    command = sys.argv[1]

    if command == "register":
        register_best_model()
    elif command == "staging":
        version = sys.argv[2] if len(sys.argv) > 2 else "1"
        promote_to_staging(version)
    elif command == "production":
        version = sys.argv[2] if len(sys.argv) > 2 else "1"
        promote_to_production(version)
    elif command == "list":
        show_model_versions()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    show_model_versions()
```

### Step 3: Work Through the Promotion Flow

```bash
# Register your best model
python scripts/register_model.py register

# Check status
python scripts/register_model.py list

# Promote to staging
python scripts/register_model.py staging 1

# Promote to production
python scripts/register_model.py production 1

# Train a new model (tweak params if you like)
python -m src.train

# Register the new version
python scripts/register_model.py register

# Promote new version - old one auto-archives
python scripts/register_model.py production 2
```

### Step 4: Update Your Prediction Code

Modify `project/src/predict.py` to load from the registry:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_production_model():
    """Load current production model from registry."""
    return mlflow.pyfunc.load_model("models:/sentiment-classifier/Production")

def predict(texts: list[str]) -> list[str]:
    """Predict sentiment for input texts."""
    model = load_production_model()
    return model.predict(texts).tolist()

if __name__ == "__main__":
    model = load_production_model()
    test_texts = ["This is amazing!", "Worst purchase ever."]
    predictions = model.predict(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"{pred}: {text}")
```

### Step 5: Test Registry-Based Loading

```bash
python -m src.predict
```

Your prediction code now automatically uses whatever model is in Production.

---

## Key Takeaways

1. **Registry = Release Management for Models**: Same principles as software releases—stages, promotion, rollback.

2. **Deployment Decoupling**: Load models by stage (`models:/name/Production`), not by run ID. Deployment code stays constant.

3. **Instant Rollback**: Bad model in production? One API call to swap versions. No code changes, no redeployment.

4. **Audit Trail**: Every version has metadata—who trained it, what metrics, when promoted. Critical for ML governance.

5. **Automation Ready**: The `MlflowClient` API enables CI/CD pipelines to automatically register and promote models based on test results.

---

## Common Patterns

**Pattern 1: CI/CD Integration**
```
Train → Register → Auto-promote to Staging → Run tests → If pass, promote to Production
```

**Pattern 2: Shadow Mode**
```
Production: stable model
Staging: new model receiving shadow traffic for comparison
Promote when metrics meet threshold
```

**Pattern 3: Canary Deployment**
```
Production: stable model (90% traffic)
Challenger alias: new model (10% traffic)
Gradually shift traffic based on metrics
```

---

## Next Steps

You now have experiment tracking and a model registry. The missing piece: how do you trust that a new model version is safe to promote?

In **Lesson 2.4**, we'll implement automated testing for ML models—data validation, model performance checks, and behavioral tests that gate promotions.

**Continue with:** `/start-2-4`
