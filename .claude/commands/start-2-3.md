# Lesson 2.3: Model Registry

Read the lesson content from `lesson-modules/2-pipeline/2-3-model-registry.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"You've tracked 50 experiments. One model is clearly best. How do you promote it to production? How do you roll back if it fails?"

### 2. Socratic Question
Ask: "What's the difference between 'I saved my model' and 'My model is in production'?"

Guide them to: versioning, approval workflows, deployment states, rollback capability.

### 3. Model Lifecycle Stages (10 min)
Cover the MLflow model stages:
- **None**: Just logged, not reviewed
- **Staging**: Ready for testing
- **Production**: Live, serving requests
- **Archived**: Retired, kept for reference

Ask: "Who decides when a model moves from Staging to Production at a real company?"

### 4. Register a Model (10 min)
Guide them:
```python
# After training, register the model
mlflow.register_model(
    f"runs:/{run_id}/model",
    "sentiment-classifier"
)
```

Show in UI how registered models appear.

### 5. Promote Through Stages (10 min)
```python
from mlflow import MlflowClient

client = MlflowClient()

# Move to staging
client.transition_model_version_stage(
    name="sentiment-classifier",
    version=1,
    stage="Staging"
)

# After testing, promote to production
client.transition_model_version_stage(
    name="sentiment-classifier",
    version=1,
    stage="Production"
)
```

### 6. Load by Stage (10 min)
Show how deployment code loads models:
```python
import mlflow

model = mlflow.pyfunc.load_model(
    model_uri="models:/sentiment-classifier/Production"
)
predictions = model.predict(data)
```

Key insight: deployment code never changes—model version updates happen in registry.

### 7. Wrap Up
- Models now have clear lifecycle and versioning
- Deployment is decoupled from model updates
- Preview: Lesson 2.4 adds testing to validate before promotion
- Next: `/start-2-4`

## Teaching Notes
- Registry is about governance, not just storage
- Draw parallels to software release management
- Emphasize: this enables rollback without code changes
