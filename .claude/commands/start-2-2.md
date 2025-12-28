# Lesson 2.2: Experiment Tracking

Read the lesson content from `lesson-modules/2-pipeline/2-2-experiment-tracking.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"You've trained 47 models with different hyperparameters. Which one was best? What settings did you use for each?"

### 2. Socratic Question
Ask: "How have you tracked experiments in the past? What went wrong?"

Common answers: spreadsheets, notebooks, file naming (model_v2_final_FINAL.pkl). Acknowledge the pain.

### 3. MLflow Introduction (10 min)
Cover:
- Experiment tracking: parameters, metrics, artifacts
- Model registry: versioning, staging, production
- UI for comparison and visualization
- Language-agnostic, works with any ML framework

### 4. Setup MLflow (5 min)
```bash
pip install mlflow
mlflow ui
```

Open localhost:5000 in browser. Explain the UI components.

### 5. Instrument Training Code (20 min)
Guide them to modify `src/train.py`:

```python
import mlflow

mlflow.set_experiment("sentiment-classifier")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_features", 5000)

    # Train model...

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 6. Run Multiple Experiments (10 min)
Have them:
1. Train with different hyperparameters
2. View results in MLflow UI
3. Compare runs side by side
4. Identify best model

### 7. Artifacts (5 min)
Show logging additional artifacts:
- Confusion matrix plots
- Feature importance
- Training data sample

### 8. Wrap Up
- Every experiment is now tracked and reproducible
- Can compare models objectively
- Preview: Lesson 2.3 promotes best model to registry
- Next: `/start-2-3`

## Teaching Notes
- MLflow UI is powerful—let them explore
- Emphasize: this solves "which model was that?" forever
- Connect to production: registry enables deployment
