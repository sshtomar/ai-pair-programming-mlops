# MLflow Cheatsheet

## Quick Reference

| Command | Description |
|---------|-------------|
| `mlflow ui` | Start tracking UI at localhost:5000 |
| `mlflow run .` | Run MLproject in current directory |
| `mlflow models serve -m runs:/<id>/model` | Serve a logged model |
| `mlflow models build-docker -m runs:/<id>/model` | Build Docker image for model |

## Tracking API

### Basic Experiment Tracking

```python
import mlflow

# Set experiment (creates if doesn't exist)
mlflow.set_experiment("sentiment-classifier")

# Start a run
with mlflow.start_run(run_name="baseline"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)
    mlflow.log_params({"batch_size": 32, "model": "logistic"})

    # Log metrics
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("f1_score", 0.85)

    # Log metric over time (with step)
    for epoch in range(10):
        mlflow.log_metric("loss", loss_value, step=epoch)

    # Log artifacts (files)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifacts("plots/")  # Log directory
```

### Log Models

```python
# Log sklearn model
from mlflow.sklearn import log_model
log_model(model, "model", input_example=X_sample)

# Log with signature (recommended)
from mlflow.models import infer_signature
signature = infer_signature(X_train, model.predict(X_train))
log_model(model, "model", signature=signature)

# Log PyTorch model
mlflow.pytorch.log_model(model, "model")

# Log any model with custom code
mlflow.pyfunc.log_model("model", python_model=MyModelWrapper())
```

### Load and Use Models

```python
# Load model from run
model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# Load model from registry
model = mlflow.sklearn.load_model("models:/sentiment-classifier/Production")

# Load as generic pyfunc
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
predictions = model.predict(data)
```

## Common Workflows

### Auto-logging (Quick Start)

```python
import mlflow

# Enable auto-logging for supported libraries
mlflow.sklearn.autolog()  # or
mlflow.pytorch.autolog()  # or
mlflow.tensorflow.autolog()

# Just train normally - everything gets logged
model.fit(X_train, y_train)
```

### Model Registry

```python
# Register model during logging
mlflow.sklearn.log_model(
    model, "model",
    registered_model_name="sentiment-classifier"
)

# Register existing run's model
mlflow.register_model(
    "runs:/<run_id>/model",
    "sentiment-classifier"
)

# Transition model stage
from mlflow import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="sentiment-classifier",
    version=1,
    stage="Production"
)
```

### Compare Runs

```python
# Search runs
runs = mlflow.search_runs(
    experiment_names=["sentiment-classifier"],
    filter_string="metrics.accuracy > 0.8",
    order_by=["metrics.f1_score DESC"]
)

# Get best run
best_run = runs.iloc[0]
print(f"Best run: {best_run.run_id}, F1: {best_run['metrics.f1_score']}")
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No active run` | Use `with mlflow.start_run():` context |
| `Experiment not found` | Call `mlflow.set_experiment()` first |
| `Cannot serialize model` | Check model dependencies are logged |
| `Artifact too large` | Use `log_artifact` for files, not `log_param` |
| `Port 5000 in use` | `mlflow ui --port 5001` |

## Best Practices

1. **Use run names**: `start_run(run_name="lr-0.01-batch-32")` for clarity
2. **Log input examples**: Helps with model validation and serving
3. **Log signatures**: Documents expected input/output types
4. **Tag runs**: `mlflow.set_tag("model_type", "baseline")` for filtering
5. **Use experiments**: Group related runs by experiment name
