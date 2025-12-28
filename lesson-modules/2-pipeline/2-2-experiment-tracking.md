# Lesson 2.2: Experiment Tracking with MLflow

## Learning Objectives

By the end of this lesson, students will:
1. Track experiments with MLflow
2. Log parameters, metrics, and artifacts
3. Compare experiments and select best models

## Duration: 45 minutes

---

## Part 1: The Experiment Tracking Problem

### The Familiar Chaos

You've been here before:

```
models/
├── model_lr0.01.pkl
├── model_lr0.001.pkl
├── model_lr0.001_epochs50.pkl
├── model_final.pkl
├── model_final_v2.pkl
├── model_final_v2_actually_good.pkl
└── model_BEST_USE_THIS_ONE.pkl
```

Or worse, the spreadsheet:

| Date | LR | Epochs | Accuracy | Notes |
|------|-----|--------|----------|-------|
| 3/15 | 0.01 | 20 | 0.82 | baseline |
| 3/15 | 0.001 | 20 | 0.85 | better! |
| 3/16 | 0.001 | 50 | ??? | forgot to write down |
| 3/17 | ? | ? | 0.89 | best so far but what params? |

### The Questions You Can't Answer

Without proper experiment tracking:
- "Which hyperparameters produced that 0.89 accuracy model?"
- "What data version was used for the model in production?"
- "Can you reproduce last week's training run exactly?"
- "How does the new model compare to the baseline?"

These questions become unanswerable—and they **will** be asked.

### What Experiment Tracking Solves

A proper tracking system records for every training run:
- **Parameters**: Learning rate, batch size, epochs, model architecture
- **Metrics**: Accuracy, loss, F1, any custom metric
- **Artifacts**: The model file, plots, confusion matrices
- **Environment**: Python version, package versions, git commit
- **Metadata**: Who ran it, when, what data version

All searchable. All comparable. All reproducible.

```
MLflow Tracking: Experiments, Runs, and What Gets Logged
═══════════════════════════════════════════════════════════════════════════════

  EXPERIMENT: "sentiment-classifier"
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  RUN: abc123                        RUN: def456                         │
  │  ┌─────────────────────────────┐    ┌─────────────────────────────┐     │
  │  │ Parameters                  │    │ Parameters                  │     │
  │  │ ├── learning_rate: 0.001    │    │ ├── learning_rate: 0.01     │     │
  │  │ ├── max_features: 5000      │    │ ├── max_features: 10000     │     │
  │  │ ├── epochs: 50              │    │ ├── epochs: 100             │     │
  │  │ └── batch_size: 32          │    │ └── batch_size: 64          │     │
  │  │                             │    │                             │     │
  │  │ Metrics                     │    │ Metrics                     │     │
  │  │ ├── accuracy: 0.847         │    │ ├── accuracy: 0.891         │     │
  │  │ ├── f1_score: 0.832         │    │ ├── f1_score: 0.878         │     │
  │  │ └── loss: 0.423             │    │ └── loss: 0.312             │     │
  │  │                             │    │                             │     │
  │  │ Artifacts                   │    │ Artifacts                   │     │
  │  │ ├── model/                  │    │ ├── model/                  │     │
  │  │ │   └── model.pkl           │    │ │   └── model.pkl           │     │
  │  │ ├── confusion_matrix.png    │    │ ├── confusion_matrix.png    │     │
  │  │ └── vectorizer.pkl          │    │ └── vectorizer.pkl          │     │
  │  │                             │    │                             │     │
  │  │ Tags                        │    │ Tags                        │     │
  │  │ ├── developer: alice        │    │ ├── developer: alice        │     │
  │  │ └── data_version: v2.1      │    │ └── data_version: v2.1      │     │
  │  └─────────────────────────────┘    └─────────────────────────────┘     │
  │           │                                    │                        │
  │           └──────────────┬─────────────────────┘                        │
  │                          ▼                                              │
  │              ┌─────────────────────┐                                    │
  │              │ Compare & Analyze   │                                    │
  │              │ Which params work?  │                                    │
  │              │ Best model = def456 │                                    │
  │              └─────────────────────┘                                    │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  Flow:  Experiment (logical grouping)
              └── Run (single training execution)
                    ├── Parameters (inputs: what you set)
                    ├── Metrics (outputs: what you measured)
                    ├── Artifacts (files: models, plots, data)
                    └── Tags (metadata: who, when, why)
```

---

## Part 2: MLflow Overview

### What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle. It has four components:

| Component | Purpose |
|-----------|---------|
| **Tracking** | Record parameters, metrics, artifacts from runs |
| **Projects** | Package ML code in a reproducible format |
| **Models** | Deploy models in diverse serving environments |
| **Registry** | Store, version, and manage models centrally |

Today we focus on **Tracking**. We'll cover Registry in Lesson 2.3.

### Why MLflow?

- **Open source**: No vendor lock-in
- **Language agnostic**: Python, R, Java, REST API
- **Framework agnostic**: Works with scikit-learn, PyTorch, TensorFlow, XGBoost
- **Simple**: Can be added to existing code with minimal changes
- **Scalable**: From local files to production databases

### Installation

```bash
pip install mlflow
```

Add to your `requirements.txt`:

```
mlflow>=2.9.0
```

---

## Part 3: Setting Up MLflow

### Starting the Tracking Server

MLflow can run in three modes:

1. **Local files** (default): Stores everything in `./mlruns/`
2. **Local server**: Web UI at localhost
3. **Remote server**: Shared team tracking

For development, the local server is ideal:

```bash
# Start the MLflow UI
mlflow ui --port 5000
```

Open http://localhost:5000 in your browser.

### The MLflow UI Components

```
+------------------------------------------------------------------+
|  MLflow                                              [Search...] |
+------------------------------------------------------------------+
| Experiments          |  Runs                                     |
|                      |                                           |
| > Default            |  Run Name     | Duration | Accuracy | LR  |
|   sentiment-exp      |  ------------+----------+----------+---- |
|   baseline           |  run_abc123  | 2m 30s   | 0.847    | 0.01|
|                      |  run_def456  | 3m 15s   | 0.862    | 0.001|
|                      |  run_ghi789  | 5m 02s   | 0.871    | 0.001|
+----------------------+-------------------------------------------+
```

**Left sidebar**: List of experiments (logical groupings of runs)
**Main area**: Runs table with sortable/filterable columns
**Run detail view**: Parameters, metrics over time, artifacts

### Directory Structure

After running experiments, you'll see:

```
mlruns/
├── 0/                          # Experiment ID
│   ├── meta.yaml              # Experiment metadata
│   ├── abc123/                # Run ID
│   │   ├── params/            # Logged parameters
│   │   ├── metrics/           # Logged metrics
│   │   ├── artifacts/         # Logged files (models, plots)
│   │   └── meta.yaml          # Run metadata
│   └── def456/
└── .trash/
```

---

## Part 4: Instrumenting Training Code

### Before: Training Without Tracking

Here's a typical training script:

```python
# src/train.py (before)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train():
    # Load data
    df = pd.read_csv("data/reviews.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save model
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

if __name__ == "__main__":
    train()
```

Problems:
- No record of what parameters were used
- Metrics printed but not saved
- Can't compare runs
- Which `model.pkl` corresponds to which training run?

### After: Training With MLflow

```python
# src/train.py (after)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
EXPERIMENT_NAME = "sentiment-classifier"
RANDOM_STATE = 42

def train(
    c_param: float = 1.0,
    max_features: int = 5000,
    max_iter: int = 1000,
    test_size: float = 0.2
):
    """Train sentiment classifier with full experiment tracking."""

    # Set experiment (creates if doesn't exist)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start a run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("c_param", c_param)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Load data
        df = pd.read_csv("data/reviews.csv")
        mlflow.log_param("dataset_size", len(df))

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"],
            test_size=test_size,
            random_state=RANDOM_STATE
        )

        # Feature extraction
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train model
        model = LogisticRegression(C=c_param, max_iter=max_iter)
        model.fit(X_train_vec, y_train)

        # Evaluate
        predictions = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Log the model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None  # We'll use registry in 2.3
        )

        # Log vectorizer as artifact
        import joblib
        joblib.dump(vectorizer, "vectorizer.pkl")
        mlflow.log_artifact("vectorizer.pkl")

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, f1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--max-features", type=int, default=5000, help="Max TF-IDF features")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations")
    args = parser.parse_args()

    train(c_param=args.c, max_features=args.max_features, max_iter=args.max_iter)
```

### What Changed

1. **`mlflow.set_experiment()`**: Groups related runs together
2. **`mlflow.start_run()`**: Creates a new tracked run
3. **`mlflow.log_param()`**: Records hyperparameters
4. **`mlflow.log_metric()`**: Records evaluation metrics
5. **`mlflow.log_artifact()`**: Saves files (plots, data samples)
6. **`mlflow.sklearn.log_model()`**: Saves the model with metadata

---

## Part 5: The MLflow API

### Core Logging Functions

```python
import mlflow

# Parameters (logged once per run)
mlflow.log_param("learning_rate", 0.001)
mlflow.log_params({"epochs": 50, "batch_size": 32})  # Log multiple

# Metrics (can be logged multiple times for tracking over time)
mlflow.log_metric("loss", 0.5)
mlflow.log_metric("loss", 0.3, step=1)  # With step number
mlflow.log_metric("loss", 0.1, step=2)

# Artifacts (files)
mlflow.log_artifact("path/to/file.png")
mlflow.log_artifacts("path/to/directory/")  # Log entire directory

# Tags (metadata, searchable)
mlflow.set_tag("developer", "alice")
mlflow.set_tag("model_type", "logistic_regression")
```

### Run Context Options

```python
# Option 1: Context manager (recommended)
with mlflow.start_run():
    mlflow.log_param("x", 1)
    # Run automatically ends when exiting the block

# Option 2: Manual start/end
run = mlflow.start_run()
mlflow.log_param("x", 1)
mlflow.end_run()

# Option 3: Nested runs (for hyperparameter sweeps)
with mlflow.start_run(run_name="hyperparameter-sweep"):
    for lr in [0.1, 0.01, 0.001]:
        with mlflow.start_run(nested=True, run_name=f"lr={lr}"):
            mlflow.log_param("lr", lr)
            # Train and evaluate...
            mlflow.log_metric("accuracy", accuracy)
```

### Querying Runs Programmatically

```python
import mlflow

# Get experiment by name
experiment = mlflow.get_experiment_by_name("sentiment-classifier")

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.85",
    order_by=["metrics.accuracy DESC"]
)

# Get best run
best_run = runs.iloc[0]
print(f"Best run: {best_run.run_id}")
print(f"Accuracy: {best_run['metrics.accuracy']}")

# Load model from best run
best_model = mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/model")
```

---

## Part 6: Auto-Logging vs Explicit Logging

### Auto-Logging

MLflow can automatically log parameters and metrics for supported frameworks:

```python
import mlflow

# Enable auto-logging for sklearn
mlflow.sklearn.autolog()

# Now just train normally
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.5, max_iter=500)
model.fit(X_train, y_train)
# MLflow automatically logs: C, max_iter, accuracy, etc.
```

Supported frameworks:
- scikit-learn: Model parameters, metrics, model artifact
- PyTorch: Parameters, metrics, model
- TensorFlow/Keras: Epochs, loss curves, model
- XGBoost, LightGBM, etc.

### When to Use Each

| Use Auto-Logging When | Use Explicit Logging When |
|----------------------|---------------------------|
| Quick experiments | Custom metrics needed |
| Standard workflows | Non-standard artifacts |
| Prototyping | Fine-grained control required |
| Framework-native code | Preprocessing parameters matter |

### Combining Both

```python
import mlflow
import mlflow.sklearn

# Enable auto-logging
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Add custom logging on top of auto-logging
    mlflow.log_param("data_version", "v2.1")
    mlflow.log_param("feature_engineering", "tfidf_5000")

    # Auto-logged: model params, training metrics
    model = LogisticRegression(C=0.5)
    model.fit(X_train, y_train)

    # Custom metrics
    custom_metric = calculate_business_metric(model, X_test, y_test)
    mlflow.log_metric("business_value_score", custom_metric)
```

---

## Part 7: Organizing Experiments

### Experiment Naming Conventions

```
{project}-{model_type}-{purpose}

Examples:
- sentiment-logreg-baseline
- sentiment-bert-hyperparameter-sweep
- sentiment-ensemble-production-candidate
```

### Tagging Strategy

Use tags to add searchable metadata:

```python
mlflow.set_tag("team", "nlp")
mlflow.set_tag("stage", "development")  # development, staging, production
mlflow.set_tag("data_version", "dvc:abc123")
mlflow.set_tag("git_commit", "def456")
```

### Run Naming

```python
with mlflow.start_run(run_name=f"lr={lr}_epochs={epochs}"):
    # Descriptive names make the UI more useful
```

---

## Part 8: Viewing and Comparing Runs

### In the UI

1. **Sorting**: Click column headers to sort by any metric
2. **Filtering**: Use the search bar with expressions like `metrics.accuracy > 0.85`
3. **Comparing**: Select multiple runs and click "Compare"
4. **Charts**: Visualize metrics across runs

### Comparison View Features

When comparing runs, you see:
- **Parameter diff**: Which parameters changed between runs
- **Metric comparison**: Side-by-side metrics
- **Parallel coordinates plot**: Visualize parameter-metric relationships
- **Scatter plots**: Plot any two metrics against each other

### Filter Expression Syntax

```
# Numeric comparisons
metrics.accuracy > 0.85
params.learning_rate = "0.001"

# String matching
tags.developer = "alice"
attributes.run_name LIKE "%baseline%"

# Combining conditions
metrics.accuracy > 0.8 AND params.max_features = "5000"
```

---

## Exercises

### Exercise 2.2.1: Add MLflow to Training

Modify your `src/train.py` to include MLflow tracking:

1. Set up an experiment called "sentiment-classifier"
2. Log these parameters: vectorizer max_features, model C parameter
3. Log these metrics: accuracy, f1_score
4. Log the trained model as an artifact

Run training and verify the run appears at http://localhost:5000.

### Exercise 2.2.2: Run Multiple Experiments

Run training with different hyperparameters:

```bash
python src/train.py --c 0.1
python src/train.py --c 1.0
python src/train.py --c 10.0
python src/train.py --max-features 1000
python src/train.py --max-features 5000
python src/train.py --max-features 10000
```

In the MLflow UI:
1. Sort runs by accuracy
2. Which C value performed best?
3. Which max_features setting performed best?
4. Use the Compare feature to see the parameter-metric relationships

### Exercise 2.2.3: Log Custom Artifacts

Extend your training to log:

1. A confusion matrix plot (saved as PNG)
2. A classification report (saved as text file)
3. Sample predictions (10 random test examples with their predictions)

Verify these appear in the Artifacts tab of each run.

### Exercise 2.2.4: Query Runs Programmatically

Write a script that:

1. Finds all runs with accuracy > 0.80
2. Prints the top 3 runs by F1 score
3. Loads the best model and makes a prediction

```python
# Start here
import mlflow

experiment = mlflow.get_experiment_by_name("sentiment-classifier")
runs = mlflow.search_runs(...)
```

---

## Common Pitfalls

### 1. Forgetting to End Runs

```python
# BAD: Run never ends if exception occurs
mlflow.start_run()
train()  # If this crashes, run stays "active"
mlflow.end_run()

# GOOD: Context manager handles cleanup
with mlflow.start_run():
    train()
```

### 2. Logging After Run Ends

```python
# BAD: This logs to no run
with mlflow.start_run():
    model = train()

mlflow.log_metric("accuracy", accuracy)  # ERROR: No active run

# GOOD: Log inside the context
with mlflow.start_run():
    model = train()
    mlflow.log_metric("accuracy", accuracy)
```

### 3. Hardcoded Tracking URI

```python
# BAD: Hardcoded path
mlflow.set_tracking_uri("file:///home/user/mlruns")

# GOOD: Use environment variable
import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
```

### 4. Not Versioning the Data

MLflow tracks code and model, but not data. Always log:

```python
mlflow.log_param("data_version", "dvc:abc123")  # Reference DVC version
mlflow.log_param("data_path", "data/reviews.csv")
mlflow.log_param("data_hash", hashlib.md5(data.encode()).hexdigest()[:8])
```

---

## Key Takeaways

1. **Experiment tracking solves the "which model was that?" problem**—every run is logged with parameters, metrics, and artifacts
2. **MLflow's tracking API is minimal**—`log_param()`, `log_metric()`, `log_artifact()` cover most needs
3. **Use `start_run()` as a context manager**—ensures proper cleanup even on errors
4. **Combine auto-logging with explicit logging**—auto-logging for convenience, explicit for custom needs
5. **Good organization matters**—use meaningful experiment names, run names, and tags

---

## Next Steps

You now have experiment tracking, but how do you promote a good model to production? How do you know which model is currently deployed?

Run `/start-2-3` to learn about the MLflow Model Registry—versioning and staging models for deployment.
