# Lesson 4.1: CI/CD for Machine Learning

**Duration:** 50 minutes
**Prerequisites:** Deployed model from Level 3, familiarity with Git, basic testing knowledge

## Learning Objectives

By the end of this lesson, you will:
1. Build a GitHub Actions workflow tailored for ML pipelines
2. Automate testing, building, and deployment with appropriate gates
3. Implement model validation checks that prevent bad models from reaching production

---

## Why CI/CD for ML is Different

Traditional CI/CD: `code change -> test -> build -> deploy`

ML CI/CD: `code OR data OR model change -> test code -> test data -> train -> validate model -> build -> deploy to staging -> validate in staging -> deploy to production`

### ML CI/CD Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ML CI/CD Pipeline                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  COMMIT  │───>│   TEST   │───>│  BUILD   │───>│  DEPLOY  │───>│ MONITOR  │
  │          │    │          │    │          │    │          │    │          │
  │ ┌──────┐ │    │ ┌──────┐ │    │ ┌──────┐ │    │ ┌──────┐ │    │ ┌──────┐ │
  │ │ Code │ │    │ │ Lint │ │    │ │Docker│ │    │ │Stage │ │    │ │Metrics│ │
  │ │ Data │ │    │ │ Unit │ │    │ │Image │ │    │ │ Prod │ │    │ │ Logs │ │
  │ │Config│ │    │ │Model │ │    │ │ Push │ │    │ │Smoke │ │    │ │Alerts│ │
  │ └──────┘ │    │ └──────┘ │    │ └──────┘ │    │ └──────┘ │    │ └──────┘ │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │               │               │               │               │
       │               │               │               │               │
       ▼               ▼               ▼               ▼               ▼
   Triggers:       Gates:          Artifacts:     Environments:   Observability:
   - git push      - Tests pass    - Container    - Staging       - Prometheus
   - PR opened     - Accuracy OK   - SBOM         - Production    - Grafana
   - Schedule      - Latency OK    - Metadata     - Approval      - Alerts
```

### The Unique Challenges

| Challenge | Traditional Software | Machine Learning |
|-----------|---------------------|------------------|
| What triggers a build? | Code changes | Code, data, config, or hyperparameters |
| What constitutes "passing"? | Tests pass | Tests pass AND model meets quality bar |
| Build duration | Minutes | Minutes to hours (training) |
| Artifacts | Binary/container | Binary + model weights + metadata |
| Rollback criteria | Errors/crashes | Errors OR accuracy drop OR latency spike |
| Testing | Deterministic | Statistical thresholds |

### The Three Pipelines

ML systems need three interconnected pipelines:

```
┌─────────────────────────────────────────────────────────────┐
│                     CI Pipeline (Code)                       │
│  lint -> unit tests -> integration tests -> build container │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Training Pipeline (Model)                  │
│  data validation -> train -> evaluate -> register model     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CD Pipeline (Deployment)                  │
│  model validation -> staging deploy -> smoke tests -> prod  │
└─────────────────────────────────────────────────────────────┘
```

This lesson focuses on automating these pipelines with GitHub Actions.

---

## GitHub Actions Fundamentals

### Core Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Workflow** | Automated process defined in YAML | `.github/workflows/ml-pipeline.yml` |
| **Event** | Trigger that starts a workflow | `push`, `pull_request`, `schedule` |
| **Job** | Set of steps that run on same runner | `test`, `build`, `deploy` |
| **Step** | Individual task within a job | `run: pytest tests/` |
| **Runner** | Server that executes jobs | `ubuntu-latest`, `self-hosted` |
| **Action** | Reusable unit of code | `actions/checkout@v4` |

### Workflow Structure

```yaml
name: Workflow Name

on:                          # Events that trigger this workflow
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:                         # Environment variables for all jobs
  PYTHON_VERSION: "3.11"

jobs:
  job-name:
    runs-on: ubuntu-latest   # Runner type
    steps:
      - name: Step description
        uses: action/name@version  # Use a pre-built action
        with:
          parameter: value

      - name: Another step
        run: |                     # Run shell commands
          echo "Hello"
          python script.py
```

### Common Triggers for ML

```yaml
on:
  # Code changes
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'tests/**'
      - 'requirements.txt'

  # Pull requests
  pull_request:
    branches: [main]

  # Scheduled training (nightly)
  schedule:
    - cron: '0 2 * * *'    # 2 AM UTC daily

  # Manual trigger with inputs
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment target'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
      force_retrain:
        description: 'Force model retraining'
        type: boolean
        default: false
```

---

## Complete ML Pipeline Workflow

Create `.github/workflows/ml-pipeline.yml`:

```yaml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'tests/**'
      - 'models/**'
      - 'requirements*.txt'
      - '.github/workflows/ml-pipeline.yml'

  pull_request:
    branches: [main]

  workflow_dispatch:
    inputs:
      deploy_target:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

env:
  PYTHON_VERSION: "3.11"
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ============================================================
  # Stage 1: Code Quality & Unit Tests
  # ============================================================
  lint-and-test:
    name: Lint & Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203
          black --check src/ tests/
          isort --check-only src/ tests/

      - name: Run type checking
        run: mypy src/ --ignore-missing-imports

      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=term-missing \
            -v

      - name: Upload coverage report
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          fail_ci_if_error: false

  # ============================================================
  # Stage 2: Integration Tests
  # ============================================================
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: lint-and-test

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v --tb=short

  # ============================================================
  # Stage 3: Model Validation
  # ============================================================
  validate-model:
    name: Validate Model
    runs-on: ubuntu-latest
    needs: lint-and-test

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Download model artifact
        run: |
          # In production, fetch from model registry (MLflow, DVC, S3)
          # For this example, we use DVC
          dvc pull models/sentiment_model.pkl

      - name: Run model validation
        id: validate
        run: |
          python scripts/validate_model.py \
            --model-path models/sentiment_model.pkl \
            --test-data data/test_holdout.csv \
            --min-accuracy 0.85 \
            --max-latency-ms 100 \
            --output-file validation_report.json

      - name: Upload validation report
        uses: actions/upload-artifact@v4
        with:
          name: model-validation-report
          path: validation_report.json

      - name: Check validation passed
        run: |
          if [ "$(jq -r '.passed' validation_report.json)" != "true" ]; then
            echo "Model validation failed!"
            jq '.' validation_report.json
            exit 1
          fi

  # ============================================================
  # Stage 4: Build Container Image
  # ============================================================
  build:
    name: Build Container
    runs-on: ubuntu-latest
    needs: [integration-tests, validate-model]
    if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'

    permissions:
      contents: read
      packages: write

    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build-push.outputs.digest }}

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata for Docker
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}

      - name: Build and push Docker image
        id: build-push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  # ============================================================
  # Stage 5: Deploy to Staging
  # ============================================================
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment:
      name: staging
      url: https://staging.example.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Deploy to staging
        run: |
          echo "Deploying ${{ needs.build.outputs.image-tag }} to staging..."
          # Example: kubectl, helm, or cloud CLI
          # kubectl set image deployment/sentiment-api \
          #   api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build.outputs.image-digest }}

      - name: Wait for deployment
        run: |
          echo "Waiting for deployment to stabilize..."
          sleep 30
          # kubectl rollout status deployment/sentiment-api --timeout=300s

      - name: Run smoke tests
        run: |
          echo "Running smoke tests against staging..."
          python scripts/smoke_tests.py --endpoint https://staging.example.com/predict

  # ============================================================
  # Stage 6: Deploy to Production (Manual Approval)
  # ============================================================
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://api.example.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Production deployment commands

      - name: Verify deployment
        run: |
          echo "Verifying production deployment..."
          python scripts/smoke_tests.py --endpoint https://api.example.com/predict

      - name: Notify team
        if: success()
        run: |
          echo "Deployment successful! Notifying team..."
          # curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
          #   -H 'Content-type: application/json' \
          #   -d '{"text":"Production deployment completed: ${{ github.sha }}"}'
```

---

## Model Validation Script

Create `scripts/validate_model.py`:

```python
#!/usr/bin/env python3
"""Model validation script for CI/CD pipeline.

Validates that a model meets minimum quality requirements before deployment.
Exits with code 1 if validation fails, blocking the pipeline.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd


class ValidationResult(TypedDict):
    passed: bool
    accuracy: float
    accuracy_threshold: float
    latency_p50_ms: float
    latency_p99_ms: float
    latency_threshold_ms: float
    per_class_recall: dict[str, float]
    recall_threshold: float
    checks: dict[str, bool]
    errors: list[str]


def load_model(model_path: Path):
    """Load model from path."""
    import joblib
    return joblib.load(model_path)


def measure_latency(model, texts: list[str], n_runs: int = 100) -> dict[str, float]:
    """Measure prediction latency statistics."""
    # Warm up
    for text in texts[:5]:
        model.predict([text])

    latencies = []
    for _ in range(n_runs):
        text = np.random.choice(texts)
        start = time.perf_counter()
        model.predict([text])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def calculate_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, float]:
    """Calculate accuracy and per-class recall."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    per_class_recall = {}
    for label in labels:
        mask = y_true_arr == label
        if mask.sum() > 0:
            correct = (y_pred_arr[mask] == label).sum()
            per_class_recall[label] = float(correct / mask.sum())
        else:
            per_class_recall[label] = 0.0

    return {"accuracy": accuracy, "per_class_recall": per_class_recall}


def validate_model(
    model_path: Path,
    test_data_path: Path,
    min_accuracy: float = 0.85,
    min_recall: float = 0.70,
    max_latency_ms: float = 100.0,
) -> ValidationResult:
    """Run all validation checks on the model."""
    errors: list[str] = []
    checks: dict[str, bool] = {}

    # Load model
    try:
        model = load_model(model_path)
        checks["model_loads"] = True
    except FileNotFoundError as e:
        errors.append(f"Model file not found: {e}")
        checks["model_loads"] = False
        return ValidationResult(
            passed=False,
            accuracy=0.0,
            accuracy_threshold=min_accuracy,
            latency_p50_ms=0.0,
            latency_p99_ms=0.0,
            latency_threshold_ms=max_latency_ms,
            per_class_recall={},
            recall_threshold=min_recall,
            checks=checks,
            errors=errors,
        )

    # Load test data
    try:
        test_df = pd.read_csv(test_data_path)
        texts = test_df["text"].tolist()
        labels = test_df["label"].tolist()
        unique_labels = test_df["label"].unique().tolist()
        checks["data_loads"] = True
    except FileNotFoundError as e:
        errors.append(f"Test data not found: {e}")
        checks["data_loads"] = False
        return ValidationResult(
            passed=False,
            accuracy=0.0,
            accuracy_threshold=min_accuracy,
            latency_p50_ms=0.0,
            latency_p99_ms=0.0,
            latency_threshold_ms=max_latency_ms,
            per_class_recall={},
            recall_threshold=min_recall,
            checks=checks,
            errors=errors,
        )

    # Get predictions
    try:
        predictions = model.predict(texts)
        checks["model_predicts"] = True
    except RuntimeError as e:
        errors.append(f"Model prediction failed: {e}")
        checks["model_predicts"] = False
        return ValidationResult(
            passed=False,
            accuracy=0.0,
            accuracy_threshold=min_accuracy,
            latency_p50_ms=0.0,
            latency_p99_ms=0.0,
            latency_threshold_ms=max_latency_ms,
            per_class_recall={},
            recall_threshold=min_recall,
            checks=checks,
            errors=errors,
        )

    # Calculate metrics
    metrics = calculate_metrics(labels, predictions.tolist(), unique_labels)
    accuracy = metrics["accuracy"]
    per_class_recall = metrics["per_class_recall"]

    # Check accuracy threshold
    checks["accuracy_above_threshold"] = accuracy >= min_accuracy
    if not checks["accuracy_above_threshold"]:
        errors.append(
            f"Accuracy {accuracy:.2%} below threshold {min_accuracy:.2%}"
        )

    # Check per-class recall
    checks["recall_above_threshold"] = all(
        r >= min_recall for r in per_class_recall.values()
    )
    if not checks["recall_above_threshold"]:
        for label, recall in per_class_recall.items():
            if recall < min_recall:
                errors.append(
                    f"Recall for '{label}' is {recall:.2%}, below {min_recall:.2%}"
                )

    # Measure latency
    latency = measure_latency(model, texts)
    checks["latency_acceptable"] = latency["p99_ms"] <= max_latency_ms
    if not checks["latency_acceptable"]:
        errors.append(
            f"P99 latency {latency['p99_ms']:.1f}ms exceeds {max_latency_ms}ms"
        )

    # Overall pass/fail
    passed = all(checks.values())

    return ValidationResult(
        passed=passed,
        accuracy=accuracy,
        accuracy_threshold=min_accuracy,
        latency_p50_ms=latency["p50_ms"],
        latency_p99_ms=latency["p99_ms"],
        latency_threshold_ms=max_latency_ms,
        per_class_recall=per_class_recall,
        recall_threshold=min_recall,
        checks=checks,
        errors=errors,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate ML model for deployment")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--max-latency-ms", type=float, default=100.0)
    parser.add_argument("--output-file", type=Path, default=None)

    args = parser.parse_args()

    print(f"Validating model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Thresholds: accuracy>={args.min_accuracy}, latency<={args.max_latency_ms}ms")
    print("-" * 50)

    result = validate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        min_accuracy=args.min_accuracy,
        min_recall=args.min_recall,
        max_latency_ms=args.max_latency_ms,
    )

    # Print results
    print(f"\nAccuracy: {result['accuracy']:.2%} (threshold: {result['accuracy_threshold']:.2%})")
    print(f"Latency P50: {result['latency_p50_ms']:.1f}ms")
    print(f"Latency P99: {result['latency_p99_ms']:.1f}ms (threshold: {result['latency_threshold_ms']}ms)")

    print("\nPer-class recall:")
    for label, recall in result["per_class_recall"].items():
        status = "PASS" if recall >= result["recall_threshold"] else "FAIL"
        print(f"  {label}: {recall:.2%} [{status}]")

    print("\nValidation checks:")
    for check, passed in result["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")

    if result["errors"]:
        print("\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")

    # Write output file
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport written to: {args.output_file}")

    # Exit with appropriate code
    print(f"\n{'='*50}")
    if result["passed"]:
        print("VALIDATION PASSED - Model approved for deployment")
        sys.exit(0)
    else:
        print("VALIDATION FAILED - Model blocked from deployment")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Secrets Management

### GitHub Secrets Setup

Never commit secrets to code. Use GitHub's encrypted secrets:

```yaml
# Access secrets in workflow
env:
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

### Required Secrets for ML Pipelines

| Secret | Purpose | Example |
|--------|---------|---------|
| `AWS_ACCESS_KEY_ID` | S3 access for DVC/artifacts | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | S3 access for DVC/artifacts | `wJal...` |
| `MLFLOW_TRACKING_URI` | Experiment tracking | `https://mlflow.example.com` |
| `DOCKER_PASSWORD` | Container registry | `dckr_pat_...` |
| `SLACK_WEBHOOK` | Deployment notifications | `https://hooks.slack.com/...` |

### Setting Secrets via CLI

```bash
# Using GitHub CLI
gh secret set AWS_ACCESS_KEY_ID --body "AKIA..."
gh secret set AWS_SECRET_ACCESS_KEY --body "wJal..."

# List secrets (values hidden)
gh secret list
```

### Environment-Specific Secrets

Use GitHub Environments for staging vs production secrets:

```yaml
jobs:
  deploy-staging:
    environment: staging  # Uses staging secrets

  deploy-production:
    environment: production  # Uses production secrets
```

---

## Matrix Builds for Multiple Python Versions

Test across Python versions to catch compatibility issues:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false  # Don't cancel other jobs if one fails
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install and test
        run: |
          pip install -r requirements.txt
          pytest tests/
```

### Matrix for Multiple OS

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ["3.10", "3.11"]

runs-on: ${{ matrix.os }}
```

---

## Caching for Faster Builds

### Cache pip Dependencies

```yaml
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

### Cache DVC Data

```yaml
- name: Cache DVC data
  uses: actions/cache@v4
  with:
    path: .dvc/cache
    key: dvc-${{ hashFiles('data/*.dvc') }}
    restore-keys: |
      dvc-
```

### Cache Docker Layers

```yaml
- name: Build and push
  uses: docker/build-push-action@v5
  with:
    context: .
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Caching Impact

| Without Cache | With Cache | Savings |
|--------------|------------|---------|
| pip install: 90s | pip install: 5s | 85s |
| DVC pull: 120s | DVC pull: 10s | 110s |
| Docker build: 180s | Docker build: 30s | 150s |
| **Total: 390s** | **Total: 45s** | **88% faster** |

---

## Artifact Storage Between Jobs

Jobs run on separate runners. Use artifacts to pass data between them:

```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python train.py --output model.pkl

      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: model.pkl
          retention-days: 5

  validate:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Download model artifact
        uses: actions/download-artifact@v4
        with:
          name: trained-model

      - name: Validate model
        run: python validate.py --model model.pkl
```

---

## Branch Protection Rules

Enforce quality gates before merging to main:

### Required Settings

1. **Require pull request reviews**
   - At least 1 approval required
   - Dismiss stale reviews when new commits pushed

2. **Require status checks**
   - `lint-and-test` must pass
   - `validate-model` must pass
   - Require branches to be up to date

3. **Require conversation resolution**
   - All review comments must be resolved

### Setting Up via GitHub CLI

```bash
# Enable branch protection
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["lint-and-test","validate-model"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1}' \
  --field restrictions=null
```

### Branch Protection Visualization

```
main branch ──────────────────────────────────
                       ▲
                       │ PR + Checks + Review
                       │
feature branch ────────┘
    │
    ├── lint-and-test: PASS
    ├── validate-model: PASS
    ├── Code review: APPROVED
    └── Merge allowed
```

---

## Manual Approval Gates

Critical deployments should require human approval:

### GitHub Environments

```yaml
jobs:
  deploy-production:
    environment:
      name: production
      url: https://api.example.com

    steps:
      - name: Deploy
        run: ./deploy.sh
```

Configure in GitHub:
1. Settings > Environments > New environment
2. Name: `production`
3. Add required reviewers
4. Set deployment branch rules (only `main`)

### Approval Flow

```
Pipeline runs:
  [lint] -> [test] -> [validate] -> [build] -> [deploy-staging]
                                                      │
                                                      ▼
                                            [Waiting for approval]
                                                      │
                                        (Reviewer approves in GitHub)
                                                      │
                                                      ▼
                                              [deploy-production]
```

---

## Rollback Strategies

When deployments go wrong:

### Strategy 1: Revert Commit

```bash
# Revert the problematic commit
git revert HEAD
git push origin main
# Pipeline runs again with reverted code
```

### Strategy 2: Re-deploy Previous Version

```yaml
# Workflow dispatch with version input
on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Image tag to deploy'
        required: true

jobs:
  rollback:
    steps:
      - name: Deploy specific version
        run: |
          kubectl set image deployment/api \
            api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ inputs.version }}
```

### Strategy 3: Blue-Green Deployment

```yaml
- name: Blue-green deploy
  run: |
    # Deploy to green (inactive) environment
    kubectl apply -f k8s/green-deployment.yaml

    # Run smoke tests against green
    python smoke_tests.py --endpoint $GREEN_URL

    # Switch traffic to green
    kubectl patch service/api -p '{"spec":{"selector":{"version":"green"}}}'

    # Green is now blue (active), old blue can be removed
```

### Strategy 4: Canary Deployment

```yaml
- name: Canary deploy
  run: |
    # Deploy new version to 10% of traffic
    kubectl apply -f k8s/canary-deployment.yaml

    # Monitor for 10 minutes
    sleep 600

    # Check error rates
    ERROR_RATE=$(curl -s "$METRICS_URL/error_rate")
    if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
      echo "Error rate too high, rolling back"
      kubectl delete -f k8s/canary-deployment.yaml
      exit 1
    fi

    # Promote canary to full deployment
    kubectl apply -f k8s/full-deployment.yaml
```

---

## Exercises

### Exercise 4.1.1: Create Your First Workflow

Create `.github/workflows/test.yml` that:
1. Triggers on push to any branch
2. Sets up Python 3.11
3. Installs dependencies
4. Runs pytest

<details>
<summary>Solution</summary>

```yaml
name: Tests

on: push

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/ -v
```

</details>

### Exercise 4.1.2: Add Model Validation Gate

Modify the workflow to:
1. Download the model using DVC
2. Run the validation script
3. Fail the pipeline if accuracy < 0.85

<details>
<summary>Hint</summary>

Add a step that runs:
```bash
python scripts/validate_model.py \
  --model-path models/sentiment_model.pkl \
  --test-data data/test.csv \
  --min-accuracy 0.85
```

The script exits with code 1 if validation fails, which stops the pipeline.

</details>

### Exercise 4.1.3: Set Up Branch Protection

Using the GitHub CLI or web interface:
1. Protect the `main` branch
2. Require the `test` workflow to pass
3. Require 1 approval for PRs
4. Test by creating a PR with a failing test

### Exercise 4.1.4: Implement Caching

Add pip caching to reduce build times:
1. Add the cache action before `pip install`
2. Use `hashFiles('requirements*.txt')` as the cache key
3. Compare build times with and without cache

---

## Key Takeaways

1. **ML CI/CD has three pipelines**: code (CI), model (training), and deployment (CD). They're interconnected but have different triggers and criteria.

2. **Model validation gates are essential**. Never deploy a model without verifying accuracy, latency, and per-class metrics meet thresholds.

3. **Cache aggressively**. pip dependencies, DVC data, and Docker layers. This can reduce build times by 80%+.

4. **Use environments for deployment control**. Staging deploys automatically; production requires manual approval.

5. **Branch protection enforces quality**. Require status checks and reviews before merging to main.

6. **Secrets never go in code**. Use GitHub Secrets, and scope them to environments when possible.

7. **Plan for rollback from day one**. Have a documented, tested process for reverting bad deployments.

8. **Matrix builds catch compatibility issues**. Test across Python versions if you support multiple.

---

## Next Steps

Your ML pipeline is now automated. When code is pushed, it's tested, validated, and deployed without manual intervention.

But automation isn't enough. How do you know the model is still performing well in production? How do you detect data drift before users complain?

Run `/start-4-2` to learn about **Monitoring and Observability** - tracking model performance, detecting drift, and building alerts for production ML systems.
