# Lesson 4.1: CI/CD for ML

Read the lesson content from `lesson-modules/4-production/4-1-cicd-ml.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"You've deployed manually. What happens when you need to deploy 10 times a day? Or when 5 people are pushing changes?"

### 2. Socratic Question
Ask: "What should happen automatically when you push code to your ML repository?"

Expected: Tests run, linting, build container, deploy to staging, maybe deploy to prod. Guide them to think about the full pipeline, not just deployment.

### 3. CI/CD for ML vs Traditional Software (10 min)
What's different about ML pipelines:

**Traditional CI/CD**
- Code changes trigger pipeline
- Build, test, deploy

**ML CI/CD adds**
- Data validation
- Model training (sometimes)
- Model validation (performance thresholds)
- Model registry updates
- A/B testing / canary deployments

Show the expanded pipeline:
```
Code Push → Lint → Unit Tests → Build Container →
  → Integration Tests → Model Validation →
  → Push to Registry → Deploy to Staging →
  → Smoke Tests → Deploy to Production
```

### 4. GitHub Actions Workflow (20 min)
Create `.github/workflows/ml-pipeline.yml`:
```yaml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run linting
        run: |
          pip install ruff
          ruff check .

      - name: Run tests
        run: pytest tests/ -v

      - name: Validate model performance
        run: python scripts/validate_model.py

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build container
        run: docker build -t sentiment-api:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USER }} --password-stdin
          docker push sentiment-api:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy sentiment-api \
            --image sentiment-api:${{ github.sha }} \
            --region us-central1
```

Walk through each job and step.

### 5. Model Validation in CI (15 min)
Create `scripts/validate_model.py`:
```python
import joblib
import json
import sys

def validate_model():
    model = joblib.load("models/sentiment_model.pkl")

    # Test predictions work
    result = model.predict(["test text"])
    assert result is not None, "Model failed to predict"

    # Check against baseline metrics
    with open("models/metrics.json") as f:
        metrics = json.load(f)

    if metrics["accuracy"] < 0.85:
        print(f"Model accuracy {metrics['accuracy']} below threshold 0.85")
        sys.exit(1)

    print(f"Model validation passed: accuracy={metrics['accuracy']}")

if __name__ == "__main__":
    validate_model()
```

Ask: "What happens if a PR reduces model accuracy? Should it be blocked?"

### 6. Branch Protection and Quality Gates (10 min)
Set up branch protection:
- Require PR reviews
- Require status checks to pass
- No direct pushes to main

Discuss quality gates:
- Test coverage thresholds
- Model performance thresholds
- Security scanning

### 7. Wrap Up
- Every push is validated automatically
- Model quality is enforced in CI
- Deployment is consistent and reproducible
- Preview: Lesson 4.2 adds monitoring to see what happens after deployment
- Next: `/start-4-2`

## Teaching Notes
- GitHub Actions is most accessible; adapt to GitLab CI if needed
- Secrets management is crucial—never commit credentials
- Start simple, add complexity incrementally
- CI should catch the issues from Level 2 testing automatically
