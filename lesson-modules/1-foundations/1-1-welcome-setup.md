# Lesson 1.1: Welcome & Setup

## Learning Objectives

By the end of this lesson, students will:
1. Understand the course structure and how to navigate it
2. Have a verified development environment
3. Know what they'll build throughout the course

## Duration: 20 minutes

---

## Part 1: Course Overview

### What You'll Build

Throughout this course, you'll build a **production-ready sentiment classifier** for customer feedback. This isn't a toy project—by the end, you'll have:

- A trained model that predicts sentiment (positive/negative/neutral)
- Version-controlled data and experiments
- Automated testing and validation
- A deployed API serving predictions
- Monitoring for model drift
- CI/CD pipeline for updates

### Why This Project?

Text classification is simple enough to focus on MLOps practices, yet complex enough to demonstrate real production concerns: data preprocessing, model versioning, inference latency, and monitoring.

### Course Structure

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          MLOps LEARNING PATH                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

   LEVEL 1              LEVEL 2              LEVEL 3              LEVEL 4
   Foundations          Pipeline             Deployment           Production
   ───────────          ────────             ──────────           ──────────
   ┌─────────┐         ┌─────────┐         ┌─────────┐          ┌─────────┐
   │ 1.1     │         │ 2.1     │         │ 3.1     │          │ 4.1     │
   │ Setup ●─┼────────▶│ DVC     │────────▶│ Serving │─────────▶│ CI/CD   │
   └────┬────┘         └────┬────┘         └────┬────┘          └────┬────┘
        │                   │                   │                    │
        ▼                   ▼                   ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐          ┌─────────┐
   │ 1.2     │         │ 2.2     │         │ 3.2     │          │ 4.2     │
   │Principles│        │ MLflow  │         │ FastAPI │          │Monitor  │
   └────┬────┘         └────┬────┘         └────┬────┘          └────┬────┘
        │                   │                   │                    │
        ▼                   ▼                   ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐          ┌─────────┐
   │ 1.3     │         │ 2.3     │         │ 3.3     │          │ 4.3     │
   │ Model   │         │Registry │         │ Docker+ │          │ Drift   │
   └────┬────┘         └────┬────┘         └────┬────┘          └────┬────┘
        │                   │                   │                    │
        ▼                   ▼                   ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐          ┌─────────┐
   │ 1.4     │         │ 2.4     │         │ 3.4     │          │ 4.4     │
   │ Docker  │         │ Testing │         │ Cloud   │          │Capstone │
   └─────────┘         └─────────┘         └─────────┘          └─────────┘

   [Notebook] ────▶ [Container] ────▶ [Cloud API] ────▶ [Production System]
```

### Navigation Commands

- `/start-X-Y` - Begin lesson X.Y
- `/status` - Check your progress
- `/help-mlops` - Get help with concepts

---

## Part 2: Environment Setup

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Primary language |
| Docker | 20+ | Containerization |
| Git | 2.30+ | Version control |
| VS Code | Latest | Recommended editor |

### Verification Script

Run these commands to verify your setup:

```bash
# Python version
python --version  # Should be 3.9 or higher

# pip available
pip --version

# Docker running
docker --version
docker run hello-world

# Git configured
git --version
git config user.name
git config user.email
```

### Python Environment

Create a virtual environment for this course:

```bash
python -m venv mlops-env
source mlops-env/bin/activate  # On Windows: mlops-env\Scripts\activate
pip install --upgrade pip
```

### Initial Dependencies

```bash
pip install pandas scikit-learn numpy pytest
```

---

## Part 3: Project Structure

The `project/` directory is where you'll build your classifier:

```
project/
├── src/                  # Source code
│   ├── __init__.py
│   ├── data.py          # Data loading
│   ├── features.py      # Feature engineering
│   ├── train.py         # Training script
│   └── predict.py       # Inference
├── tests/               # Test suite
├── data/                # Datasets (gitignored)
├── models/              # Saved models (gitignored)
├── config/              # Configuration files
└── requirements.txt     # Dependencies
```

This structure separates concerns: data handling, feature engineering, training, and inference each have dedicated modules. This isn't just organization—it enables testing, versioning, and independent deployment of components.

---

## Part 4: First Interaction

### Verify Project Access

```bash
cd project
ls -la
```

You should see the directory structure with starter code already in place. The `src/` directory contains production-ready modules that we'll explore in Lesson 1.3.

### Explore the Starter Code

Take a moment to look at what's provided:
- `src/data.py` - Data loading utilities
- `src/features.py` - Text vectorization
- `src/train.py` - Training script
- `src/predict.py` - Prediction class

We'll understand each module in Lesson 1.3.

---

## Exercises

### Exercise 1.1.1: Environment Check
Run the verification script and fix any issues.

### Exercise 1.1.2: Explore Structure
List all files in the `project/` directory. What's the purpose of `__init__.py`?

### Exercise 1.1.3: First Commit
Initialize git and make your first commit:
```bash
git init
git add .
git commit -m "Initial project structure"
```

---

## Key Takeaways

1. This course builds one project progressively—each lesson adds capabilities
2. Production ML requires more than model accuracy: versioning, testing, deployment, monitoring
3. Project structure matters—modular code enables MLOps practices

---

## Next Steps

Run `/start-1-2` to learn why most ML projects fail in production—and how to avoid common pitfalls.
