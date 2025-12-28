# Bad ML Project - Anti-patterns Example

This directory represents what a typical ML project looks like **without proper MLOps practices**. Use it as a teaching reference for why we need version control, experiment tracking, and proper tooling.

## The Files You'd Find Here

```
bad-ml-project/
├── model_final.pkl
├── model_final_v2.pkl
├── model_final_v2_FIXED.pkl
├── model_final_v2_FIXED_actually_works.pkl
├── model_prod_backup_dec15.pkl
├── train.ipynb
├── train_copy.ipynb
├── train_old.ipynb
├── train_johns_version.ipynb
├── data.csv
├── data_cleaned.csv
├── data_cleaned_v2.csv
├── data_cleaned_v2_final.csv
├── preprocess.py
├── preprocess_backup.py
├── config.py.bak
└── notes.txt
```

## What's Wrong Here

### 1. No Model Versioning

| Problem | Why It's Bad | MLOps Solution |
|---------|--------------|----------------|
| `model_final_v2_FIXED.pkl` | Which model is actually in production? | **MLflow Model Registry** with stages (Staging, Production) |
| Manual filename versioning | No metadata about training params or performance | **MLflow tracking** logs params, metrics, artifacts |
| `_backup_dec15` naming | No way to reproduce this model | **DVC** + git tags for exact reproducibility |

### 2. No Data Versioning

| Problem | Why It's Bad | MLOps Solution |
|---------|--------------|----------------|
| `data_cleaned_v2_final.csv` | Which data trained which model? | **DVC** tracks data versions alongside code |
| Manually copying data files | Can't reproduce past experiments | **DVC pipelines** define data transformations |
| Data scattered in project root | No clear lineage from raw to processed | **Directory structure** with raw/, processed/, features/ |

### 3. No Code Version Control

| Problem | Why It's Bad | MLOps Solution |
|---------|--------------|----------------|
| `train_copy.ipynb` | Which notebook has the right code? | **Git** - one version of truth |
| `preprocess_backup.py` | Can't trace changes or revert | **Git history** for all changes |
| `johns_version.ipynb` | Collaboration nightmare | **Git branches** for experiments |

### 4. No Experiment Tracking

| Problem | Why It's Bad | MLOps Solution |
|---------|--------------|----------------|
| `notes.txt` with hyperparameters | Impossible to compare experiments | **MLflow** logs all parameters |
| No recorded metrics | Don't know why v2 is "better" | **MLflow** tracks metrics over time |
| Lost training history | Can't explain model behavior | **MLflow** artifacts store everything |

### 5. No Reproducibility

| Problem | Why It's Bad | MLOps Solution |
|---------|--------------|----------------|
| No requirements.txt | "Works on my machine" syndrome | **Docker** for environment consistency |
| No random seeds logged | Can't reproduce results | **Config files** tracked in git |
| No pipeline definition | Manual multi-step process | **DVC pipelines** or Makefiles |

## The Real Cost

When something goes wrong in production:

- **Without MLOps**: "Which model is deployed? What data trained it? What were the hyperparameters? Let me check notes.txt... it says 'lr=0.01 or maybe 0.001, tried both'"

- **With MLOps**: `mlflow models describe -m models:/production-model/Production` gives you everything instantly

## How This Course Fixes It

| Lesson | What You'll Learn |
|--------|-------------------|
| 1.3-1.4 | Proper project structure, Docker packaging |
| 2.1 | DVC for data versioning |
| 2.2 | MLflow for experiment tracking |
| 2.3 | Testing to catch issues early |
| 3.x | Deployment with proper versioning |
| 4.x | CI/CD automation, monitoring |

---

*Look at this folder whenever you're tempted to skip proper versioning. Your future self will thank you.*
