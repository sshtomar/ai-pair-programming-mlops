# Lesson 4.4: Putting It All Together

## Learning Objectives

By the end of this lesson, students will:
1. Review the complete MLOps architecture built throughout the course
2. Identify gaps and areas for improvement in their implementation
3. Understand the MLOps maturity model and where this course lands
4. Have a clear roadmap for continued learning

## Duration: 60 minutes

---

## Part 1: Architecture Review

### What We Built

Over four levels, you constructed a production ML system. Here's the complete architecture:

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE MLOPS ARCHITECTURE                               ║
╚═════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ DEVELOPMENT LAYER                                                            │
│                                                                              │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐         │
│    │   Git    │     │   DVC    │     │  MLflow  │     │  pytest  │         │
│    │ ──────── │     │ ──────── │     │ ──────── │     │ ──────── │         │
│    │  Code    │     │  Data    │     │Experiment│     │  Tests   │         │
│    │ Version  │     │ Version  │     │ Tracking │     │Behavioral│         │
│    └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘         │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                          │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CI/CD LAYER (GitHub Actions)                                                 │
│                                                                              │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│    │  Lint   │───>│  Test   │───>│Validate │───>│  Build  │───>│ Deploy  │ │
│    │         │    │         │    │  Model  │    │ Docker  │    │         │ │
│    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────┬────┘ │
│                                                                      │      │
└──────────────────────────────────────────────────────────────────────┼──────┘
                                                                       │
                    ┌──────────────────────────────────────────────────┤
                    │                                                  │
                    ▼                                                  ▼
┌────────────────────────────────────┐    ┌────────────────────────────────────┐
│ MODEL REGISTRY                      │    │ CONTAINER REGISTRY                  │
│ ┌────────────────────────────────┐ │    │ ┌────────────────────────────────┐ │
│ │         MLflow Registry         │ │    │ │    Docker Hub / GHCR / ECR     │ │
│ │  ┌─────────┐  ┌─────────┐      │ │    │ │  ┌─────────┐  ┌─────────┐     │ │
│ │  │ Staging │  │  Prod   │      │ │    │ │  │ v1.0.0  │  │ v1.1.0  │     │ │
│ │  │  v1.1   │  │  v1.0   │      │ │    │ │  │  Image  │  │  Image  │     │ │
│ │  └─────────┘  └─────────┘      │ │    │ │  └─────────┘  └─────────┘     │ │
│ └────────────────────────────────┘ │    │ └────────────────────────────────┘ │
└────────────────────────────────────┘    └──────────────────┬─────────────────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SERVING LAYER (Cloud)                                                        │
│                                                                              │
│    ┌───────────────────────────────────────────────────────────────────┐    │
│    │                     FastAPI Application                            │    │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │    │
│    │  │ /predict │  │ /health  │  │ /metrics │  │ Model Inference  │  │    │
│    │  │ endpoint │  │ endpoint │  │ endpoint │  │                  │  │    │
│    │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │    │
│    └───────────────────────────────────────────────────────────────────┘    │
│                                         │                                    │
└─────────────────────────────────────────┼────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MONITORING LAYER                                                             │
│                                                                              │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│    │  Prometheus  │───>│   Grafana    │    │  Alerting    │                 │
│    │   Metrics    │    │  Dashboards  │    │ (Slack/PD)   │                 │
│    └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                              │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│    │  Structured  │───>│     Log      │    │    Drift     │                 │
│    │    Logs      │    │  Aggregation │    │  Detection   │                 │
│    └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              ║ DATA FLOW ║
                              ╚═══════════╝
     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
     │Raw Data  │───>│ DVC      │───>│ Training │───>│ Model    │
     │(CSV/JSON)│    │ Tracked  │    │ Pipeline │    │ Artifact │
     └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                          │
                     ┌────────────────────────────────────┘
                     │
                     ▼
     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
     │ Registry │───>│ Docker   │───>│  Cloud   │───>│Predictions│
     │(MLflow)  │    │ Container│    │ Deploy   │    │  (API)   │
     └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

```
+------------------------------------------------------------------+
|                         MLOps Architecture                        |
+------------------------------------------------------------------+

                          +-----------------+
                          |   Git + GitHub  |
                          |  (Code Version) |
                          +--------+--------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
          v                        v                        v
+------------------+    +------------------+    +------------------+
|       DVC        |    |     MLflow       |    |    Pre-commit    |
| (Data Versioning)|    | (Experiments &   |    |    (Quality)     |
|                  |    |  Model Registry) |    |                  |
+--------+---------+    +--------+---------+    +--------+---------+
         |                       |                       |
         v                       v                       v
+------------------------------------------------------------------+
|                      Training Pipeline                            |
|  +------------+   +------------+   +------------+   +-----------+ |
|  | Load Data  |-->| Preprocess |-->| Train Model|-->| Evaluate  | |
|  +------------+   +------------+   +------------+   +-----------+ |
+----------------------------------+-------------------------------+
                                   |
                                   v
                          +--------+--------+
                          |  Model Artifact |
                          |  (Versioned)    |
                          +--------+--------+
                                   |
          +------------------------+------------------------+
          |                                                 |
          v                                                 v
+------------------+                              +------------------+
|     Docker       |                              |    pytest        |
| (Reproducibility)|                              | (Test Suite)     |
+--------+---------+                              +------------------+
         |
         v
+------------------+
|     FastAPI      |
|  (Serving API)   |
+--------+---------+
         |
         v
+------------------+
| GitHub Actions   |
|  (CI/CD)         |
+--------+---------+
         |
         v
+------------------+
|  Cloud Deploy    |
| (AWS/GCP/Azure)  |
+--------+---------+
         |
         v
+------------------+
|   Monitoring     |
| - Health checks  |
| - Logging        |
| - Drift detect   |
+------------------+
```

### Data Flow Summary

```
Raw Data                 Processed Data              Model                   Predictions
   |                          |                        |                          |
   v                          v                        v                          v
[reviews.csv] --> [DVC tracks] --> [features] --> [trained.pkl] --> [API] --> [sentiment]
                       |                               |                          |
                       v                               v                          v
                 [Remote Storage]               [MLflow Registry]          [Monitoring]
                 (S3/GCS bucket)               (Stage: Production)        (Drift alerts)
```

---

## Part 2: Component Checklist

### What's In Place

| Component | Tool | Status |
|-----------|------|--------|
| Code versioning | Git/GitHub | Implemented |
| Data versioning | DVC | Implemented |
| Experiment tracking | MLflow | Implemented |
| Model registry | MLflow | Implemented |
| Unit tests | pytest | Implemented |
| Behavioral tests | pytest | Implemented |
| Data validation | pytest/custom | Implemented |
| Containerization | Docker | Implemented |
| API serving | FastAPI | Implemented |
| CI/CD pipeline | GitHub Actions | Implemented |
| Cloud deployment | Provider of choice | Implemented |
| Health monitoring | Custom endpoints | Implemented |
| Basic logging | Python logging | Implemented |
| Drift detection | Statistical tests | Implemented |

### What's Partially Implemented

| Component | Current State | Production Enhancement |
|-----------|--------------|------------------------|
| Monitoring | Basic health checks | Full observability stack (Prometheus/Grafana) |
| Logging | Local/basic | Centralized logging (ELK, CloudWatch) |
| Alerting | Manual checks | Automated alerts (PagerDuty, Slack) |
| A/B testing | Not implemented | Feature flags, experiment infrastructure |
| Rollback | Manual | Automated rollback on failure |

### What's Missing (By Design)

We deliberately did not cover these production components:

| Component | Why Omitted | When You Need It |
|-----------|-------------|------------------|
| Feature store | Complexity for beginner course | When sharing features across models |
| Kubernetes | Requires significant DevOps knowledge | When scaling beyond single container |
| Orchestration (Airflow/Kubeflow) | Overkill for simple pipelines | When managing complex DAGs |
| Multi-model serving | Advanced topic | When serving multiple models |
| GPU inference | Hardware-specific | When latency matters for large models |
| Model compression | Specialized optimization | When deployment size/speed matters |

---

## Part 3: The MLOps Maturity Model

Google's MLOps maturity model defines four levels. Understanding where you are helps plan next steps.

### Level 0: No MLOps (Manual Everything)

```
Characteristics:
- Jupyter notebooks in production
- Manual model training and deployment
- No versioning of data or models
- No monitoring
- "It works on my machine" syndrome

Team structure: Data scientist does everything
Deployment frequency: Rarely (quarterly or less)
Recovery time: Days to weeks
```

**Most organizations start here.** The 87% failure rate comes from staying here too long.

### Level 1: DevOps but Not MLOps

```
Characteristics:
- Code is versioned (Git)
- CI/CD exists for application code
- But: Data not versioned
- But: Experiments not tracked
- But: Models deployed manually
- But: No ML-specific monitoring

Team structure: Data scientists + DevOps (separate)
Deployment frequency: Monthly
Recovery time: Hours to days
```

**The gap:** DevOps practices apply, but ML-specific concerns (data drift, model performance) are ignored.

### Level 2: ML Pipeline Automation

```
Characteristics:
- Automated training pipelines
- Data and model versioning (DVC, MLflow)
- Experiment tracking
- Automated testing including behavioral tests
- Model registry with stages
- Automated deployment from registry
- Basic monitoring

Team structure: ML Engineers emerge as a role
Deployment frequency: Weekly
Recovery time: Hours
```

**This is where this course lands you.** You have the foundation for reliable, reproducible ML systems.

### Level 3: Full MLOps

```
Characteristics:
- Continuous training (automatic retraining on data changes)
- Feature stores for shared features
- Advanced A/B testing and experimentation
- Full observability (metrics, logs, traces)
- Automated rollback and recovery
- Sophisticated drift detection and alerts
- Multi-model orchestration

Team structure: Dedicated ML platform team
Deployment frequency: Daily or continuous
Recovery time: Minutes
```

**This requires a platform team** and significant infrastructure investment. Most organizations don't need this immediately.

### Where You Are Now

```
+-------+-------+-------+-------+
|   0   |   1   |   2   |   3   |
| No    | DevOps| ML    | Full  |
| MLOps | Only  |Pipeline|MLOps |
+-------+-------+-------+-------+
                  ^
                  |
            [YOU ARE HERE]
```

You've moved from Level 0 to Level 2. This represents the highest-value transition:

- Level 0 to 1: Modest improvement (code versioning helps but doesn't solve ML problems)
- **Level 1 to 2: Massive improvement** (reproducibility, tracking, automated deployment)
- Level 2 to 3: Incremental improvement (optimization, scale)

---

## Part 4: Self-Assessment

Rate your understanding of each pillar. Be honest--this identifies where to focus next.

### The Five Pillars Assessment

Rate each from 1 (need more practice) to 5 (could teach this):

```
PILLAR 1: DESIGN
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Problem framing and success metrics
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Baseline selection
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Requirements gathering (latency, throughput)
+---+---+---+---+---+

PILLAR 2: DATA
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Data versioning with DVC
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Data validation and quality checks
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Feature engineering practices
+---+---+---+---+---+

PILLAR 3: MODEL
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Experiment tracking with MLflow
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Model registry and versioning
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Behavioral testing
+---+---+---+---+---+

PILLAR 4: DEPLOY
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Docker containerization
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  FastAPI serving
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  CI/CD with GitHub Actions
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Cloud deployment
+---+---+---+---+---+

PILLAR 5: MONITOR
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Health check endpoints
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Logging and observability
+---+---+---+---+---+
| 1 | 2 | 3 | 4 | 5 |  Drift detection concepts
+---+---+---+---+---+
```

### Interpreting Your Scores

- **Mostly 4-5:** You've internalized the concepts. Focus on applying them to new problems.
- **Mostly 3:** Solid foundation. Practice by building another project from scratch.
- **Some 1-2:** Revisit those specific lessons. Use `/start-X-Y` to review.

### Common Gaps by Background

**Coming from Data Science:**
- Often weaker on: Docker, CI/CD, cloud deployment
- Focus on: Lessons 1.4, 3.2, 4.1

**Coming from Software Engineering:**
- Often weaker on: Experiment tracking, behavioral testing, drift detection
- Focus on: Lessons 2.2, 2.4, 4.3

**Coming from DevOps/SRE:**
- Often weaker on: ML-specific testing, model registry concepts
- Focus on: Lessons 2.3, 2.4, 4.3

---

## Part 5: What We Didn't Cover

### Feature Stores

**What:** Centralized repository for ML features, enabling feature sharing across models and consistent feature computation between training and serving.

**Why we skipped it:** Adds significant complexity. Most teams don't need one until they have 3+ models sharing features.

**When you need it:**
- Multiple models use the same features
- Training/serving feature skew is causing production bugs
- Feature computation is expensive and needs caching

**Tools to explore:** Feast (open source), Tecton, AWS SageMaker Feature Store

### Kubernetes

**What:** Container orchestration platform for managing containerized applications at scale.

**Why we skipped it:** Significant learning curve. Docker Compose or single-container deployment handles most early-stage needs.

**When you need it:**
- Scaling beyond a single machine
- Managing multiple services
- Need auto-scaling based on load
- Multi-region deployment

**Learn it when:** Your Docker deployment hits scaling limits, or your organization mandates Kubernetes.

### Advanced Orchestration (Airflow, Kubeflow)

**What:** Workflow orchestration tools for managing complex multi-step pipelines.

**Why we skipped it:** Our pipeline is simple enough to run with Python scripts and CI/CD.

**When you need it:**
- Pipelines with many dependent steps
- Need to schedule and retry failed jobs
- Complex data dependencies between steps
- Multiple teams contributing to pipelines

**Tools to explore:** Apache Airflow, Kubeflow Pipelines, Prefect, Dagster

### GPU Inference

**What:** Serving model predictions using GPU acceleration for faster inference.

**Why we skipped it:** Our sentiment model is small. GPU adds cost and complexity.

**When you need it:**
- Large models (LLMs, large vision models)
- Latency requirements can't be met with CPU
- High throughput requirements

**Note:** Start with CPU. Only move to GPU when you have clear evidence you need it.

### Model Compression

**What:** Techniques to reduce model size and inference time (quantization, pruning, distillation).

**Why we skipped it:** Premature optimization. Our model is already fast.

**When you need it:**
- Edge deployment (mobile, IoT)
- Strict latency requirements
- Cost optimization at scale

---

## Part 6: Next Steps by Role

### For ML Engineers

Your path deepens the ML side of MLOps:

```
Immediate (0-3 months):
+---> Advanced experiment tracking (hyperparameter optimization)
+---> AutoML tools (understanding, not dependency)
+---> Model interpretability (SHAP, LIME)

Medium-term (3-6 months):
+---> Feature stores (Feast or managed)
+---> Advanced monitoring (custom drift metrics)
+---> Model compression and optimization

Long-term (6-12 months):
+---> ML system design for scale
+---> Multi-model architectures
+---> Continuous training pipelines
```

**Recommended resources:**
- "Designing Machine Learning Systems" by Chip Huyen
- MLflow documentation (advanced features)
- Feast documentation

### For DevOps/SRE Engineers

Your path deepens the infrastructure side:

```
Immediate (0-3 months):
+---> Kubernetes fundamentals
+---> Infrastructure as Code (Terraform)
+---> Advanced observability (Prometheus, Grafana)

Medium-term (3-6 months):
+---> Kubernetes for ML workloads
+---> GPU cluster management
+---> Multi-cloud deployment

Long-term (6-12 months):
+---> ML platform engineering
+---> Cost optimization for ML workloads
+---> Security for ML systems
```

**Recommended resources:**
- "Kubernetes Up & Running" by Burns et al.
- Terraform documentation
- Cloud provider ML service documentation

### For Data Scientists

Your path connects data science to production:

```
Immediate (0-3 months):
+---> Apply this course to your next project
+---> Refactor existing notebooks to production code
+---> Add testing to current models

Medium-term (3-6 months):
+---> Learn more Docker and deployment
+---> Collaborate with platform teams
+---> Implement monitoring for existing models

Long-term (6-12 months):
+---> Transition to ML Engineer role or
+---> Specialize in research with production awareness
```

**Recommended resources:**
- Practice deploying Kaggle competition solutions
- "The Pragmatic Programmer" for software engineering fundamentals
- Your organization's production systems (learn from what exists)

### Decision Tree: What to Learn Next

```
                    What's your biggest pain point?
                              |
         +--------------------+--------------------+
         |                    |                    |
    Scaling issues       Model quality        Deployment speed
         |                    |                    |
         v                    v                    v
   Learn Kubernetes    Learn advanced       Learn advanced
   and Terraform       testing & monitoring  CI/CD patterns
         |                    |                    |
         v                    v                    v
   Then: Feature       Then: Continuous      Then: GitOps and
   stores, GPU         training, AutoML      Infrastructure
   inference                                 as Code
```

---

## Part 7: Capstone Exercise

### The Challenge: End-to-End from Scratch

Build a new ML application using everything you've learned. Time: 2-4 hours.

**Choose a problem:**

Option A: **Spam Classifier**
- Input: Email text
- Output: spam/not_spam
- Find a dataset: SpamAssassin corpus or similar

Option B: **Movie Review Sentiment**
- Input: Movie review text
- Output: positive/negative
- Find a dataset: IMDB or similar

Option C: **Your Own Problem**
- Choose something relevant to your work
- Must have labeled text data

### Requirements Checklist

Your submission should include all of these:

**Repository Structure:**
```
[ ] project/
    [ ] src/
        [ ] __init__.py
        [ ] data.py
        [ ] features.py
        [ ] train.py
        [ ] predict.py
    [ ] tests/
        [ ] test_data.py
        [ ] test_model.py (behavioral tests)
    [ ] Dockerfile
    [ ] requirements.txt
    [ ] pyproject.toml or setup.py
[ ] .dvc/
[ ] .github/workflows/
    [ ] ci.yml
[ ] dvc.yaml
[ ] .pre-commit-config.yaml
[ ] README.md (brief, just setup instructions)
```

**MLOps Components:**
```
[ ] Data versioned with DVC
[ ] Remote storage configured (even if just local-remote)
[ ] Experiments tracked in MLflow
[ ] At least one model registered
[ ] Dockerfile builds successfully
[ ] API serves predictions
[ ] Health check endpoint works
[ ] At least 5 behavioral tests pass
[ ] CI pipeline runs on push
[ ] Pre-commit hooks configured
```

**Documentation:**
```
[ ] README has setup instructions
[ ] API endpoint documented
[ ] Model card with performance metrics
```

### Evaluation Criteria

Not a formal grade, but a self-check:

| Criterion | Check |
|-----------|-------|
| Can I reproduce the model from scratch? | Clone, run pipeline, get same model |
| Can I see experiment history? | MLflow shows all runs |
| Will I know if the model breaks? | Tests catch common failures |
| Can I deploy with one command? | Docker build and run works |
| Can I rollback if needed? | Previous model versions exist |

---

## Part 8: Course Completion

### What You've Accomplished

Over this course, you built:

1. **A reproducible training pipeline** that anyone can run
2. **Version-controlled data and models** that you can always recreate
3. **A tested model** with behavioral tests that catch subtle bugs
4. **A production API** that can serve predictions reliably
5. **Automated deployment** that reduces human error
6. **Monitoring infrastructure** that alerts you to problems

This is not a toy project. These are the same practices used by ML teams at successful companies.

### The Knowledge Gap You've Closed

```
Before this course:              After this course:

"Model works in notebook"   -->  "Model is deployed with CI/CD"
"Data is somewhere"         -->  "Data is versioned and tracked"
"I think this was the       -->  "I can see every experiment
 best hyperparameters"            in MLflow"
"Deployment is scary"       -->  "Deployment is a git push"
"Hope it keeps working"     -->  "I'll know if it breaks"
```

### Staying Current

MLOps evolves rapidly. Here's how to stay informed:

**Follow:**
- MLOps Community (mlops.community)
- Chip Huyen's blog
- Neptune.ai blog
- Weights & Biases blog

**Read:**
- "Designing Machine Learning Systems" (Chip Huyen)
- "Reliable Machine Learning" (Cathy Chen et al.)
- Google's ML best practices documentation

**Practice:**
- Apply these techniques to every new project
- Refactor one existing project to use proper MLOps
- Contribute to open-source MLOps tools

**Experiment:**
- Try new tools as they emerge
- But don't chase every new framework
- Evaluate based on your actual needs

### Community

You're not alone in this journey:

- **MLOps Community Slack:** Active discussion, job postings, tool comparisons
- **Local meetups:** Search for MLOps or ML Engineering meetups
- **Conferences:** MLOps World, ML Platform Summit
- **Your colleagues:** Find others interested in production ML

---

## Final Checklist

Before considering the course complete, verify:

```
LEVEL 1: FOUNDATIONS
[ ] Can explain the 5 pillars of MLOps
[ ] Have a working model training script
[ ] Dockerfile builds and runs
[ ] Understand why notebooks aren't production-ready

LEVEL 2: PIPELINE
[ ] DVC initialized and tracking data
[ ] Remote storage configured
[ ] MLflow tracking experiments
[ ] Model registered with version
[ ] Test suite with behavioral tests
[ ] Pre-commit hooks installed

LEVEL 3: DEPLOYMENT
[ ] FastAPI serving predictions
[ ] Health check endpoint works
[ ] Docker container runs API
[ ] Deployed to at least one cloud
[ ] Can access deployed endpoint

LEVEL 4: PRODUCTION
[ ] CI/CD pipeline runs on push
[ ] Tests run in pipeline
[ ] Deployment automated
[ ] Basic monitoring in place
[ ] Understand drift detection concepts
```

If any boxes are unchecked, revisit the relevant lesson with `/start-X-Y`.

---

## Closing

You started this course with a notebook and a dream of production ML. You're finishing with a complete, reproducible, deployable system.

The tools will change. The specific commands will evolve. But the principles remain:

- **Version everything** (code, data, models, config)
- **Test behavior**, not just accuracy
- **Automate the boring parts** (deployment, testing)
- **Monitor what matters** (not just uptime)
- **Make it reproducible** (for future you)

These principles separate hobby projects from production systems. You now have both the knowledge and the hands-on experience to build ML systems that actually work in the real world.

Go build something.

---

## Commands Reference

For future reference, here are all course commands:

```
/start-1-1  - Welcome & Setup
/start-1-2  - MLOps Principles
/start-1-3  - Your First Model
/start-1-4  - Docker Basics

/start-2-1  - Data Versioning with DVC
/start-2-2  - Experiment Tracking with MLflow
/start-2-3  - Model Registry
/start-2-4  - Testing ML Code

/start-3-1  - API with FastAPI
/start-3-2  - Cloud Deployment
/start-3-3  - Configuration Management
/start-3-4  - API Testing & Documentation

/start-4-1  - CI/CD Pipelines
/start-4-2  - Monitoring Basics
/start-4-3  - Drift Detection
/start-4-4  - Capstone (this lesson)

/status     - Check your progress
/help-mlops - Get help on concepts
```

---

*Course complete. Go ship some models.*
