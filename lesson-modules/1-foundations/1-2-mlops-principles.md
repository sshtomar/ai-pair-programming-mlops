# Lesson 1.2: MLOps Principles

## Learning Objectives

By the end of this lesson, students will:
1. Understand why most ML projects fail in production
2. Know the 5 pillars of MLOps: Design, Data, Model, Deploy, Monitor
3. Recognize technical debt patterns in ML systems

## Duration: 30 minutes

---

## Part 1: The Production ML Problem

### The Sobering Statistics

According to VentureBeat (2019), **87% of ML projects never make it to production** [1]. Gartner similarly reported that through 2022, only 53% of AI projects would move from prototype to production [2]. This isn't a technology problem—it's a systems problem. The gap between "model works in notebook" and "model serves predictions reliably" is vast.

### Why ML Projects Fail

| Failure Mode | Root Cause |
|--------------|------------|
| Works locally, fails in production | Environment differences, missing dependencies |
| Model degrades silently | No monitoring, data drift undetected |
| Can't reproduce results | No experiment tracking, data not versioned |
| Takes months to update | No CI/CD, manual deployment process |
| Wrong model in production | No registry, no versioning |

### The Notebook Trap

Jupyter notebooks are excellent for exploration. They're terrible for production. Key problems:

1. **Hidden state**: Cell execution order matters, but isn't captured
2. **No testing**: Notebooks can't be unit tested effectively
3. **Version control pain**: JSON format makes diffs unreadable
4. **No modularity**: Hard to import notebook functions elsewhere

---

## Part 2: The Five Pillars of MLOps

```
                         ╔═══════════════════╗
                         ║   PRODUCTION ML   ║
                         ╚═════════╤═════════╝
                                   │
     ┌─────────────┬───────────────┼───────────────┬─────────────┐
     │             │               │               │             │
     ▼             ▼               ▼               ▼             ▼
┌─────────┐  ┌─────────┐    ┌───────────┐   ┌─────────┐   ┌─────────┐
│ DESIGN  │  │  DATA   │    │   MODEL   │   │ DEPLOY  │   │ MONITOR │
├─────────┤  ├─────────┤    ├───────────┤   ├─────────┤   ├─────────┤
│Problem  │  │Version  │    │Track      │   │Serve    │   │Track    │
│Framing  │  │Quality  │    │Reproduce  │   │Scale    │   │Alert    │
│Metrics  │  │Lineage  │    │Registry   │   │Rollback │   │Retrain  │
└─────────┘  └─────────┘    └───────────┘   └─────────┘   └─────────┘
     │             │               │               │             │
     └─────────────┴───────────────┴───────────────┴─────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │  Reliable ML at   │
                         │      Scale        │
                         └───────────────────┘
```

### Pillar 1: Design

**Before writing code, answer these questions:**

- What business problem are we solving?
- How will we measure success? (Not model accuracy—business impact)
- What's the simplest baseline that might work?
- What are the latency/throughput requirements?
- Who are the stakeholders and how will they use predictions?

**Anti-patterns:**
- Starting with a complex model before trying logistic regression
- Optimizing accuracy when the business needs interpretability
- Building real-time serving when batch predictions suffice

### Pillar 2: Data

**Data is the hardest part of ML.** Production data pipelines must:

- Version datasets (not just code)
- Track data lineage (where did this feature come from?)
- Validate data quality (schema, distributions, missing values)
- Handle data freshness (when was this last updated?)

**The data versioning problem:**

```
# This doesn't work
git add training_data.csv  # File is 2GB, git chokes

# This doesn't work either
# training_data_v1.csv
# training_data_v2.csv
# training_data_v2_final.csv
# training_data_v2_final_FINAL.csv
```

We'll solve this with DVC in Lesson 2.1.

### Pillar 3: Model

**Beyond accuracy, production models need:**

- **Reproducibility**: Same data + same code = same model
- **Interpretability**: Why did the model predict X?
- **Efficiency**: Latency, memory, compute requirements
- **Fairness**: Does the model discriminate unfairly?

**Experiment tracking matters:**

Without tracking, you'll ask: "Which hyperparameters produced that good model from last Tuesday?" and have no answer. We'll implement MLflow in Lesson 2.2.

### Pillar 4: Deploy

**Serving predictions is different from training:**

| Training | Serving |
|----------|---------|
| Batch processing | Real-time or batch |
| Optimize for throughput | Often optimize for latency |
| Can take hours | Must respond in milliseconds |
| Run occasionally | Run continuously |
| Failure means retry | Failure means user impact |

**Deployment questions:**
- Batch inference or real-time API?
- How do we handle model updates without downtime?
- What happens when the model fails? Fallback behavior?
- How do we roll back a bad deployment?

### Pillar 5: Monitor

**Models degrade in ways software doesn't:**

- **Data drift**: Input distribution changes (users start using different words)
- **Concept drift**: Relationship between inputs and outputs changes (what "positive sentiment" means evolves)
- **Feedback loops**: Model predictions influence future training data

**What to monitor:**
- Prediction distributions (sudden changes = problem)
- Latency and error rates
- Feature distributions vs. training data
- Business metrics (did the model actually help?)

---

## Part 3: ML Technical Debt

The seminal paper "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., NeurIPS 2015) [3] identified ML-specific debt categories that every practitioner should know:

### Data Dependencies

> "Data dependencies are more insidious than code dependencies."

Your model depends on upstream data pipelines you don't control. When they change, your model may break silently.

### Configuration Debt

ML systems have many hyperparameters, thresholds, and feature flags. Without proper management:
- Nobody knows what settings are in production
- Experiments use different configs than production
- Rollbacks become impossible

### Pipeline Jungles

ML pipelines often evolve organically:

```
raw_data → clean_data → features_v1 → features_v2 → model
               ↓
          other_features → merged_features → model_v2
                                 ↓
                          yet_more_features → model_v3
```

This becomes unmaintainable. We'll build clean pipelines instead.

---

## Part 4: Design Review Exercise

### The Scenario

A startup wants to build a customer feedback classifier:
- Input: Customer reviews (text)
- Output: Sentiment (positive/negative/neutral)
- Use case: Route complaints to support team, positive reviews to marketing

### Design Questions

Work through these questions:

1. **Success metric**: Accuracy? Or something else?
   - Consider: What's the cost of a false negative (missing a complaint)?
   - Consider: What's the cost of a false positive (routing positive review to support)?

2. **Baseline**: What's the simplest approach that might work?
   - Keyword matching? ("angry", "terrible" → negative)
   - Why might this fail?

3. **Latency requirements**: Real-time or batch?
   - How quickly do complaints need routing?

4. **Data source**: Where do reviews come from?
   - How often are new reviews generated?
   - What's the labeling strategy?

---

## Exercises

### Exercise 1.2.1: Failure Mode Analysis

Look at `reference/bad-ml-project/` (we'll create this structure):

```
bad_project/
├── model_final_v2_FIXED.pkl
├── train.ipynb
├── train_copy.ipynb
├── train_old.ipynb
├── data.csv
├── data_cleaned.csv
├── data_cleaned_v2.csv
└── predictions.py
```

List 5 problems with this structure and which MLOps pillar each violates.

### Exercise 1.2.2: Design Document

Write a 1-paragraph design document for the sentiment classifier answering:
- What problem does it solve?
- How will success be measured?
- What's the MVP (minimum viable product)?

---

## Key Takeaways

1. **Most ML projects fail not from bad models but from bad systems**
2. **The 5 pillars—Design, Data, Model, Deploy, Monitor—must all be addressed**
3. **ML technical debt compounds faster than regular software debt**
4. **Start with the simplest solution that could work**

---

## Next Steps

Run `/start-1-3` to build your first sentiment classifier with proper production structure.

---

## References

[1] VentureBeat. "Why do 87% of data science projects never make it into production?" (2019). https://venturebeat.com/ai/why-do-87-of-data-science-projects-never-make-it-into-production/

[2] Gartner. "Gartner Says Nearly Half of CIOs Are Planning to Deploy Artificial Intelligence" (2018). https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence

[3] Sculley, D., et al. "Hidden Technical Debt in Machine Learning Systems." Advances in Neural Information Processing Systems 28 (NeurIPS 2015). https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
