# PRD: Interactive MLOps Course (Claude Code-Powered)

## Description: What is it?

An interactive, git-based MLOps course where students learn production ML principles by working alongside Claude Code as a learning partner. Students clone a repo, run `claude`, and progress through modules using slash commands (`/start-1-1`, etc.).

The course teaches MLOps concepts (design → data → model → deployment → monitoring) through hands-on exercises where Claude guides them through building and deploying a real ML model to production.

Inspired by Made With ML's principles-first approach, but delivered through an AI-guided interactive format rather than static documentation.

---

## Problem: What problem is this solving?

**MLOps education is fragmented and overwhelming:**

1. **Scattered documentation** - Learners jump between Kubernetes docs, MLflow tutorials, cloud provider guides, and Medium posts. No unified learning path.

2. **Too theoretical OR too tactical** - Content is either abstract principles with no hands-on practice, or copy-paste tutorials with no conceptual grounding.

3. **No feedback loop** - Traditional tutorials can't answer follow-up questions. Learners get stuck, context-switch to Google/ChatGPT, lose momentum.

4. **Tool overload** - MLOps spans 50+ tools. Beginners don't know what matters vs. what's noise.

**The gap:** There's no interactive, principles-based MLOps curriculum that teaches *why* before *how* and lets learners ask questions in context.

---

## Why: How do we know this is a real problem and worth solving?

**Evidence:**

- Made With ML has 35k+ GitHub stars - massive demand for structured MLOps education
- "How do I get into MLOps?" is a perennial question on r/MachineLearning and Twitter/X
- MLOps bootcamps charge $5k-15k, signaling willingness to pay for structured learning
- Most ML projects fail in production (87% according to VentureBeat) - skills gap is real

**Why Claude Code specifically:**

- Students can ask "wait, why did we use this instead of X?" without leaving the terminal
- Claude can read their actual code and give contextual feedback
- The interactive format mirrors real-world pairing with senior engineers
- Slash commands create clear progression without overwhelming learners

**Why now:**

- Claude Code is mature enough for complex teaching workflows
- MLOps tooling has stabilized (the "modern data stack" for ML is clearer now)
- Remote/async learning is normalized post-COVID

---

## Success: How do we know if we've solved this problem?

**Primary metrics:**

- **Completion rate** - % of students who finish all modules (target: >40%, vs. typical MOOC 5-15%)
- **Deployment success** - % who successfully deploy a model to production by course end (target: >70%)

**Secondary metrics:**

- GitHub stars / forks (social proof)
- Time to complete (target: 10-15 hours total)
- Student NPS / qualitative feedback
- "Would you recommend this?" responses

**Qualitative success:**

- Students report feeling confident to deploy ML models at work
- Students understand *why* they're using specific tools, not just *how*
- Follow-up questions decrease over time (concepts are sticking)

---

## Audience: Who are we building for?

**Primary persona: The ML Engineer moving into Ops**

- Knows Python, pandas, sklearn/PyTorch
- Has trained models locally, never deployed to production
- Overwhelmed by Kubernetes, Docker, CI/CD, monitoring options
- Wants structured path, not another "awesome-mlops" link dump

**Secondary persona: The DevOps/SRE moving into ML**

- Knows infrastructure, containers, CI/CD
- Doesn't understand ML-specific concerns (model drift, feature stores, experiment tracking)
- Wants to understand the "ML" part of MLOps

**Not for:**

- Complete beginners (need Python/ML fundamentals first)
- Experts looking for advanced edge cases

---

## What: Roughly, what does this look like in the product?

### Course Structure

```
Level 1: Foundations
├── 1.1 Welcome & Setup
├── 1.2 MLOps Principles (why production ML is different)
├── 1.3 Your First Model (simple classifier)
└── 1.4 Packaging for Production (Docker basics)

Level 2: The ML Pipeline
├── 2.1 Data Versioning (DVC or similar)
├── 2.2 Experiment Tracking (MLflow/W&B)
├── 2.3 Model Registry
└── 2.4 Testing ML Code

Level 3: Deployment
├── 3.1 Model Serving Options (batch vs. real-time)
├── 3.2 Building an API (FastAPI + model)
├── 3.3 Containerization Deep Dive
└── 3.4 Deploying to Cloud (pick one: AWS/GCP/Azure)

Level 4: Production Operations
├── 4.1 CI/CD for ML
├── 4.2 Monitoring & Observability
├── 4.3 Model Drift & Retraining
└── 4.4 Putting It All Together
```

### Core Project

Students build and deploy a **text classification model** (e.g., sentiment analysis or intent detection) throughout the course:

- Level 1: Train locally, package in Docker
- Level 2: Add versioning, tracking, testing
- Level 3: Deploy as API to cloud
- Level 4: Add CI/CD, monitoring, drift detection

### Repo Structure (mirroring this PM course)

```
mlops-course/
├── course-structure.json          # Single source of truth
├── .claude/
│   ├── commands/                   # Slash commands (start-1-1, etc.)
│   ├── agents/                     # Reviewer personas (ML engineer, SRE, etc.)
│   └── SCRIPT_INSTRUCTIONS.md
├── lesson-modules/
│   ├── 1-foundations/
│   ├── 2-pipeline/
│   ├── 3-deployment/
│   └── 4-production/
├── project/                        # The model students build
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   └── ...
└── reference/                      # Cheat sheets, architecture diagrams
```

### Interactive Elements

- **Socratic checkpoints** - Claude asks "why do you think we version data separately from code?" before explaining
- **Hands-on exercises** - "Now run `docker build` and tell me what you see"
- **Debug scenarios** - "Your deployment failed. Here's the error log. What would you check first?"
- **Custom agents** - `/review-deployment` spins up an SRE persona to critique their setup

---

## How: What is the experiment plan?

### Phase 1: MVP (4-6 weeks)

Build Levels 1-2 only:
- Welcome + principles
- Basic model training
- Docker packaging
- Experiment tracking with MLflow

**Validate:**
- Can 5-10 beta testers complete it?
- Where do they get stuck?
- Is the Claude Code format working?

### Phase 2: Full Course (6-8 weeks)

Add Levels 3-4 based on Phase 1 feedback:
- Deployment to cloud
- CI/CD pipeline
- Monitoring setup

### Phase 3: Polish & Launch

- Add custom sub-agents for code review
- Create "company context" (fake ML startup scenario)
- Write reference docs / cheat sheets
- Public launch on GitHub

### Distribution

- Free, open-source on GitHub
- Optional: Paid "cohort" version with office hours
- Cross-promote with existing MLOps communities

---

## When: When does it ship and what are the milestones?

| Milestone | Deliverable |
|-----------|-------------|
| **M1** | Course architecture + Level 1 modules written |
| **M2** | Level 2 complete, 3-5 beta testers recruited |
| **M3** | Beta feedback incorporated, Levels 3-4 drafted |
| **M4** | Full course complete, custom agents added |
| **M5** | Public launch on GitHub |

---

## Open Questions

1. **Which cloud provider to focus on?** AWS (most jobs), GCP (better ML tooling), or cloud-agnostic?
2. **What model/dataset for the core project?** Needs to be simple enough to not distract, complex enough to be realistic
3. **How much infrastructure?** Do we have students set up real cloud accounts, or use local simulation?
4. **Monetization?** Pure open-source, or freemium with paid cohorts/support?

---

## References

- [Made With ML](https://madewithml.com/) - Inspiration for curriculum structure
- [Claude Code PM Course](https://github.com/carlvellotti/claude-code-pm-course) - Architecture reference
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) - Another MLOps curriculum approach
