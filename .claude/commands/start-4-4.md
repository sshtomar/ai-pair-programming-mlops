# Lesson 4.4: Putting It All Together (Capstone)

Read the lesson content from `lesson-modules/4-production/4-4-capstone.md` and guide the student through it.

## Lesson Flow

### 1. Opener (3 min)
"You've learned the pieces. Now let's see the complete picture—and build it."

Acknowledge their journey:
- Level 1: Built and containerized a model
- Level 2: Added versioning, tracking, testing
- Level 3: Created and deployed an API
- Level 4: Automated with CI/CD and monitoring

### 2. Socratic Question
Ask: "If you had to explain MLOps to a colleague in one sentence, what would you say?"

Use their answer to frame the capstone: MLOps is the discipline of reliably delivering ML systems to production and keeping them healthy.

### 3. Architecture Review (15 min)
Draw the complete system together:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEVELOPMENT                               │
├─────────────────────────────────────────────────────────────────┤
│  Code (Git)  ←→  Data (DVC)  ←→  Experiments (MLflow)           │
│       ↓              ↓                  ↓                        │
│  Tests (pytest)   Validation      Model Registry                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         CI/CD                                    │
├─────────────────────────────────────────────────────────────────┤
│  GitHub Actions: Lint → Test → Build → Validate → Deploy        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       PRODUCTION                                 │
├─────────────────────────────────────────────────────────────────┤
│  Container (Docker)  →  Cloud (Cloud Run)  →  Users             │
│       ↓                      ↓                                   │
│  Health Checks         Monitoring (Prometheus/Logs)             │
│                              ↓                                   │
│                        Drift Detection → Alerts → Retrain        │
└─────────────────────────────────────────────────────────────────┘
```

Ask them to explain each component and connection.

### 4. Capstone Exercise: End-to-End (45 min)
Challenge: Simulate a full MLOps cycle.

**Part 1: Make a Model Change (10 min)**
- Modify preprocessing or model hyperparameters
- Run experiments with MLflow
- Choose the best model

**Part 2: Update and Test (10 min)**
- Update the training code
- Run the test suite
- Commit with DVC tracking

**Part 3: CI/CD Validation (10 min)**
- Push to a feature branch
- Watch the CI pipeline run
- Address any failures

**Part 4: Deploy and Monitor (15 min)**
- Merge to main
- Verify deployment succeeded
- Check logs and metrics
- Run a drift check

Guide them through, but let them drive.

### 5. Production Checklist Review (10 min)
Walk through the production readiness checklist:

**Code Quality**
- [ ] Linting passes
- [ ] Tests pass (unit, integration, behavioral)
- [ ] Type hints used
- [ ] Error handling complete

**Model Quality**
- [ ] Performance exceeds baseline
- [ ] Behavioral tests pass
- [ ] Model registered with metadata

**Operational Readiness**
- [ ] Container builds and runs
- [ ] Health check endpoint works
- [ ] Logging configured
- [ ] Metrics exposed
- [ ] Alerts defined

**Deployment**
- [ ] CI/CD pipeline functional
- [ ] Rollback procedure documented
- [ ] Drift monitoring active

Ask: "What would you add to this checklist for your specific use case?"

### 6. What's Next (10 min)
Point to advanced topics they can explore:

**Immediate Next Steps**
- Add authentication to the API
- Set up a proper monitoring dashboard
- Implement canary deployments

**Advanced MLOps**
- Feature stores (Feast, Tecton)
- ML platforms (Kubeflow, MLflow + Kubernetes)
- Advanced testing (shadow mode, A/B testing)
- Multi-model systems

**Specialization Paths**
- ML Platform Engineer
- MLOps/DevOps for ML
- Applied ML Engineer with ops skills

### 7. Course Wrap Up (5 min)
Celebrate completion:
- They've built a production ML system
- They understand the full lifecycle
- They have patterns to apply to future projects

Final advice:
- Start simple, add complexity as needed
- Automate what hurts, not everything
- Monitor what matters, alert on what's actionable
- Document decisions, not just code

Encourage them:
- Build something real with these skills
- Share what they learned
- Come back to the course materials as reference

## Teaching Notes
- This is a celebration—they made it through the course
- Let them struggle with the capstone, but help if stuck
- Focus on connections between lessons, not new content
- Leave them confident to continue learning independently
- Consider asking for feedback on the course
