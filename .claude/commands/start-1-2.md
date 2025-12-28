# Lesson 1.2: MLOps Principles

Read the lesson content from `lesson-modules/1-foundations/1-2-mlops-principles.md` and guide the student through it.

## Lesson Flow

### 1. Opener (3 min)
Reference what they learned in 1.1. Ask: "Why do you think 87% of ML projects never make it to production?"

Let them speculate before presenting the research.

### 2. The 5 Pillars (15 min)
Cover each pillar interactively:

**Design**
- Ask: "What should you figure out BEFORE you start training models?"
- Cover: problem framing, success metrics, baseline models

**Data**
- Ask: "How is data management for ML different from regular software?"
- Cover: versioning, lineage, quality monitoring

**Model**
- Ask: "Beyond accuracy, what else matters for a production model?"
- Cover: reproducibility, interpretability, resource requirements

**Deploy**
- Ask: "What's the difference between 'model.predict()' and a production API?"
- Cover: serving strategies, infrastructure, scaling

**Monitor**
- Ask: "Your model is deployed. What could go wrong over time?"
- Cover: drift, performance degradation, feedback loops

### 3. Technical Debt Exercise (10 min)
Show them `reference/bad-ml-project/` (a poorly structured example). Have them identify problems.

### 4. Design Review (5 min)
Introduce the sentiment classifier project requirements. Have them propose success metrics.

### 5. Wrap Up
- Connect pillars to upcoming lessons
- Emphasize: we'll implement all 5 pillars in this course
- Next: `/start-1-3` to build the first model

## Teaching Notes
- This is conceptual—keep energy up with frequent questions
- Use real examples of ML failures (Knight Capital, etc.)
- Make them articulate ideas before you do
