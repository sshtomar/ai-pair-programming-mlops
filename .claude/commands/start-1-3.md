# Lesson 1.3: Your First Model

Read the lesson content from `lesson-modules/1-foundations/1-3-first-model.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"In Lesson 1.2, we talked about principles. Now we build. By the end of this lesson, you'll have a working sentiment classifier—but more importantly, you'll structure it for production, not just experimentation."

### 2. Socratic Question
Ask: "What's wrong with developing ML models in Jupyter notebooks for production?"

Expected answers: hard to version, no modularity, hidden state, can't test easily. Build on this.

### 3. Project Structure (10 min)
Walk through creating production-ready structure:

```
project/
├── src/
│   ├── __init__.py
│   ├── data.py        # Data loading and preprocessing
│   ├── features.py    # Feature engineering
│   ├── train.py       # Training logic
│   └── predict.py     # Inference logic
├── tests/
├── config/
│   └── config.yaml    # Hyperparameters, paths
└── requirements.txt
```

Have them create this structure.

### 4. Build the Classifier (25 min)
Guide them through implementing:
1. `data.py` - Load and split Amazon reviews dataset
2. `features.py` - TF-IDF vectorization
3. `train.py` - Train LogisticRegression, save model
4. `predict.py` - Load model, make predictions

Key teaching moments:
- Proper train/val/test splits
- Config-driven hyperparameters
- Model serialization with joblib

### 5. Run Training (5 min)
Have them run `python -m src.train` and verify outputs.

### 6. Review Code Structure
Ask: "How would you add a new feature type to this codebase?"
Ensure they understand the modularity.

### 7. Wrap Up
- Model works locally—but how do we ensure it runs anywhere?
- Preview: Docker in Lesson 1.4
- Next: `/start-1-4`

## Teaching Notes
- Let them write code, don't just paste
- When they struggle, give hints not solutions
- Emphasize separation of concerns
