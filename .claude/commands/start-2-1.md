# Lesson 2.1: Data Versioning

Read the lesson content from `lesson-modules/2-pipeline/2-1-data-versioning.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"You've trained a model. Three months later, you need to reproduce the results. But the training data changed. Can you?"

### 2. Socratic Question
Ask: "Why can't we just version data with git like we do code?"

Expected: large files, binary formats, storage limits. Guide them to understand why specialized tools exist.

### 3. DVC Introduction (10 min)
Cover:
- DVC = "git for data"
- Stores metadata in git, data in remote storage
- Links data versions to code versions
- Works with any storage backend (S3, GCS, local)

### 4. Initialize DVC (10 min)
Walk through:
```bash
pip install dvc
dvc init
```

Explain what was created:
- `.dvc/` directory
- `.dvcignore`

### 5. Version the Dataset (15 min)
Guide them:
```bash
dvc add project/data/reviews.csv
git add project/data/reviews.csv.dvc project/data/.gitignore
git commit -m "Track training data with DVC"
```

Explain the `.dvc` file contents—it's just metadata pointing to the actual data.

### 6. Simulate Data Update (10 min)
Exercise:
1. Modify the dataset (add rows or change values)
2. Run `dvc add` again
3. See how the hash changes
4. Commit both code and .dvc file

Ask: "If you checkout an old git commit, how do you get the matching data?"
Answer: `dvc checkout`

### 7. Remote Storage (optional, 5 min)
Show how to configure remote storage if they want persistent storage beyond local.

### 8. Wrap Up
- Data is now versioned alongside code
- Every git commit can reproduce exact data state
- Preview: Lesson 2.2 adds experiment tracking
- Next: `/start-2-2`

## Teaching Notes
- DVC can be confusing at first—use analogies to git
- Emphasize the hash-based content addressing
- If storage permissions are tricky, skip remote for now
