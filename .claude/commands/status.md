# Course Progress Check

Read the course-structure.json and check the student's progress.

## Tasks

1. Check which lesson files exist in `project/` directory
2. Look for completion markers (e.g., Dockerfile present = Level 1 complete)
3. Summarize current progress with visual

## Progress Indicators

- **Level 1 complete**: `project/Dockerfile` exists and builds
- **Level 2 complete**: `.dvc/` directory exists, MLflow experiments logged
- **Level 3 complete**: `project/src/api.py` exists with FastAPI app
- **Level 4 complete**: `.github/workflows/` contains CI/CD config

## Response Format

Use this visual progress display:

```
╔═══════════════════════════════════════════════════════════════════╗
║                     MLOps COURSE PROGRESS                         ║
╚═══════════════════════════════════════════════════════════════════╝

   LEVEL 1              LEVEL 2              LEVEL 3              LEVEL 4
   Foundations          Pipeline             Deployment           Production

   [████] 4/4          [██░░] 2/4          [░░░░] 0/4          [░░░░] 0/4
   ✓ Setup             ✓ DVC                ○ Serving            ○ CI/CD
   ✓ Principles        ✓ MLflow             ○ FastAPI            ○ Monitor
   ✓ Model             ○ Registry           ○ Docker+            ○ Drift
   ✓ Docker            ○ Testing            ○ Cloud              ○ Capstone

═══════════════════════════════════════════════════════════════════════
   Progress: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  6/16 lessons (38%)
═══════════════════════════════════════════════════════════════════════

   Current: Level 2 - The ML Pipeline
   Next up: /start-2-3 - Model Registry
```

Replace ✓ with completed lessons, ○ with pending.
Replace █ with filled blocks based on completion.

Be encouraging but honest about progress.
