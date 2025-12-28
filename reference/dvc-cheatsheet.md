# DVC Cheatsheet

## Quick Reference

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in a git repo |
| `dvc add data.csv` | Track a file with DVC |
| `dvc push` | Upload tracked files to remote storage |
| `dvc pull` | Download tracked files from remote |
| `dvc status` | Show changed files |
| `dvc diff` | Show diff between commits |
| `dvc repro` | Reproduce a pipeline |
| `dvc dag` | Visualize pipeline DAG |
| `dvc remote add -d name path` | Add default remote storage |

## Common Workflows

### Initial Setup

```bash
# Initialize DVC in existing git repo
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Configure remote storage (S3 example)
dvc remote add -d myremote s3://mybucket/dvc-store
git add .dvc/config
git commit -m "Configure DVC remote"
```

### Track Data Files

```bash
# Track a data file
dvc add data/raw/reviews.csv

# Git tracks the .dvc file, not the data
git add data/raw/reviews.csv.dvc data/raw/.gitignore
git commit -m "Add raw reviews dataset"

# Push data to remote
dvc push
```

### Reproduce Experiments

```bash
# Pull data for a specific git commit
git checkout v1.0
dvc pull

# Reproduce pipeline from scratch
dvc repro

# Reproduce specific stage
dvc repro train
```

### Define a Pipeline (dvc.yaml)

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/reviews.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false
```

### Compare Experiments

```bash
# Show metrics across experiments
dvc metrics show

# Compare with another branch
dvc metrics diff main

# Show params across experiments
dvc params diff
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `DVC is not initialized` | Run `dvc init` in git repo root |
| `No remote storage specified` | Run `dvc remote add -d name path` |
| `file not in cache` | Run `dvc fetch` or `dvc pull` |
| `changed deps/outs` | Run `dvc repro` to update |
| `locked stages` | Delete `dvc.lock` and rerun |
| `ERROR: unexpected files` | Add to `.dvcignore` or track with DVC |

## Best Practices

1. **Always commit .dvc files**: They're your data version pointers
2. **Use dvc.yaml for pipelines**: Makes experiments reproducible
3. **Separate raw and processed data**: Track both but in different directories
4. **Tag releases**: `git tag v1.0 && dvc push` for model versions
5. **Use params.yaml**: Centralize hyperparameters for easy comparison
