# Lesson 2.1: Data Versioning with DVC

## Learning Objectives

By the end of this lesson, students will:
1. Understand why git alone fails for ML data
2. Set up DVC for data versioning
3. Link data versions to code versions

## Duration: 40 minutes

---

## Part 1: The Data Versioning Problem

### Why Git Fails for ML Data

Git was designed for source code: small text files that change incrementally. ML data breaks every assumption:

| Git Assumption | ML Data Reality |
|----------------|-----------------|
| Files are small (< 100MB) | Datasets are often gigabytes |
| Files are text | Images, audio, binary formats |
| Diffs are meaningful | Binary diffs are useless |
| Files change incrementally | New data often means full replacement |

### The Breaking Point

```bash
# This will fail or make git unusable
git add training_data.csv  # 500MB file

# Even if it works, your repo becomes:
# - Slow to clone
# - Expensive to host
# - Impossible to diff
```

GitHub enforces a 100MB file limit. GitLab has similar restrictions. This isn't a bug—it's protecting you from a broken workflow.

### The Naming Convention Anti-Pattern

Without proper versioning, teams resort to:

```
data/
├── reviews.csv
├── reviews_v2.csv
├── reviews_v2_cleaned.csv
├── reviews_v2_cleaned_FINAL.csv
├── reviews_v2_cleaned_FINAL_fixed.csv
└── reviews_USE_THIS_ONE.csv
```

Problems:
- No link between data version and code version
- No way to reproduce old experiments
- No audit trail of what changed
- Confusion about which version is current

---

## Part 2: DVC = Git for Data

### The Core Concept

DVC (Data Version Control) solves this with a simple insight: **store metadata in git, store data elsewhere**.

```
DVC Workflow: How Pointer Files Track Large Data
═══════════════════════════════════════════════════════════════════════════════

  YOUR PROJECT (Git Repository)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │   Source Code              .dvc Pointer Files (small, tracked by git)   │
  │   ┌──────────────┐         ┌────────────────────────────────────────┐   │
  │   │ train.py     │         │ data/reviews.csv.dvc                   │   │
  │   │ predict.py   │         │ ┌────────────────────────────────────┐ │   │
  │   │ config.yaml  │         │ │ outs:                              │ │   │
  │   │ ...          │         │ │   - md5: ab12cd34ef56...           │─┼───┼──┐
  │   └──────────────┘         │ │     size: 52428800                 │ │   │  │
  │                            │ │     path: reviews.csv              │ │   │  │
  │                            │ └────────────────────────────────────┘ │   │  │
  │                            └────────────────────────────────────────┘   │  │
  │   .gitignore: /data/reviews.csv  (actual data NOT in git)               │  │
  └─────────────────────────────────────────────────────────────────────────┘  │
                                                                               │
                                         Hash pointer ─────────────────────────┘
                                                │
                                                ▼
  REMOTE STORAGE (S3 / GCS / Azure / Local)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │   .dvc/cache/files/md5/                                                 │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │  ab/                                                            │   │
  │   │   └── 12cd34ef56...  ◄── Actual data file (50MB)                │   │
  │   │  cd/                     stored by content hash                 │   │
  │   │   └── 89ef01ab23...  ◄── Another version (same file, new data)  │   │
  │   └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  Key Insight: Git tracks the tiny .dvc file. The hash inside points to the
               actual data in remote storage. Change data → new hash → new pointer.
```

Git tracks the `.dvc` files (tiny metadata). The actual data lives in remote storage (S3, GCS, local directory, etc.).

### Content-Addressable Storage

DVC uses MD5 hashes to identify files:

```bash
# When you run: dvc add data/reviews.csv
# DVC calculates: md5("file contents") → "ab12cd34ef56..."

# The file is stored as:
# .dvc/cache/ab/12cd34ef56...
```

**Why this matters:**
- Identical files are stored once (deduplication)
- Any change creates a new hash (immutability)
- Hash in git = exact pointer to data version

This is the same principle behind git's object storage, Docker image layers, and IPFS.

---

## Part 3: Setting Up DVC

### Installation

```bash
pip install dvc

# For remote storage support (choose based on your cloud):
pip install dvc[s3]    # AWS S3
pip install dvc[gs]    # Google Cloud Storage
pip install dvc[azure] # Azure Blob Storage
```

Verify installation:

```bash
dvc version
```

Expected output:
```
DVC version: 3.x.x
Platform: Python 3.x.x on ...
```

### Initializing DVC

Navigate to your project directory:

```bash
cd project
dvc init
```

Expected output:
```
Initialized DVC repository.

You can now commit the changes to git.

+---------------------------------------------------------------------+
|                                                                     |
|        DVC has enabled anonymous aggregate usage analytics.         |
|     Read the analytics documentation (and hierarchical config)      |
|     if you want to disable it:                                      |
|             <https://dvc.org/doc/user-guide/analytics>              |
|                                                                     |
+---------------------------------------------------------------------+

What's next?
------------
- Check out the documentation: <https://dvc.org/doc>
- Get help and share ideas: <https://dvc.org/chat>
- Star us on GitHub: <https://github.com/iterative/dvc>
```

### DVC Directory Structure

After initialization:

```bash
ls -la .dvc/
```

```
.dvc/
├── config        # DVC configuration (remote storage settings)
├── .gitignore    # Ensures cache isn't committed to git
├── cache/        # Local cache of data files
└── tmp/          # Temporary files
```

Also created:
- `.dvcignore` - Like `.gitignore` but for DVC (patterns DVC should ignore)

### Commit the DVC Setup

```bash
git add .dvc .dvcignore
git commit -m "Initialize DVC for data versioning"
```

---

## Part 4: Tracking Data with DVC

### Create Sample Data

First, let's create training data to version:

```bash
mkdir -p data

# Create a sample dataset
cat > data/reviews.csv << 'EOF'
text,label
"Great product, love it!",positive
"Terrible experience, waste of money",negative
"It's okay, nothing special",neutral
"Best purchase I've ever made",positive
"Broke after one day",negative
"Does what it says",neutral
EOF
```

### Add Data to DVC

```bash
dvc add data/reviews.csv
```

Expected output:
```
To track the changes with git, run:

        git add data/.gitignore data/reviews.csv.dvc
```

### What Happened?

DVC created two files:

**1. `data/reviews.csv.dvc`** - The metadata file:

```bash
cat data/reviews.csv.dvc
```

```yaml
outs:
- md5: 7a3b8c9d0e1f2a3b4c5d6e7f8a9b0c1d
  size: 285
  hash: md5
  path: reviews.csv
```

This tiny file is what git tracks. It contains:
- `md5`: Content hash (the pointer to actual data)
- `size`: File size in bytes
- `path`: Original filename

**2. `data/.gitignore`** - Prevents git from tracking the data:

```bash
cat data/.gitignore
```

```
/reviews.csv
```

### The Cache

The actual data now lives in the DVC cache:

```bash
ls -la .dvc/cache/files/md5/
```

The file is stored by its hash, enabling:
- Multiple versions without duplication
- Verification that data hasn't been corrupted
- Sharing across machines via remote storage

### Commit the Changes

```bash
git add data/.gitignore data/reviews.csv.dvc
git commit -m "Add training data v1"
```

**Critical insight**: You just committed a pointer to your data. The data itself is not in git. When someone clones your repo, they get the `.dvc` file. They must run `dvc pull` to get the actual data.

---

## Part 5: Linking Data to Code Versions

### The Git + DVC Workflow

Every git commit can be associated with a specific data version:

```
git log --oneline
```

```
abc1234 Add training data v1         ← data/reviews.csv.dvc points to hash X
def5678 Initialize DVC               ← no data yet
ghi9012 Initial project structure    ← no data yet
```

When you checkout any commit, the `.dvc` file tells DVC exactly which data version belongs there.

### Simulating a Data Update

Let's add more training data:

```bash
# Append new reviews
cat >> data/reviews.csv << 'EOF'
"Customer service was unhelpful",negative
"Exceeded my expectations!",positive
"Arrived damaged",negative
"Perfect for my needs",positive
EOF
```

Track the updated file:

```bash
dvc add data/reviews.csv
```

Expected output:
```
To track the changes with git, run:

        git add data/reviews.csv.dvc
```

Check what changed:

```bash
cat data/reviews.csv.dvc
```

```yaml
outs:
- md5: 1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b   # NEW HASH
  size: 412                               # Larger size
  hash: md5
  path: reviews.csv
```

Commit the update:

```bash
git add data/reviews.csv.dvc
git commit -m "Add training data v2 (expanded)"
```

Now your git history shows:

```
git log --oneline
```

```
xyz7890 Add training data v2 (expanded)  ← hash 1e2f3a...
abc1234 Add training data v1             ← hash 7a3b8c...
def5678 Initialize DVC
```

---

## Part 6: Switching Between Data Versions

### Checkout Previous Data Version

To restore the original training data:

```bash
# Checkout the git commit
git checkout abc1234

# Restore the data that matches this commit
dvc checkout
```

Expected output:
```
M       data/reviews.csv
```

Your `data/reviews.csv` now contains the original 6 reviews.

### Return to Latest

```bash
git checkout main  # or master
dvc checkout
```

Data is back to the expanded version.

### The Power of This

Consider this scenario:
- You trained Model A on data v1 → achieved 82% accuracy
- You trained Model B on data v2 → achieved 79% accuracy
- You want to understand why performance dropped

With DVC:

```bash
git checkout <model-A-commit>
dvc checkout

# Now you have:
# - The exact code that trained Model A
# - The exact data that trained Model A
# - Full reproducibility
```

---

## Part 7: Remote Storage Configuration

### Why Remote Storage?

The local cache (`.dvc/cache/`) only exists on your machine. To collaborate:

1. **Share data across team members**
2. **Backup data off your machine**
3. **Enable CI/CD pipelines to access data**

### Configure a Local Remote (For Learning)

```bash
# Create a "remote" directory (simulating S3/GCS)
mkdir -p ~/dvc-storage

# Configure DVC to use it
dvc remote add -d myremote ~/dvc-storage
```

Expected output:
```
Setting 'myremote' as a default remote.
```

Check the configuration:

```bash
cat .dvc/config
```

```ini
[core]
    remote = myremote
[remote "myremote"]
    url = /Users/yourname/dvc-storage
```

### Push Data to Remote

```bash
dvc push
```

Expected output:
```
2 files pushed
```

### Production Remote Examples

**AWS S3:**
```bash
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote region us-east-1
```

**Google Cloud Storage:**
```bash
dvc remote add -d gcsremote gs://my-bucket/dvc-storage
```

**Azure Blob Storage:**
```bash
dvc remote add -d azureremote azure://my-container/dvc-storage
```

### Pull Data on Another Machine

When a teammate clones your repo:

```bash
git clone <repo-url>
cd project
dvc pull   # Downloads data from remote storage
```

---

## Part 8: Complete Workflow Summary

### Daily Workflow

```bash
# 1. Make changes to your data
# (add/remove/modify data files)

# 2. Track changes with DVC
dvc add data/reviews.csv

# 3. Commit metadata to git
git add data/reviews.csv.dvc
git commit -m "Update training data: add 500 new reviews"

# 4. Push to remotes
git push           # Code and metadata
dvc push           # Actual data files
```

### Reproducing Past Results

```bash
# Find the commit you want
git log --oneline

# Checkout that state
git checkout <commit-hash>
dvc checkout

# You now have the exact code + data from that point
```

### Collaboration Workflow

```bash
# Teammate pulls latest
git pull
dvc pull

# They now have your code AND your data
```

---

## Exercises

### Exercise 2.1.1: Track Your Data

1. Initialize DVC in your project if not done
2. Create or copy training data to `data/`
3. Track it with `dvc add`
4. Commit the `.dvc` file

Verify with:
```bash
git log --oneline  # Should show your data commit
cat data/*.dvc     # Should show the hash
```

### Exercise 2.1.2: Simulate Data Update and Rollback

1. Add 5 more rows to your dataset
2. Track and commit the change
3. Checkout the previous data version
4. Verify the data is restored
5. Return to the latest version

```bash
# Hint: use these commands
git log --oneline                    # See commits
git checkout <old-commit>            # Go back
dvc checkout                         # Restore data
wc -l data/reviews.csv               # Count lines
git checkout main && dvc checkout    # Return
```

### Exercise 2.1.3: Configure Remote Storage

1. Create a local "remote" directory
2. Configure DVC to use it
3. Push your data
4. Verify files exist in the remote

```bash
ls ~/dvc-storage/files/md5/   # Should see hash-named files
```

---

## Key Takeaways

1. **Git can't handle large/binary files** - This is by design, not a limitation to work around
2. **DVC stores metadata in git, data in remote storage** - Best of both worlds
3. **Content-addressable storage (hashes) enables deduplication and verification**
4. **Every git commit can point to a specific data version** - True reproducibility
5. **`dvc checkout` restores data for any git commit** - Time travel for your data
6. **Remote storage enables collaboration** - Like git push/pull, but for data

---

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| `dvc add` is slow on large files | Normal; hashing takes time. Consider `dvc add --no-commit` for batch operations |
| Data not restoring with `dvc checkout` | Ensure you ran `dvc pull` first (data must be in cache) |
| `.dvc` files showing as modified | Run `dvc status` to see what changed; `dvc add` to update |
| Remote push fails | Check credentials and remote URL in `.dvc/config` |

---

## Next Steps

Run `/start-2-2` to learn experiment tracking with MLflow. You'll track which model hyperparameters work best with which data versions.
