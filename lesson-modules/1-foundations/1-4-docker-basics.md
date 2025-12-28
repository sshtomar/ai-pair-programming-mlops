# Lesson 1.4: Packaging for Production

**Duration:** 45 minutes
**Prerequisites:** Completed Lessons 1.1-1.3, working model in `project/src/`

## Learning Objectives

By the end of this lesson, you will:
1. Understand why containers are essential for ML deployment
2. Write a production-ready Dockerfile for ML applications
3. Handle model artifacts and dependencies correctly

---

## The "Works on My Machine" Problem

Your model trains perfectly locally. You ship it to production. It fails.

**Common causes:**

| Issue | Local | Production |
|-------|-------|------------|
| Python version | 3.11.4 | 3.9.2 |
| NumPy version | 1.26.0 | 1.24.3 |
| scikit-learn | 1.3.0 | 1.2.0 |
| System libraries | macOS libstdc++ | Linux glibc |
| CUDA version | 12.1 | 11.8 |

**Real failure examples:**

```
# Pickle protocol mismatch
_pickle.UnpicklingError: invalid load key, '\x80'.

# Missing system dependency
ImportError: libgomp.so.1: cannot open shared object file

# API change between versions
AttributeError: 'RandomForestClassifier' object has no attribute 'n_features_'
```

Containers solve this by packaging your entire environment—Python, dependencies, system libraries—into a single deployable unit.

---

## Containers vs Virtual Machines

| Aspect | Containers | Virtual Machines |
|--------|------------|------------------|
| Size | MBs (50-500 MB typical) | GBs (10+ GB typical) |
| Startup | Seconds | Minutes |
| Isolation | Process-level (shared kernel) | Full OS isolation |
| Resource overhead | Low (~1-2% CPU) | High (~15-20% CPU) |
| Portability | Excellent (single file) | Poor (hypervisor-dependent) |
| Use case | Microservices, ML inference | Legacy apps, security isolation |

**Key insight:** Containers share the host kernel. They're not VMs—they're isolated processes with their own filesystem.

```
┌─────────────────────────────────────┐
│           Host Machine              │
├─────────────────────────────────────┤
│              Host OS                │
├─────────────────────────────────────┤
│           Docker Engine             │
├───────────┬───────────┬─────────────┤
│ Container │ Container │  Container  │
│   App A   │   App B   │    App C    │
│  Python   │  Python   │   Python    │
│   3.11    │   3.9     │    3.10     │
└───────────┴───────────┴─────────────┘
```

---

## Images vs Containers

**Image:** A read-only template. Think of it as a class definition.
**Container:** A running instance of an image. Think of it as an object.

```bash
# Build an image (creates template)
docker build -t sentiment-model:v1 .

# Run a container (creates instance from template)
docker run sentiment-model:v1

# You can run multiple containers from one image
docker run --name instance-1 sentiment-model:v1
docker run --name instance-2 sentiment-model:v1
```

---

## Dockerfile Anatomy

A Dockerfile is a script that defines how to build an image.

### Core Instructions

| Instruction | Purpose | Example |
|-------------|---------|---------|
| `FROM` | Base image | `FROM python:3.11-slim` |
| `WORKDIR` | Set working directory | `WORKDIR /app` |
| `COPY` | Copy files from host | `COPY requirements.txt .` |
| `RUN` | Execute commands | `RUN pip install -r requirements.txt` |
| `ENV` | Set environment variables | `ENV MODEL_PATH=/app/models` |
| `EXPOSE` | Document port usage | `EXPOSE 8000` |
| `CMD` | Default command to run | `CMD ["python", "predict.py"]` |
| `ENTRYPOINT` | Fixed command prefix | `ENTRYPOINT ["python"]` |

### CMD vs ENTRYPOINT

```dockerfile
# CMD: Can be overridden at runtime
CMD ["python", "predict.py"]
# docker run myimage python train.py  # Overrides CMD

# ENTRYPOINT: Fixed, arguments appended
ENTRYPOINT ["python"]
CMD ["predict.py"]
# docker run myimage train.py  # Runs: python train.py
```

---

## Layer Caching Strategy

Docker builds images in layers. Each instruction creates a layer. Layers are cached.

**Bad: Invalidates cache on every code change**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .                           # Code changes invalidate this
RUN pip install -r requirements.txt # Reinstalls every time!
CMD ["python", "predict.py"]
```

**Good: Dependencies cached separately from code**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .            # Rarely changes
RUN pip install -r requirements.txt # Cached until requirements change
COPY . .                           # Code changes only rebuild from here
CMD ["python", "predict.py"]
```

**Build time difference:**
- Bad pattern: 2-3 minutes per build
- Good pattern: 10-15 seconds (when only code changes)

---

## Complete Dockerfile for Sentiment Classifier

Create this file at `project/Dockerfile`:

```dockerfile
# ============================================
# Stage 1: Base image with dependencies
# ============================================
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (if needed for certain packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Install Python dependencies
# ============================================
FROM base AS dependencies

# Copy only requirements first (layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: Production image
# ============================================
FROM dependencies AS production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Change ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "src/predict.py"]
```

**Key decisions explained:**

1. **`python:3.11-slim`**: 45MB vs 900MB for full image. Includes essentials only.
2. **`--no-cache-dir`**: Reduces image size by not storing pip cache.
3. **Non-root user**: Security best practice. Containers running as root can compromise the host.
4. **Multi-stage potential**: This structure allows adding a build stage later if needed.

### Training vs Serving Containers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRAINING ENVIRONMENT                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │  │
│  │  │ pandas │ │sklearn │ │ mlflow │ │ pytest │ │  DVC   │          │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘          │  │
│  │                       train.py                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                    500MB+ container                                     │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                               ▼
                        [ model.joblib ]
                               │
                               ▼
┌──────────────────────────────┼──────────────────────────────────────────┐
│                              │                                          │
│                     50MB container                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ┌─────────┐  ┌────────┐  ┌───────┐                               │  │
│  │  │ fastapi │  │ joblib │  │ numpy │  ← Minimal dependencies       │  │
│  │  └─────────┘  └────────┘  └───────┘                               │  │
│  │                       predict.py                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                       SERVING ENVIRONMENT                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this matters:** Smaller containers = faster deploys, less attack surface, lower costs.

---

## Building and Running

### Build Commands

```bash
# Navigate to project directory
cd project/

# Build the image
docker build -t sentiment-model:v1 .

# Build with no cache (force fresh build)
docker build --no-cache -t sentiment-model:v1 .

# Build with specific Dockerfile
docker build -f Dockerfile.prod -t sentiment-model:v1 .
```

### Run Commands

```bash
# Basic run
docker run sentiment-model:v1

# Interactive mode (for debugging)
docker run -it sentiment-model:v1 /bin/bash

# Mount local directory (development)
docker run -v $(pwd)/data:/app/data sentiment-model:v1

# Set environment variables
docker run -e MODEL_PATH=/app/models/v2 sentiment-model:v1

# Expose ports (for API serving later)
docker run -p 8000:8000 sentiment-model:v1

# Run in background
docker run -d --name sentiment-api sentiment-model:v1

# View logs
docker logs sentiment-api

# Stop container
docker stop sentiment-api
```

### Useful Inspection Commands

```bash
# List images
docker images

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Check image size and layers
docker history sentiment-model:v1

# Inspect image details
docker inspect sentiment-model:v1

# Remove unused images
docker image prune

# Remove all stopped containers
docker container prune
```

---

## Debugging Common Docker Errors

### Error: "No such file or directory"

```
COPY failed: file not found in build context: requirements.txt
```

**Cause:** File doesn't exist or is in `.dockerignore`

**Fix:**
```bash
# Check file exists
ls -la requirements.txt

# Check .dockerignore isn't excluding it
cat .dockerignore
```

### Error: "pip install fails"

```
ERROR: Could not find a version that satisfies the requirement torch==2.0.0
```

**Cause:** Package not available for the container's platform/Python version

**Fix:**
```dockerfile
# Check platform
FROM python:3.11-slim
RUN python --version && pip --version

# Use flexible version constraints
# Bad:  torch==2.0.0
# Good: torch>=2.0.0,<3.0.0
```

### Error: "Permission denied"

```
PermissionError: [Errno 13] Permission denied: '/app/models/model.pkl'
```

**Cause:** Running as non-root user without file ownership

**Fix:**
```dockerfile
# Ensure correct ownership before switching users
COPY --chown=appuser:appuser models/ ./models/
USER appuser
```

### Error: "Killed" (OOM)

```
Killed
```

**Cause:** Container ran out of memory during build or run

**Fix:**
```bash
# Increase Docker memory limit (Docker Desktop settings)
# Or limit pip's memory usage
RUN pip install --no-cache-dir -r requirements.txt
```

### Error: "Image too large"

```
REPOSITORY          TAG       SIZE
sentiment-model     v1        2.5GB
```

**Cause:** Unnecessary files, dev dependencies, or cache

**Fix:**
```dockerfile
# Use slim base image
FROM python:3.11-slim  # Not python:3.11

# Don't install dev dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clean up in the same RUN command
RUN apt-get update && apt-get install -y gcc \
    && pip install -r requirements.txt \
    && apt-get purge -y gcc \
    && rm -rf /var/lib/apt/lists/*

# Use .dockerignore
```

Create `project/.dockerignore`:
```
# Git
.git/
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
venv/
.venv/

# IDE
.vscode/
.idea/

# Data (mount at runtime instead)
data/raw/
*.csv

# Notebooks
*.ipynb

# Local config
.env
*.log
```

---

## Exercise: Fix the Broken Dockerfiles

### Dockerfile A: Find 5 problems

```dockerfile
FROM python:latest
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN pip install jupyter notebook pandas matplotlib
EXPOSE 8000
CMD python src/predict.py
```

<details>
<summary>Click to reveal problems</summary>

1. **`python:latest`** - Unpinned version. Will break when Python updates.
2. **`COPY . .` before pip install** - Invalidates cache on every code change.
3. **Dev dependencies installed** - Jupyter/notebook not needed in production.
4. **Running as root** - Security vulnerability.
5. **`CMD python`** - Should use exec form: `CMD ["python", "src/predict.py"]`

**Fixed version:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "src/predict.py"]
```
</details>

### Dockerfile B: Find 4 problems

```dockerfile
FROM python:3.11
WORKDIR /app
RUN pip install torch tensorflow scikit-learn pandas numpy
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV MODEL_PATH=./models/model.pkl
CMD ["python", "src/predict.py"]
```

<details>
<summary>Click to reveal problems</summary>

1. **`python:3.11`** - Full image (~900MB). Use `python:3.11-slim`.
2. **Hardcoded pip install** - Dependencies should be in requirements.txt only.
3. **Duplicate pip install** - Installs packages twice, bloating image.
4. **Relative path in ENV** - Use absolute path: `ENV MODEL_PATH=/app/models/model.pkl`

**Fixed version:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/model.pkl
CMD ["python", "src/predict.py"]
```
</details>

### Dockerfile C: Find 3 problems

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y libgomp1

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/predict.py"]
```

<details>
<summary>Click to reveal problems</summary>

1. **Multiple RUN commands** - Creates unnecessary layers. Combine with `&&`.
2. **No cleanup** - apt cache remains in image, adding ~100MB+.
3. **No `--no-install-recommends`** - Installs unnecessary packages.

**Fixed version:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
CMD ["python", "src/predict.py"]
```
</details>

---

## Handling Model Artifacts

### Option 1: Bake into image (simple, versioned)

```dockerfile
COPY models/ ./models/
```

**Pros:** Single artifact, version-controlled
**Cons:** Large images, rebuild for model updates

### Option 2: Mount at runtime (flexible)

```bash
docker run -v /path/to/models:/app/models sentiment-model:v1
```

**Pros:** Update models without rebuilding
**Cons:** Requires external model storage, more moving parts

### Option 3: Download on startup (production pattern)

```dockerfile
ENV MODEL_URI=s3://bucket/models/v1/model.pkl
CMD ["python", "src/download_and_predict.py"]
```

**Pros:** Decoupled model and code versioning
**Cons:** Startup latency, needs cloud credentials

For this course, we'll use **Option 1** for simplicity. Level 3 introduces model registries.

---

## Key Takeaways

1. **Containers eliminate "works on my machine"** by packaging the entire runtime environment.

2. **Layer order matters.** Put rarely-changing items (dependencies) before frequently-changing items (code).

3. **Use slim base images.** `python:3.11-slim` is 45MB vs 900MB for full.

4. **Never run as root.** Create and switch to a non-root user.

5. **Use `.dockerignore`.** Exclude `.git/`, `__pycache__/`, `venv/`, test data.

6. **Pin versions.** Base image, Python packages, everything.

7. **Clean up in the same layer.** `apt-get install && rm -rf /var/lib/apt/lists/*`

---

## Checklist Before Moving On

Before proceeding to Level 2, verify:

- [ ] Docker is installed and running (`docker --version`)
- [ ] You can build the sentiment model image
- [ ] The container runs successfully
- [ ] You understand the layer caching strategy
- [ ] You've created a `.dockerignore` file

Test your setup:
```bash
cd project/
docker build -t sentiment-model:v1 .
docker run sentiment-model:v1
```

---

## Next Steps

You've completed Level 1: Foundations. You have:
- A working development environment
- Understanding of MLOps principles
- A trained sentiment classifier
- A production-ready Docker image

**Level 2** introduces professional ML workflows:
- **Data versioning** with DVC (track datasets like code)
- **Experiment tracking** with MLflow (never lose a good result)
- **Testing** for ML systems (beyond unit tests)

Start Level 2:
```
/start-2-1
```
