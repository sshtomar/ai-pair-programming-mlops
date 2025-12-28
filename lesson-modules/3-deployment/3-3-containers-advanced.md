# Lesson 3.3: Containerization Deep Dive

**Duration:** 40 minutes
**Prerequisites:** Lesson 1.4 (Docker basics), Lesson 3.2 (FastAPI app running)

## Learning Objectives

By the end of this lesson, you will:
1. Optimize Docker images for production (reduce size by 50-80%)
2. Implement multi-stage builds for ML applications
3. Handle secrets and configuration properly

---

## Why Image Size Matters

Your current development Dockerfile might produce a 2GB+ image. In production, this creates real costs:

| Impact | Large Image (2GB) | Optimized (400MB) |
|--------|-------------------|-------------------|
| Pull time (100 Mbps) | 2.5 minutes | 30 seconds |
| Cold start (Kubernetes) | 45+ seconds | 10 seconds |
| Storage cost (10 images) | $2.30/month | $0.46/month |
| Attack surface | 400+ packages | 50 packages |
| Vulnerability scan time | 15 minutes | 3 minutes |

**Cold starts kill user experience.** When Kubernetes scales up a new pod, users wait for the image to pull. A 2GB image means 45+ seconds of failed requests during traffic spikes.

---

## Multi-Stage Builds: The Core Technique

Multi-stage builds use multiple `FROM` statements. Each stage can copy artifacts from previous stages, leaving behind build dependencies.

### The Problem: Build Dependencies in Production

```dockerfile
# BAD: Single stage with all dependencies
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \      # Needed for pip install, not runtime
    gcc \                  # Needed for pip install, not runtime
    libffi-dev \           # Needed for pip install, not runtime
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "-m", "uvicorn", "main:app"]
# Result: ~1.5GB with unused compilers
```

### The Solution: Multi-Stage Build

```dockerfile
# ============================================
# Stage 1: Builder - compile dependencies
# ============================================
FROM python:3.11-slim AS builder

# Install build dependencies (only in this stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for clean copy
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime - lean production image
# ============================================
FROM python:3.11-slim AS runtime

# Copy only the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0"]
# Result: ~400MB, no compilers
```

**Key insight:** The `runtime` stage never sees `gcc`, `build-essential`, or any build tools. It only gets the compiled Python packages via the virtual environment.

### Multi-Stage Build Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-STAGE DOCKER BUILD                             │
└─────────────────────────────────────────────────────────────────────────────┘

  STAGE 1: BUILDER                        STAGE 2: RUNTIME (Final Image)
  ═══════════════════                     ════════════════════════════════

  ┌─────────────────────────┐             ┌─────────────────────────┐
  │ python:3.11-slim (130MB)│             │ python:3.11-slim (130MB)│
  ├─────────────────────────┤             ├─────────────────────────┤
  │ build-essential         │             │                         │
  │ gcc, g++                │─────X       │     (Not copied)        │
  │ libffi-dev              │             │                         │
  ├─────────────────────────┤             ├─────────────────────────┤
  │                         │             │                         │
  │  /opt/venv/             │────────────►│  /opt/venv/             │
  │  ├── bin/python         │   COPY      │  ├── bin/python         │
  │  ├── lib/               │  ──────►    │  ├── lib/               │
  │  │   └── site-packages/ │             │  │   └── site-packages/ │
  │  │       ├── fastapi    │             │  │       ├── fastapi    │
  │  │       ├── uvicorn    │             │  │       ├── uvicorn    │
  │  │       ├── sklearn    │             │  │       ├── sklearn    │
  │  │       └── ...        │             │  │       └── ...        │
  │  └── ...                │             │  └── ...                │
  │                         │             │                         │
  ├─────────────────────────┤             ├─────────────────────────┤
  │ pip cache               │             │  /app/                  │
  │ build artifacts         │─────X       │  ├── src/               │
  │ .pyc files              │             │  │   ├── api.py         │
  │                         │             │  │   └── schemas.py     │
  └─────────────────────────┘             │  └── models/            │
                                          │      └── model.pkl      │
       Total: ~800MB                      ├─────────────────────────┤
       (discarded)                        │ USER appuser            │
                                          │ HEALTHCHECK defined     │
                                          └─────────────────────────┘

                                                Total: ~400MB
                                                (shipped to prod)

  Legend:  ────────► = Copied to next stage
           ────X     = Discarded (not in final image)
```

---

## Complete Production Dockerfile for FastAPI + ML Model

Create `project/Dockerfile.prod`:

```dockerfile
# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

# Build arguments for flexibility
ARG PYTHON_VERSION=3.11

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip (prevents version warnings)
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim AS runtime

# Labels for image metadata
LABEL maintainer="your-team@company.com"
LABEL version="1.0.0"
LABEL description="Sentiment classifier API"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Prevent .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create non-root user BEFORE copying files
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy application code with correct ownership
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup models/ ./models/

# Switch to non-root user
USER appuser

# Expose port (documentation only)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Compare

```bash
# Build development image (single stage)
docker build -f Dockerfile -t sentiment:dev .
# Size: ~1.8GB

# Build production image (multi-stage)
docker build -f Dockerfile.prod -t sentiment:prod .
# Size: ~450MB

# Compare sizes
docker images | grep sentiment
# sentiment    prod    abc123    450MB
# sentiment    dev     def456    1.8GB
```

**Result: 75% size reduction.**

---

## Base Image Selection

Choosing the right base image is the highest-impact optimization decision.

### Python Base Image Comparison

| Base Image | Size | Use Case |
|------------|------|----------|
| `python:3.11` | 900MB | Never for production |
| `python:3.11-slim` | 130MB | Standard choice |
| `python:3.11-alpine` | 50MB | Caution: musl libc breaks some packages |
| `gcr.io/distroless/python3` | 52MB | Maximum security, no shell |

### When to Use Each

**`python:3.11-slim` (Recommended for ML)**
- Compatible with all Python packages
- Small enough for production
- Has debugging tools if needed

```dockerfile
FROM python:3.11-slim
```

**`python:3.11-alpine` (Use with caution)**
- Smallest size
- Uses musl libc instead of glibc
- **Breaks NumPy, pandas, scikit-learn** unless you compile from source
- Only use for pure Python applications

```dockerfile
# WARNING: This will fail for ML packages
FROM python:3.11-alpine
RUN pip install numpy  # Fails: needs glibc
```

**Distroless (Maximum security)**
- No shell, no package manager
- Cannot exec into container for debugging
- Smallest attack surface

```dockerfile
FROM gcr.io/distroless/python3-debian11
COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src/
WORKDIR /app
CMD ["src/main.py"]  # Must use full path, no shell
```

---

## Layer Optimization Deep Dive

### Rule 1: Order by Change Frequency

Layers that change less frequently should come first:

```dockerfile
# GOOD: Stable layers first
FROM python:3.11-slim

# 1. System packages (change monthly)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies (change weekly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Model artifacts (change daily/weekly)
COPY models/ ./models/

# 4. Application code (changes every commit)
COPY src/ ./src/
```

### Rule 2: Combine RUN Commands

Each `RUN` creates a layer. Combine related commands:

```dockerfile
# BAD: 4 layers, 4x metadata overhead
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN rm -rf /var/lib/apt/lists/*

# GOOD: 1 layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*
```

### Rule 3: Clean Up in the Same Layer

Cleanup in a separate `RUN` doesn't reduce image size:

```dockerfile
# BAD: apt cache persists in layer 1
RUN apt-get update && apt-get install -y gcc
RUN rm -rf /var/lib/apt/lists/*  # Too late!

# GOOD: Cleanup in same layer
RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*
```

---

## .dockerignore Best Practices

A comprehensive `.dockerignore` prevents unnecessary files from entering the build context.

Create `project/.dockerignore`:

```dockerignore
# Version control
.git/
.gitignore
.github/

# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/
*.egg

# Virtual environments
venv/
.venv/
ENV/
env/

# IDE and editors
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# Documentation
docs/
*.md
!README.md

# Data (mount at runtime instead)
data/raw/
data/processed/
*.csv
*.parquet
*.pkl
!models/*.pkl

# Notebooks
*.ipynb
.ipynb_checkpoints/

# Secrets (NEVER include)
.env
.env.*
*.pem
*.key
secrets/
credentials/

# Logs
*.log
logs/

# OS files
.DS_Store
Thumbs.db

# Docker files (prevent recursive context)
Dockerfile*
docker-compose*.yml
.dockerignore
```

### Measure Impact

```bash
# Check build context size
du -sh . --exclude=.git

# With .dockerignore
tar -czf - . 2>/dev/null | wc -c
# Without .dockerignore (add a '#' before each line)
# Compare the sizes
```

---

## Health Checks

Health checks tell orchestrators when your container is ready to receive traffic.

### Dockerfile HEALTHCHECK

```dockerfile
# Basic health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Without curl (smaller image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

### Health Endpoint in FastAPI

```python
# src/main.py
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

# Track startup time for health check
startup_time = datetime.utcnow()

@app.get("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "uptime_seconds": (datetime.utcnow() - startup_time).total_seconds(),
    }

@app.get("/ready")
def readiness_check():
    """Readiness check - is the model loaded and ready?"""
    # Check if model is loaded
    if not hasattr(app.state, "model") or app.state.model is None:
        return {"status": "not_ready", "reason": "model not loaded"}, 503

    return {"status": "ready"}
```

### Parameter Meanings

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--interval` | 30s | Check every 30 seconds |
| `--timeout` | 10s | Fail if check takes longer than 10s |
| `--start-period` | 5s | Wait 5s before first check (startup time) |
| `--retries` | 3 | Fail after 3 consecutive failures |

---

## Non-Root User Security

Running as root inside containers is a security risk. If an attacker escapes the container, they have root access to the host.

### Pattern: Create User Before Copying Files

```dockerfile
# Create user and group with specific IDs
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy files with correct ownership
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup models/ ./models/

# Switch to non-root user (last step before CMD)
USER appuser
```

### Verify Non-Root Execution

```bash
# Check which user the container runs as
docker run --rm sentiment:prod whoami
# Output: appuser

# Check user can't write to root-owned files
docker run --rm sentiment:prod touch /etc/passwd
# Output: touch: cannot touch '/etc/passwd': Permission denied
```

---

## Environment Variables vs Secrets

### Environment Variables: Configuration

Use for non-sensitive configuration:

```dockerfile
# Dockerfile
ENV MODEL_PATH=/app/models/sentiment_model.pkl
ENV LOG_LEVEL=info
ENV MAX_BATCH_SIZE=32
```

```bash
# Override at runtime
docker run -e LOG_LEVEL=debug sentiment:prod
```

### Secrets: Sensitive Data

**Never bake secrets into images:**

```dockerfile
# NEVER DO THIS
ENV DATABASE_PASSWORD=supersecret123
COPY .env /app/.env
```

**Use Docker secrets or mounted files:**

```bash
# Option 1: Environment variable at runtime
docker run -e DATABASE_URL="postgres://user:pass@host/db" sentiment:prod

# Option 2: Docker secrets (Swarm/Compose)
docker run --secret db_password sentiment:prod

# Option 3: Mounted secrets file
docker run -v /path/to/secrets:/run/secrets:ro sentiment:prod
```

### Configuration Pattern in FastAPI

```python
# src/config.py
import os
from pathlib import Path

class Settings:
    """Application settings from environment variables."""

    # Non-sensitive configuration
    model_path: str = os.getenv("MODEL_PATH", "/app/models/sentiment_model.pkl")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "32"))

    # Sensitive - from secrets or env
    database_url: str = os.getenv("DATABASE_URL", "")

    @classmethod
    def from_secrets(cls):
        """Load secrets from mounted files."""
        secrets_dir = Path("/run/secrets")

        if (secrets_dir / "database_url").exists():
            cls.database_url = (secrets_dir / "database_url").read_text().strip()

        return cls

settings = Settings.from_secrets()
```

---

## Docker Compose for Local Development

Docker Compose simplifies multi-container development.

Create `project/docker-compose.yml`:

```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=debug
      - MODEL_PATH=/app/models/sentiment_model.pkl
    volumes:
      # Mount source for live reloading (dev only)
      - ./src:/app/src:ro
      # Mount model directory
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - api

  # Optional: Add database for logging
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: mlops
      POSTGRES_PASSWORD: localdev  # Only for local dev!
      POSTGRES_DB: predictions
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Development vs Production Compose

Create `docker-compose.override.yml` for development (auto-loaded):

```yaml
# docker-compose.override.yml - Development overrides
version: "3.9"

services:
  api:
    build:
      target: builder  # Use builder stage for debugging
    volumes:
      - ./src:/app/src  # Enable live reload
    environment:
      - LOG_LEVEL=debug
    command: ["python", "-m", "uvicorn", "src.main:app", "--reload", "--host", "0.0.0.0"]
```

Create `docker-compose.prod.yml` for production:

```yaml
# docker-compose.prod.yml - Production settings
version: "3.9"

services:
  api:
    build:
      target: runtime
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "1"
          memory: 1G
    environment:
      - LOG_LEVEL=warning
    restart: always
```

### Commands

```bash
# Development (uses docker-compose.yml + docker-compose.override.yml)
docker compose up --build

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# View logs
docker compose logs -f api

# Stop all
docker compose down
```

---

## Image Scanning for Vulnerabilities

Scan images before deploying to production.

### Using Docker Scout (Built-in)

```bash
# Enable Docker Scout
docker scout quickview sentiment:prod

# Full vulnerability scan
docker scout cves sentiment:prod

# Get recommendations
docker scout recommendations sentiment:prod
```

### Using Trivy (Open Source)

```bash
# Install Trivy
brew install aquasecurity/trivy/trivy  # macOS

# Scan image
trivy image sentiment:prod

# Scan and fail on HIGH/CRITICAL
trivy image --severity HIGH,CRITICAL --exit-code 1 sentiment:prod
```

### Interpreting Results

```
sentiment:prod (debian 12.1)
=============================
Total: 23 (UNKNOWN: 0, LOW: 15, MEDIUM: 6, HIGH: 2, CRITICAL: 0)

| Library   | Vulnerability | Severity | Fixed Version |
|-----------|---------------|----------|---------------|
| libssl3   | CVE-2023-1234 | HIGH     | 3.0.11-1      |
| libcrypto | CVE-2023-5678 | HIGH     | 3.0.11-1      |
```

**Action items:**
- Update base image to get fixes
- Pin specific package versions in Dockerfile
- Add to CI/CD pipeline as gate

---

## Tagging Strategies

Never use `latest` in production. Use semantic versioning plus git SHA.

### Recommended Pattern

```bash
# Get version info
VERSION="1.2.3"
GIT_SHA=$(git rev-parse --short HEAD)
BUILD_DATE=$(date -u +%Y%m%d)

# Build with multiple tags
docker build \
  -t sentiment:${VERSION} \
  -t sentiment:${VERSION}-${GIT_SHA} \
  -t sentiment:${GIT_SHA} \
  -t sentiment:latest \
  .

# In CI/CD, also push to registry
docker tag sentiment:${VERSION} ghcr.io/myorg/sentiment:${VERSION}
docker push ghcr.io/myorg/sentiment:${VERSION}
```

### Tag Meanings

| Tag | Example | Use Case |
|-----|---------|----------|
| Semantic version | `1.2.3` | Production releases |
| Version + SHA | `1.2.3-abc123f` | Debugging specific builds |
| Git SHA only | `abc123f` | CI/CD builds |
| `latest` | `latest` | Development only, never production |
| Date-based | `20240115` | Nightly builds |

---

## Before/After Comparison

### Before: Naive Dockerfile

```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0"]
```

**Problems:**
- Full Python image (900MB base)
- All files copied (including .git, tests, notebooks)
- No layer caching optimization
- Running as root
- No health check
- Build dependencies in runtime image

**Image size: 2.1GB**

### After: Production Dockerfile

```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 app
COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/ ./models/
USER app
HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Improvements:**
- Slim base image
- Multi-stage build
- Proper layer caching
- Non-root user
- Health check
- No build dependencies in runtime

**Image size: 420MB (80% reduction)**

---

## Security Checklist

Before deploying any container to production:

- [ ] **Non-root user**: Container runs as non-root
- [ ] **Read-only filesystem**: Mount with `:ro` where possible
- [ ] **No secrets in image**: Check `docker history` for exposed secrets
- [ ] **Pinned versions**: Base image and all packages pinned
- [ ] **Vulnerability scan**: No HIGH/CRITICAL CVEs
- [ ] **Minimal base**: Using slim/distroless, not full image
- [ ] **Health check defined**: Orchestrator can verify readiness
- [ ] **Resource limits**: Memory and CPU limits set in compose/k8s
- [ ] **.dockerignore**: Excludes .git, .env, tests, notebooks
- [ ] **Labels**: Maintainer, version, description for traceability

---

## Exercises

### Exercise 3.3.1: Optimize Your Dockerfile

Take your current Dockerfile and apply optimizations:

1. Convert to multi-stage build
2. Switch to `python:3.11-slim` base
3. Add non-root user
4. Add HEALTHCHECK instruction
5. Measure before/after size

**Target: 50%+ size reduction**

```bash
# Measure before
docker images sentiment:before

# Build optimized
docker build -f Dockerfile.prod -t sentiment:after .

# Measure after
docker images sentiment:after
```

### Exercise 3.3.2: Security Audit

Run these checks on your production image:

```bash
# 1. Check user
docker run --rm sentiment:prod whoami

# 2. Check for secrets (should be empty)
docker history sentiment:prod | grep -i "password\|secret\|key"

# 3. Scan for vulnerabilities
docker scout cves sentiment:prod

# 4. Check image layers
docker history sentiment:prod --no-trunc
```

Document any findings and fix them.

### Exercise 3.3.3: Docker Compose Setup

Create a complete local development environment:

1. Create `docker-compose.yml` with your API
2. Add a health check
3. Mount volumes for live reloading
4. Verify with `docker compose up`

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

---

## Key Takeaways

1. **Multi-stage builds are mandatory for ML.** They reduce image size by 50-80% by excluding build dependencies from runtime.

2. **Layer order matters.** Put stable layers (dependencies) before volatile layers (code) for faster rebuilds.

3. **Never run as root.** Create a non-root user and switch before CMD.

4. **Scan images for vulnerabilities.** Use Docker Scout or Trivy in CI/CD as a deployment gate.

5. **Use `.dockerignore`.** Exclude .git, .env, tests, data, notebooks from build context.

6. **Health checks enable orchestration.** Without them, Kubernetes cannot know if your container is ready.

7. **Secrets belong at runtime.** Never bake credentials into images. Use environment variables or mounted secrets.

8. **Tag semantically.** Use `version-gitsha` pattern. Never deploy `latest` to production.

---

## Next Steps

Your container is now production-ready: small, secure, and observable.

Next, you'll deploy to the cloud. Lesson 3.4 covers:
- Container registries (pushing your image)
- Cloud Run deployment
- Kubernetes basics
- Load balancing and scaling

Start the next lesson:
```
/start-3-4
```
