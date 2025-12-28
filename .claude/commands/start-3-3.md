# Lesson 3.3: Containerization Deep Dive

Read the lesson content from `lesson-modules/3-deployment/3-3-containerization.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"Your API works on your machine. How do you guarantee it works on any machine?"

### 2. Socratic Question
Ask: "You built a Dockerfile in Level 1. What would you change now that you're deploying an API with a model?"

Expected: Multi-stage builds, model file handling, port exposure, health checks. Connect to their earlier Docker experience.

### 3. Production Dockerfile Patterns (15 min)
Build on Level 1 knowledge with production patterns:

```dockerfile
# Build stage
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY api/ ./api/
COPY models/ ./models/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Cover each pattern:
- Multi-stage builds (smaller images)
- Non-root users (security)
- Health checks (orchestration)
- Layer ordering (cache efficiency)

### 4. Build and Run (10 min)
```bash
docker build -t sentiment-api:v1 .
docker run -p 8000:8000 sentiment-api:v1
```

Test the containerized API:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great service!"}'
```

### 5. Model Versioning in Containers (10 min)
Discuss strategies:

**Baked-in model** (current approach)
- Model is part of image
- New model = new image
- Simple, reproducible

**External model loading**
- Mount model from volume or fetch from storage
- Same image, different models
- More flexible, more complex

Ask: "What are the trade-offs of each approach?"

### 6. Image Optimization (10 min)
Check image size:
```bash
docker images sentiment-api
```

Optimization techniques:
- Use slim/alpine base images
- Minimize layers
- Clean up in same layer as install
- Use .dockerignore

Create `.dockerignore`:
```
__pycache__
*.pyc
.git
.env
tests/
*.md
```

Rebuild and compare sizes.

### 7. Wrap Up
- Container is production-ready with health checks and security
- Multi-stage builds reduce image size
- Model packaging strategy is a key decision
- Preview: Lesson 3.4 deploys to cloud
- Next: `/start-3-4`

## Teaching Notes
- Compare before/after image sizes—it's motivating
- Security patterns (non-root) are easy to skip but crucial
- If Docker is slow, focus on concepts over repeated builds
- Reference Level 1 Docker knowledge—this is advancement, not repetition
