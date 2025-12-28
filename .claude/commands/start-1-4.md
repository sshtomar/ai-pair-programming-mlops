# Lesson 1.4: Packaging for Production

Read the lesson content from `lesson-modules/1-foundations/1-4-docker-basics.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"Your model runs on your machine. Will it run on the deployment server? On your colleague's laptop? This lesson ensures the answer is always 'yes.'"

### 2. Socratic Question
Ask: "Why can't we just copy our Python files to a server and run them?"

Expected answers: dependencies, Python version, system libraries, OS differences. Build on this to motivate containers.

### 3. Container Concepts (10 min)
Cover:
- Containers vs VMs (lightweight, share kernel)
- Images vs containers (blueprint vs instance)
- Dockerfile as recipe
- Reproducibility guarantee

Ask: "What belongs in a Dockerfile for an ML model?"

### 4. Write the Dockerfile (20 min)
Guide them through writing:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

CMD ["python", "-m", "src.predict"]
```

Teaching moments:
- Layer caching (requirements before code)
- Slim base images
- No unnecessary files

### 5. Build and Test (10 min)
Walk through:
```bash
docker build -t sentiment-classifier .
docker run sentiment-classifier
```

Debug any issues together.

### 6. Common Issues Exercise
Present 3 broken Dockerfiles. Have them identify problems:
1. Installing dev dependencies in production
2. Running as root
3. Huge image size from unoptimized layers

### 7. Wrap Up
- Level 1 complete: trained model, packaged in Docker
- Preview Level 2: But how do we track experiments? Version data?
- Next: `/start-2-1` for data versioning

## Teaching Notes
- Docker errors can be frustrating—be patient
- Explain error messages line by line
- If Docker not installed, help with installation first
