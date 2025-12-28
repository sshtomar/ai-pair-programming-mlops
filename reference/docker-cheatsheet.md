# Docker Cheatsheet for ML

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker build -t name .` | Build image from Dockerfile |
| `docker run -it name` | Run container interactively |
| `docker run -d -p 8000:8000 name` | Run detached with port mapping |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers (including stopped) |
| `docker images` | List local images |
| `docker logs container_id` | View container logs |
| `docker exec -it container_id bash` | Shell into running container |
| `docker stop container_id` | Stop a container |
| `docker rm container_id` | Remove a container |
| `docker rmi image_id` | Remove an image |

## Common Workflows

### Build and Run ML Model

```bash
# Build the image
docker build -t sentiment-model:v1 .

# Run with GPU support (if using nvidia-docker)
docker run --gpus all -p 8000:8000 sentiment-model:v1

# Run with volume mount for data
docker run -v $(pwd)/data:/app/data sentiment-model:v1
```

### Development Workflow

```bash
# Build with no cache (force fresh build)
docker build --no-cache -t sentiment-model:dev .

# Run with live code mounting
docker run -v $(pwd)/src:/app/src -p 8000:8000 sentiment-model:dev

# View logs in real-time
docker logs -f container_id
```

### Multi-Stage Build (Production)

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY model/ ./model/
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0"]
```

### Clean Up Resources

```bash
# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune

# Nuclear option: remove everything unused
docker system prune -a
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `port already in use` | Change port: `-p 8001:8000` or stop conflicting container |
| `no space left on device` | Run `docker system prune -a` |
| `permission denied` | Add user to docker group or use `sudo` |
| `cannot find module` | Check COPY commands include all needed files |
| `killed` (OOM) | Increase memory: `docker run -m 4g ...` |
| Image too large | Use slim base images, multi-stage builds, add `.dockerignore` |

## Best Practices for ML

1. **Pin versions**: `FROM python:3.11.4-slim` not `python:latest`
2. **Order matters**: Copy requirements.txt before code (layer caching)
3. **Use .dockerignore**: Exclude data files, notebooks, .git
4. **Don't store models in images**: Mount or download at runtime
5. **Health checks**: Add `HEALTHCHECK` for production readiness
