# Lesson 3.4: Deploying to Cloud

Read the lesson content from `lesson-modules/3-deployment/3-4-cloud-deployment.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"Your API runs in a container. Now let's make it accessible to the world—and keep it running."

### 2. Socratic Question
Ask: "What does 'deploying to production' mean beyond just running your container somewhere?"

Expected: Scaling, load balancing, HTTPS, monitoring, secrets management, reliability. Guide them to see deployment as a system, not just a command.

### 3. Cloud Deployment Options (10 min)
Overview of options (pick one to implement):

**Container-as-a-Service (Recommended for learning)**
- AWS App Runner, Google Cloud Run, Azure Container Apps
- Simple: push container, get URL
- Auto-scaling, managed HTTPS

**Kubernetes**
- Full orchestration control
- Complex but powerful
- Overkill for single services

**Serverless (Lambda/Functions)**
- Pay-per-request
- Cold start issues for ML models
- Best for lightweight models

**Managed ML Platforms**
- SageMaker, Vertex AI
- ML-specific features (A/B testing, model monitoring)
- Higher cost, vendor lock-in

For this lesson: Focus on Cloud Run or App Runner (simplest path).

### 4. Prepare for Deployment (10 min)
Ensure the container is ready:
```bash
# Test locally one more time
docker run -p 8000:8000 sentiment-api:v1
curl http://localhost:8000/health
```

Create `cloudbuild.yaml` or equivalent:
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/sentiment-api:$SHORT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/sentiment-api:$SHORT_SHA']
```

### 5. Deploy to Cloud Run (15 min)
Walk through deployment:
```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/sentiment-api

# Deploy to Cloud Run
gcloud run deploy sentiment-api \
  --image gcr.io/YOUR_PROJECT/sentiment-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

Note: If they don't have GCP, discuss the equivalent AWS/Azure commands or use a local simulation.

### 6. Verify Deployment (10 min)
Test the deployed service:
```bash
# Get the service URL
gcloud run services describe sentiment-api --format='value(status.url)'

# Test it
curl -X POST https://YOUR-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

Explore the console:
- View logs
- Check metrics
- See scaling behavior

### 7. Production Considerations (10 min)
Discuss what's still missing:
- **Authentication**: API keys, OAuth, IAM
- **Rate limiting**: Prevent abuse
- **Custom domain**: Professional URLs
- **Secrets management**: Don't hardcode credentials
- **Cost monitoring**: Set budgets and alerts

Ask: "What would break if 1000 users hit your API simultaneously?"

### 8. Wrap Up
- Level 3 complete: API deployed and accessible
- You have a production-grade ML service
- Preview Level 4: CI/CD, monitoring, and operational excellence
- Next: `/start-4-1` for CI/CD pipelines

## Teaching Notes
- Cloud accounts can be tricky—have local alternatives ready
- Emphasize this is a starting point, not production-complete
- Cost awareness: remind them to clean up resources
- This is a milestone—acknowledge their progress
