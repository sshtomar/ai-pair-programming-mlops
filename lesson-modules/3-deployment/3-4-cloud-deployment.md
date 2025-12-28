# Lesson 3.4: Deploying to Cloud

## Learning Objectives

By the end of this lesson, students will:
1. Deploy a containerized ML model to a cloud platform
2. Configure networking, scaling, and resource allocation
3. Set up basic monitoring and alerting

## Duration: 60 minutes

---

## Part 1: Cloud Platform Options

### Comparison of Major Platforms

| Platform | Service | Pros | Cons | Best For |
|----------|---------|------|------|----------|
| **AWS** | ECS/Fargate | Full control, mature ecosystem | Complex setup, many moving parts | Enterprise teams with AWS expertise |
| **AWS** | Lambda | Serverless, pay-per-invocation | Cold starts, 15min timeout, 10GB limit | Lightweight inference, low traffic |
| **GCP** | Cloud Run | Simple, auto-scaling, pay-per-use | Less control than GKE | Containerized APIs, variable traffic |
| **GCP** | GKE | Full Kubernetes | Operational overhead | Large-scale, multi-service deployments |
| **Azure** | Container Apps | Serverless containers, KEDA scaling | Newer, smaller community | Azure-native organizations |

### Why Cloud Run for This Course

We use GCP Cloud Run because:

1. **Simplest path to production**: One command deploys a container
2. **Pay-per-use**: Billed per 100ms of actual request time (not idle)
3. **Auto-scaling**: 0 to 1000 instances without configuration
4. **HTTPS by default**: Automatic TLS certificates
5. **Free tier**: 2 million requests/month, 360,000 GB-seconds

For learning MLOps, Cloud Run removes infrastructure complexity so you focus on deployment patterns.

### Target Architecture

```
                                  ┌─────────────────┐
                                  │    INTERNET     │
                                  └────────┬────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │    LOAD BALANCER      │
                               │  (Cloud Run managed)  │
                               └───────────┬───────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     │                     │                     │
                     ▼                     ▼                     ▼
              ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
              │ ┌─────────┐ │       │ ┌─────────┐ │       │ ┌─────────┐ │
              │ │ FastAPI │ │       │ │ FastAPI │ │       │ │ FastAPI │ │
              │ ├─────────┤ │       │ ├─────────┤ │       │ ├─────────┤ │
              │ │  Model  │ │       │ │  Model  │ │       │ │  Model  │ │
              │ │  v1.2   │ │       │ │  v1.2   │ │       │ │  v1.2   │ │
              │ └─────────┘ │       │ └─────────┘ │       │ └─────────┘ │
              │  Instance   │       │  Instance   │       │  Instance   │
              └─────────────┘       └─────────────┘       └─────────────┘
                     │                     │                     │
                     └─────────────────────┼─────────────────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │      MONITORING       │
                               │  Logs │ Metrics │ Alerts
                               └───────────────────────┘
```

Cloud Run automatically scales instances (0 to many) based on incoming traffic.

---

## Part 2: Prerequisites Setup

### Install Google Cloud CLI

```bash
# macOS
brew install google-cloud-sdk

# Linux (Debian/Ubuntu)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Verify installation
gcloud version
```

Expected output:
```
Google Cloud SDK 458.0.1
bq 2.0.101
core 2024.01.26
gcloud-crc32c 1.0.0
gsutil 5.27
```

### Authenticate and Configure

```bash
# Login (opens browser)
gcloud auth login

# Set project (create one at console.cloud.google.com if needed)
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com

# Set default region (choose closest to your users)
gcloud config set run/region us-central1
```

### Verify Billing

Cloud Run requires billing enabled. Check status:

```bash
gcloud billing accounts list
gcloud billing projects describe YOUR_PROJECT_ID
```

---

## Part 3: Push Container to Artifact Registry

### Create Repository

Artifact Registry replaced Container Registry as GCP's recommended container storage.

```bash
# Create repository
gcloud artifacts repositories create ml-models \
    --repository-format=docker \
    --location=us-central1 \
    --description="ML model containers"

# Verify creation
gcloud artifacts repositories list --location=us-central1
```

Expected output:
```
REPOSITORY  FORMAT  DESCRIPTION           LOCATION     CREATE_TIME
ml-models   DOCKER  ML model containers   us-central1  2024-01-15T10:30:00
```

### Configure Docker Authentication

```bash
# Configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Tag and Push Image

```bash
# Set variables for clarity
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
REPO=ml-models
IMAGE=sentiment-api

# Tag local image for Artifact Registry
docker tag sentiment-api:latest \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1.0.0

# Push to registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1.0.0
```

Expected output:
```
The push refers to repository [us-central1-docker.pkg.dev/my-project/ml-models/sentiment-api]
a1b2c3d4e5f6: Pushed
b2c3d4e5f6a1: Pushed
v1.0.0: digest: sha256:abc123... size: 1234
```

### Verify Upload

```bash
gcloud artifacts docker images list \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}
```

---

## Part 4: Deploy to Cloud Run

### Basic Deployment

```bash
gcloud run deploy sentiment-api \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1.0.0 \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated
```

Expected output:
```
Deploying container to Cloud Run service [sentiment-api] in project [my-project] region [us-central1]
OK Deploying new service... Done.
  OK Creating Revision...
  OK Routing traffic...
Done.
Service [sentiment-api] revision [sentiment-api-00001-abc] has been deployed and is serving 100 percent of traffic.
Service URL: https://sentiment-api-abc123-uc.a.run.app
```

### Verify Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe sentiment-api \
    --region ${REGION} \
    --format 'value(status.url)')

echo "Service URL: ${SERVICE_URL}"

# Test health endpoint
curl ${SERVICE_URL}/health

# Test prediction
curl -X POST ${SERVICE_URL}/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This product is excellent!"}'
```

Expected output:
```json
{"prediction": "positive", "confidence": 0.94, "model_version": "1.0.0"}
```

---

## Part 5: Configuration Options

### Cloud Run Configuration Reference

| Setting | Flag | Default | Recommended for ML |
|---------|------|---------|-------------------|
| Memory | `--memory` | 512Mi | 1Gi-2Gi (models need RAM) |
| CPU | `--cpu` | 1 | 1-2 (inference is CPU-bound) |
| Concurrency | `--concurrency` | 80 | 10-20 (if model is heavy) |
| Timeout | `--timeout` | 300s | 60s (fail fast) |
| Min instances | `--min-instances` | 0 | 1 (avoid cold starts) |
| Max instances | `--max-instances` | 100 | 10-50 (cost control) |
| CPU allocation | `--cpu-throttling` | true | false (consistent perf) |

### Production Configuration

```bash
gcloud run deploy sentiment-api \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1.0.0 \
    --platform managed \
    --region ${REGION} \
    --memory 1Gi \
    --cpu 1 \
    --concurrency 20 \
    --timeout 60s \
    --min-instances 1 \
    --max-instances 10 \
    --no-cpu-throttling \
    --allow-unauthenticated
```

### Understanding Concurrency

Concurrency controls how many requests one container instance handles simultaneously.

```
--concurrency 80 (default):
  Single instance handles 80 concurrent requests
  Good for lightweight APIs

--concurrency 10:
  Single instance handles 10 concurrent requests
  Better for memory-heavy ML models
  More instances spin up under load
```

**Rule of thumb for ML**: Set concurrency to `(Available Memory MB) / (Model Memory MB) / 2`

If your model uses 200MB and container has 1GB:
```
concurrency = 1024 / 200 / 2 = ~2-3
```

Start conservative (10-20), monitor, then adjust.

---

## Part 6: Environment Variables and Secrets

### Environment Variables

For non-sensitive configuration:

```bash
gcloud run deploy sentiment-api \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1.0.0 \
    --set-env-vars "MODEL_VERSION=1.0.0,LOG_LEVEL=INFO,MAX_BATCH_SIZE=32"
```

### Secrets with Secret Manager

For API keys, credentials, and sensitive data:

```bash
# Create secret
echo -n "your-api-key-here" | gcloud secrets create api-key --data-file=-

# Grant Cloud Run access
gcloud secrets add-iam-policy-binding api-key \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Mount secret as environment variable
gcloud run deploy sentiment-api \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1.0.0 \
    --set-secrets "API_KEY=api-key:latest"
```

In your application, access via `os.environ["API_KEY"]`.

### Secrets as Mounted Files

For certificates or config files:

```bash
gcloud run deploy sentiment-api \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1.0.0 \
    --set-secrets "/secrets/config.json=config-secret:latest"
```

---

## Part 7: Custom Domains and HTTPS

### Map Custom Domain

```bash
# Verify domain ownership first (one-time)
gcloud domains verify example.com

# Map domain to service
gcloud run domain-mappings create \
    --service sentiment-api \
    --domain api.example.com \
    --region ${REGION}
```

Output shows required DNS records:
```
NAME                  TYPE   DATA
api.example.com       CNAME  ghs.googlehosted.com.
```

Add this CNAME record to your DNS provider. Certificate provisioning takes 15-30 minutes.

### Verify Domain Mapping

```bash
gcloud run domain-mappings describe \
    --domain api.example.com \
    --region ${REGION}
```

Wait for `certificateStatus: CERTIFICATE_STATUS_ACTIVE`.

---

## Part 8: Auto-Scaling Behavior

### How Cloud Run Scales

```
Incoming Requests
       |
       v
  Load Balancer
       |
       v
+------+------+------+
| Inst | Inst | Inst |  <-- Instances scale 0 to max-instances
|  1   |  2   |  3   |
+------+------+------+
       ^
       |
  Scaling Decision:
  - Current requests > instances * concurrency? Add instance
  - Instances idle? Remove after ~15 min
  - CPU > 60%? Add instance
```

### Cold Start Mitigation

Cold starts occur when scaling from 0 or adding instances:

| Strategy | Implementation | Trade-off |
|----------|----------------|-----------|
| Min instances | `--min-instances 1` | Constant cost (~$10-30/month) |
| CPU always allocated | `--no-cpu-throttling` | Higher cost, faster response |
| Smaller container | Optimize Docker image | Development effort |
| Model warm-up | Load model at startup | Longer deploy time |

### Startup Probe Configuration

Tell Cloud Run your container needs time to load the model:

```bash
gcloud run deploy sentiment-api \
    --image ${IMAGE_URL} \
    --cpu-boost \
    --startup-cpu-boost
```

Or in YAML:
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: sentiment-api
spec:
  template:
    spec:
      containers:
      - image: IMAGE_URL
        startupProbe:
          httpGet:
            path: /health
          initialDelaySeconds: 10
          timeoutSeconds: 3
          periodSeconds: 5
          failureThreshold: 12
```

---

## Part 9: Basic Monitoring Setup

### Built-in Metrics (Cloud Console)

Cloud Run automatically tracks:

- Request count and latency (p50, p95, p99)
- Container instance count
- CPU and memory utilization
- Billable container instance time

View in Cloud Console: **Cloud Run > sentiment-api > Metrics**

### Create Alerting Policy

```bash
# Alert if error rate exceeds 1%
gcloud monitoring policies create \
    --display-name="High Error Rate - Sentiment API" \
    --condition-display-name="Error rate > 1%" \
    --condition-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class="5xx"' \
    --condition-threshold-value=0.01 \
    --condition-threshold-comparison=COMPARISON_GT \
    --notification-channels=YOUR_CHANNEL_ID \
    --documentation="Sentiment API error rate exceeded 1%. Check logs for details."
```

### Create Notification Channel

```bash
# Create email notification
gcloud monitoring channels create \
    --display-name="ML Team Email" \
    --type=email \
    --channel-labels=email_address=ml-alerts@yourcompany.com
```

### View Logs

```bash
# Stream logs in real-time
gcloud run services logs read sentiment-api --region ${REGION} --tail 50

# Filter for errors
gcloud logging read 'resource.type="cloud_run_revision" AND severity>=ERROR' \
    --limit 20 \
    --format="table(timestamp,textPayload)"
```

### Add Custom Metrics

In your FastAPI application:

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Define metrics
PREDICTIONS = Counter(
    'predictions_total',
    'Total predictions',
    ['model_version', 'sentiment']
)
LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency'
)

@app.post("/predict")
async def predict(request: PredictRequest):
    with LATENCY.time():
        result = model.predict(request.text)

    PREDICTIONS.labels(
        model_version=MODEL_VERSION,
        sentiment=result.prediction
    ).inc()

    return result

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

---

## Part 10: Cost Estimation

### Cloud Run Pricing (as of 2024)

| Resource | Free Tier | Price After Free Tier |
|----------|-----------|----------------------|
| Requests | 2M/month | $0.40 per million |
| CPU | 180,000 vCPU-seconds | $0.000024 per vCPU-second |
| Memory | 360,000 GB-seconds | $0.0000025 per GB-second |
| Min instances | Not included | Same as above |

### Cost Calculation Example

**Scenario**: ML API with:
- 100,000 requests/day
- Average latency: 200ms
- 1 vCPU, 1GB memory
- 1 min instance (to avoid cold starts)

**Monthly calculation**:

```
Requests: 3M requests
  - Free: 2M
  - Paid: 1M x $0.40/M = $0.40

CPU (active time):
  - 3M requests x 0.2s = 600,000 vCPU-seconds
  - Free: 180,000
  - Paid: 420,000 x $0.000024 = $10.08

Memory (active time):
  - 600,000 GB-seconds
  - Free: 360,000
  - Paid: 240,000 x $0.0000025 = $0.60

Min instance (idle time):
  - 1 instance x 30 days x 24h x 3600s = 2,592,000 seconds
  - Minus active time: ~2,000,000 idle seconds
  - CPU: 2,000,000 x $0.000024 = $48.00
  - Memory: 2,000,000 x $0.0000025 = $5.00

Total: $0.40 + $10.08 + $0.60 + $48.00 + $5.00 = ~$64/month
```

### Cost Optimization Strategies

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Remove min-instances | ~$50/month | Cold starts (~2-5s) |
| Use CPU throttling | ~20% | Variable latency |
| Reduce memory | Proportional | May cause OOM |
| Batch requests | ~30-50% | Client complexity |
| Regional deployment | ~20% | Single region |

**Recommendation**: Start with min-instances=0, add if cold starts are problematic.

---

## Part 11: Rollback and Traffic Splitting

### View Revisions

```bash
gcloud run revisions list --service sentiment-api --region ${REGION}
```

Output:
```
REVISION                    ACTIVE  SERVICE        DEPLOYED                 LAST_DEPLOYED_BY
sentiment-api-00003-def     yes     sentiment-api  2024-01-15T14:30:00Z    user@example.com
sentiment-api-00002-abc     yes     sentiment-api  2024-01-15T10:00:00Z    user@example.com
sentiment-api-00001-xyz            sentiment-api  2024-01-14T09:00:00Z    user@example.com
```

### Rollback to Previous Revision

```bash
# Instant rollback
gcloud run services update-traffic sentiment-api \
    --region ${REGION} \
    --to-revisions sentiment-api-00002-abc=100
```

### Canary Deployment (Traffic Splitting)

Deploy new version to 10% of traffic:

```bash
# Deploy new version (creates new revision)
gcloud run deploy sentiment-api \
    --image ${IMAGE_URL}:v1.1.0 \
    --region ${REGION} \
    --no-traffic

# Split traffic: 90% old, 10% new
gcloud run services update-traffic sentiment-api \
    --region ${REGION} \
    --to-revisions sentiment-api-00003-def=10,sentiment-api-00002-abc=90
```

Monitor error rates. If stable, increase:

```bash
# 50/50 split
gcloud run services update-traffic sentiment-api \
    --region ${REGION} \
    --to-revisions sentiment-api-00003-def=50,sentiment-api-00002-abc=50

# Full rollout
gcloud run services update-traffic sentiment-api \
    --region ${REGION} \
    --to-latest
```

### Tag-Based Routing

Test new versions without affecting production:

```bash
# Deploy with tag (no production traffic)
gcloud run deploy sentiment-api \
    --image ${IMAGE_URL}:v2.0.0-beta \
    --region ${REGION} \
    --tag beta \
    --no-traffic
```

Access via: `https://beta---sentiment-api-abc123-uc.a.run.app`

---

## Part 12: Complete Deployment Script

Create `scripts/deploy.sh`:

```bash
#!/bin/bash
set -euo pipefail

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
REGION=${GCP_REGION:-us-central1}
SERVICE_NAME=sentiment-api
REPO=ml-models
IMAGE_TAG=${1:-latest}

# Derived values
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "Deploying ${SERVICE_NAME} version ${IMAGE_TAG}"
echo "Image: ${IMAGE_URL}"
echo "Region: ${REGION}"

# Build and push
echo "Building container..."
docker build -t ${SERVICE_NAME}:${IMAGE_TAG} .
docker tag ${SERVICE_NAME}:${IMAGE_TAG} ${IMAGE_URL}

echo "Pushing to Artifact Registry..."
docker push ${IMAGE_URL}

# Deploy
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_URL} \
    --platform managed \
    --region ${REGION} \
    --memory 1Gi \
    --cpu 1 \
    --concurrency 20 \
    --timeout 60s \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "MODEL_VERSION=${IMAGE_TAG},LOG_LEVEL=INFO" \
    --allow-unauthenticated

# Verify
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)')

echo "Service deployed: ${SERVICE_URL}"

echo "Running health check..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" ${SERVICE_URL}/health)

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "Health check passed"
else
    echo "Health check failed with status ${HTTP_STATUS}"
    exit 1
fi

echo "Testing prediction endpoint..."
RESPONSE=$(curl -s -X POST ${SERVICE_URL}/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Deployment test"}')

echo "Response: ${RESPONSE}"
echo "Deployment complete!"
```

Make executable:
```bash
chmod +x scripts/deploy.sh
```

Usage:
```bash
# Deploy with version tag
./scripts/deploy.sh v1.0.0

# Deploy latest
./scripts/deploy.sh
```

---

## Exercises

### Exercise 3.4.1: Deploy Your API

1. Set up GCP project and enable APIs
2. Push your container to Artifact Registry
3. Deploy to Cloud Run with default settings
4. Verify the health and prediction endpoints work

**Verification**:
```bash
# Should return 200
curl -I ${SERVICE_URL}/health

# Should return prediction
curl -X POST ${SERVICE_URL}/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Great product!"}'
```

### Exercise 3.4.2: Optimize Configuration

1. Deploy with production settings (memory, concurrency, min-instances)
2. Run 10 sequential requests and measure latency
3. Compare cold start vs warm latency
4. Calculate estimated monthly cost for 50,000 requests/day

**Commands to measure**:
```bash
# Measure cold start (after scaling to 0)
time curl -X POST ${SERVICE_URL}/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "test"}'

# Measure warm request
time curl -X POST ${SERVICE_URL}/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "test"}'
```

### Exercise 3.4.3: Canary Deployment

1. Make a small code change (update version in response)
2. Build and push new image with v1.1.0 tag
3. Deploy as canary with 10% traffic
4. Verify both versions are receiving traffic
5. Complete rollout or rollback

**Verification**:
```bash
# Run 20 requests, count version distribution
for i in {1..20}; do
    curl -s ${SERVICE_URL}/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "test"}' | jq -r '.model_version'
done | sort | uniq -c
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Permission denied" on push | Not authenticated | `gcloud auth configure-docker` |
| Container crashes on start | Port mismatch | Ensure `PORT` env var used |
| 503 errors | Instance scaling | Check min-instances, increase timeout |
| OOM kills | Insufficient memory | Increase `--memory` |
| Slow cold starts | Large container | Optimize image, use min-instances |
| "Image not found" | Wrong region/project | Verify image URL matches registry |

### Debug Commands

```bash
# Check service status
gcloud run services describe sentiment-api --region ${REGION}

# View recent logs
gcloud run services logs read sentiment-api --region ${REGION} --limit 100

# Check revision status
gcloud run revisions describe sentiment-api-00001-abc --region ${REGION}

# Test container locally before deploying
docker run -p 8080:8080 -e PORT=8080 sentiment-api:latest
```

---

## Key Takeaways

1. **Cloud Run simplifies container deployment.** One command deploys a production-ready API with HTTPS, auto-scaling, and monitoring.

2. **Configure for ML workloads.** Increase memory (1-2GB), reduce concurrency (10-20), consider min-instances for latency-sensitive applications.

3. **Use Secret Manager for credentials.** Never put API keys or passwords in environment variables or code.

4. **Min-instances is a trade-off.** Eliminates cold starts but adds ~$50/month baseline cost. Start without it.

5. **Canary deployments reduce risk.** Always deploy new versions to a small percentage first, monitor, then increase.

6. **Monitor from day one.** Set up error rate alerts before you need them.

7. **Cost scales with usage.** Cloud Run's pay-per-use model is economical for variable traffic, expensive for sustained high load.

---

## Next Steps

**Level 3 complete!** You now have:
- A production FastAPI server
- Optimized Docker container
- Cloud deployment with auto-scaling
- Basic monitoring and alerting

Run `/start-4-1` to begin Level 4: Production Operations. You'll learn CI/CD pipelines, advanced monitoring, and drift detection.
