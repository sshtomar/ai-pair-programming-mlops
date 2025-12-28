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
CMD ["python", "-c", "from src.predict import SentimentClassifier; c = SentimentClassifier.load('models'); print(c.predict('Docker test works!'))"]
