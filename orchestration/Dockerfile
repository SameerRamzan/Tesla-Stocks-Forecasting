# Orchestration Dockerfile (for Prefect flows)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e . prefect

# Copy orchestration code
COPY orchestration/ orchestration/
COPY backend/app/ backend/app/
COPY data/ data/
COPY models/ models/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Run Prefect worker
CMD ["prefect", "worker", "start", "--pool", "default-pool"]