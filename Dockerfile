# ── Stage: Python app ─────────────────────────────────────────────────────────
# Ollama runs as a SEPARATE container (see docker-compose.yml).
# This image contains only the FastAPI web server.
FROM python:3.11-slim

# Metadata
LABEL maintainer="FAST NUCES — Generative AI Project 2026"
LABEL description="AI Interview Simulator — FastAPI backend"

# Prevents Python from writing .pyc files and enables stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ollama host — overridden in docker-compose to point at the ollama service
ENV OLLAMA_HOST=http://ollama:11434

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create data directory (dataset + benchmark results mount here)
RUN mkdir -p data/processed benchmark/results

# Expose FastAPI port
EXPOSE 8000

# Health check — hits the root endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
