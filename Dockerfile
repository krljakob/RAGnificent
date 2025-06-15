# Simple Python-only build for RAGnificent
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy requirements and install Python dependencies
COPY requirements-docker.txt ./
RUN uv pip install --system --no-cache-dir -r requirements-docker.txt

# Copy application source code
COPY RAGnificent/ ./RAGnificent/
COPY pyproject.toml README.md ./

# Install the package in editable mode
RUN uv pip install --system -e .

# Create non-root user for security
RUN groupadd -r ragnificent && useradd -r -g ragnificent ragnificent

# Create directories for data and cache
RUN mkdir -p /app/data /app/cache /app/logs /app/config \
    && chown -R ragnificent:ragnificent /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV RAGNIFICENT_CONFIG_DIR=/app/config
ENV RAGNIFICENT_DATA_DIR=/app/data
ENV RAGNIFICENT_CACHE_DIR=/app/cache
ENV RAGNIFICENT_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER ragnificent

# Expose port for API service
EXPOSE 8000

# Default command
CMD ["python", "-m", "RAGnificent.api"]