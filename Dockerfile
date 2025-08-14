# AI Solutions Lab Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# BUILD STAGE
# =============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# =============================================================================
# PRODUCTION STAGE
# =============================================================================
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r ailab && useradd -r -g ailab ailab

# Create application directories
RUN mkdir -p /app/data/docs /app/data/index /app/models /app/logs

# Set ownership
RUN chown -R ailab:ailab /app

# Switch to non-root user
USER ailab

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=ailab:ailab src/ ./src/
COPY --chown=ailab:ailab pyproject.toml .
COPY --chown=ailab:ailab README.md .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/docs data/index models logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.ai_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# DEVELOPMENT STAGE (Optional)
# =============================================================================
FROM production as development

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional development tools
RUN pip install ipython jupyter

# Set development environment
ENV DEBUG=true
ENV LOG_LEVEL=DEBUG

# Development command with reload
CMD ["uvicorn", "src.ai_lab.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

# Environment variables for configuration
ENV MODEL_BACKEND=local
ENV EMBEDDINGS_BACKEND=local
ENV VECTOR_STORE=faiss
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=false
ENV LOG_LEVEL=INFO

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Labels for metadata
LABEL maintainer="AI Solutions Engineer"
LABEL version="0.1.0"
LABEL description="AI Solutions Lab - A production-minded lab demonstrating AI Solutions Engineering skills"
LABEL org.opencontainers.image.source="https://github.com/yourusername/ai-solutions-lab"
LABEL org.opencontainers.image.licenses="MIT"
