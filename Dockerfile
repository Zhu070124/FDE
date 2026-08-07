# CQUPT AI Assistant — Docker Image
# Multi-stage build for minimal production image

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first for better layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --target=/app/deps -r /app/backend/requirements.txt

# --- Production Stage ---
FROM python:3.11-slim AS production

LABEL org.opencontainers.image.title="CQUPT AI Assistant"
LABEL org.opencontainers.image.description="RAG-powered student assistant for Chongqing University of Posts and Telecommunications"
LABEL org.opencontainers.image.version="0.2.0"

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy dependencies from builder
COPY --from=builder --chown=appuser:appuser /app/deps /usr/local/lib/python3.11/site-packages/

# Copy application code
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser frontend/ /app/frontend/
COPY --chown=appuser:appuser data/documents/README.md /app/data/documents/README.md

# Create data directories for writable data
RUN mkdir -p /app/data/store /app/data/logs /app/data/reflections /app/data/vector_db \
    && chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health').raise_for_status()" || exit 1

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
