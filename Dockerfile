# Slim, production-ready container for the Acquirer Engine API.
# Two-stage build keeps the final image small and cache-friendly.

# ----- Builder stage -----
FROM python:3.11-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ----- Runtime stage -----
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder (no build tools in final image)
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY run_api.py .

# Security: run as non-root
RUN useradd -r -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

# Healthcheck for orchestrators (ECS, K8s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "run_api.py"]
