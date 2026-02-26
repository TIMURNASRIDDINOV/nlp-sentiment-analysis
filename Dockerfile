# ---------------------------------------------------------------------------
# Stage 1 — build / install dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install fastapi uvicorn

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Non-root user for security
RUN useradd -m appuser
USER appuser

# Default to web mode
ENV NLP_HOST=0.0.0.0
ENV NLP_PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

ENTRYPOINT ["python", "main.py", "--mode", "web"]
