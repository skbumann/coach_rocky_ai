# ── Dockerfile ────────────────────────────────────────────────────────────────
# Production build — SSL is handled by ALB, not uvicorn.
# For local dev with self-signed certs, override CMD in docker-compose.yml.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app/

RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Production: plain HTTP — TLS is terminated at the ALB.
# For local dev with self-signed certs, override this in docker-compose.yml:
#CMD [uvicorn app:app --reload --ssl-keyfile ./localhost-key.pem --ssl-certfile ./localhost.pem --host 0.0.0.0 --port 8000]
# Replace your current CMD with this:
CMD ["uvicorn", "app:app", "--reload", "--ssl-keyfile", "./localhost-key.pem", "--ssl-certfile", "./localhost.pem", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
