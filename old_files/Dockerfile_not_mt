FROM python:3.11-slim

# Set environment variables to keep Python from buffering logs
ENV PYTHONUNBUFFERED=1
WORKDIR /app/

# Install minimal system dependencies for AI libraries
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libpq-dev gcc netcat-traditional && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Add non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 5000
EXPOSE 5000

CMD ["sh", "-c", "while ! nc -z db 5432; do sleep 1; done; uvicorn app:app --reload --ssl-keyfile ./localhost-key.pem --ssl-certfile ./localhost.pem --host 0.0.0.0 --port 8000"]
#uvicorn api_auth_fastapi:app --reload --ssl-keyfile ./localhost-key.pem --ssl-certfile ./localhost.pem
#CMD ["sh", "-c", "while ! nc -z db 5432; do sleep 1; done; python -m src.rag_helper && uvicorn app:app --host 0.0.0.0 --port 8000"]
