# Lightweight Python image for PDF extraction microservice
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pdfplumber and Flask dependencies
RUN pip install --no-cache-dir \
    flask==3.1.0 \
    pdfplumber==0.11.4 \
    gunicorn==23.0.0

# Copy the Flask service
COPY pdf_service.py .

# Expose port for the service
EXPOSE 9001

# Run with gunicorn for production-ready server
# 2 workers for handling concurrent requests
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:9001", "--timeout", "60", "pdf_service:app"]
