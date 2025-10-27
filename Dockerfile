# DDSP Neural Cello - Production Dockerfile for Fly.io
# embracingearth.space

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements-production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application code
COPY ddsp_server.py .
COPY ddsp_trainer_integration.py .
COPY ddsp_sample_based.py .
COPY ddsp_trainer.py .
COPY audio_processor.py .

# Create directories for output and models
RUN mkdir -p output models

# Copy trained models if they exist (include default trained model)
COPY models/ ./models/

# Copy frontend static files
COPY public/ ./public/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["python", "ddsp_server.py"]
