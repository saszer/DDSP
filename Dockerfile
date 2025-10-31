# DDSP Neural Cello - Production Dockerfile for Fly.io
# embracingearth.space

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js for Tailwind CSS
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libffi-dev \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements-production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy package files for Tailwind CSS build
COPY package.json package-lock.json tailwind.config.js postcss.config.js ./

# Install Node.js dependencies for Tailwind CSS (including dev deps for build)
RUN npm install --legacy-peer-deps

# Copy Tailwind CSS input file and public directory structure
COPY public/input.css ./public/
COPY public/index.html ./public/

# Build Tailwind CSS (create output directory first)
RUN mkdir -p public && npm run build:css || (echo "CSS build failed, will use inline styles" && echo "/* Fallback CSS */" > public/styles.css)

# Copy application code (hybrid server)
COPY ddsp_server_hybrid.py .

# Create directories for output and models
RUN mkdir -p output models

# Copy trained models if they exist (using RUN with shell to handle optional directory)
RUN if [ -d models ] && [ "$(ls -A models)" ]; then \
        echo "Models directory found, copying..."; \
        cp -r models/* /app/models/ || true; \
    else \
        echo "No models directory or empty, skipping..."; \
        mkdir -p /app/models; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Start the hybrid server
CMD ["python", "ddsp_server_hybrid.py"]
