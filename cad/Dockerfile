# Dockerfile for CAD Estimator Panel
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PostgreSQL client
    libpq-dev \
    gcc \
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (or mount it as volume)
# The src/ directory will be mounted from ai-code/src/
# COPY src/ ./src/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "/app/src/cad/presentation/app.py", \
     "--server.headless=true", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.fileWatcherType=auto", \
     "--client.toolbarMode=minimal"]
