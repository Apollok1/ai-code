# Dockerfile for Document Converter Service
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR and document processing
    tesseract-ocr \
    tesseract-ocr-pol \
    tesseract-ocr-eng \
    poppler-utils \
    # Media processing
    ffmpeg \
    # Image processing
    libgl1 \
    libglib2.0-0 \
    # Build tools (needed for some Python packages)
    gcc \
    g++ \
    # Health check
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create outputs directory
RUN mkdir -p /app/outputs

# Expose Streamlit port
EXPOSE 8502

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "/app/app/converter.py", \
     "--server.headless=true", \
     "--server.address=0.0.0.0", \
     "--server.port=8502", \
     "--server.fileWatcherType=none", \
     "--client.toolbarMode=minimal"]
