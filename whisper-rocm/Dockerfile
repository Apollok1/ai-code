# Whisper ASR with ROCm 7.0 for RDNA 4 (gfx1201 - RX 9070 XT)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch with ROCm 7.0 from gfx120X repository (for RX 9070 XT)
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir \
    --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx120X-all/ \
    torch torchvision torchaudio

# Install Whisper and dependencies
RUN pip install --no-cache-dir \
    openai-whisper \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic

# Copy server
COPY server.py .

# Expose port
EXPOSE 9000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HSA_OVERRIDE_GFX_VERSION=11.0.1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:9000/health || exit 1

# Run server
CMD ["python", "server.py"]
