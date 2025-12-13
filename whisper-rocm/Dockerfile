# Whisper ASR with ROCm support (AMD GPU)
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install Whisper and faster-whisper
RUN pip install --no-cache-dir \
    openai-whisper \
    faster-whisper \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic

# Copy server code
COPY server.py .

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Run server
CMD ["python", "server.py"]
