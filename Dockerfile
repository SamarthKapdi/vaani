# ==========================================================================
#  VoiceScript — Speech-to-Text Uploader
#  Dockerfile (HF Spaces + local Docker compatible)
# ==========================================================================
FROM python:3.11-slim AS base

# Prevent interactive prompts and set locale
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- System dependencies -------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies -------------------------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Application code ----------------------------------------------------
COPY app/ ./app/

# ---- Pre-download the Whisper model so first request is fast ---------------
# Use 'small' for Hindi support; 'base' is English-only quality
ARG MODEL_SIZE=small
ENV MODEL_SIZE=${MODEL_SIZE}

# Cache model in a world-writable path so non-root users can access it
ENV HF_HOME=/tmp/hf_cache
RUN mkdir -p /tmp/hf_cache && \
    python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='int8')" && \
    echo "Model '${MODEL_SIZE}' cached successfully." && \
    chmod -R 777 /tmp/hf_cache

# ---- Non-root user (required by HF Spaces) -------------------------------
RUN useradd -m -u 1000 appuser
USER appuser

# ---- Runtime --------------------------------------------------------------
# HF Spaces routes traffic exclusively to port 7860
# For local dev, override with PORT=8000 in docker-compose.yml
ENV PORT=7860 \
    MAX_DURATION_SECONDS=180
EXPOSE ${PORT}

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
