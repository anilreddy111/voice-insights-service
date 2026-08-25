# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# curl: container healthcheck. No ffmpeg/libsndfile apt packages needed:
# soundfile and PyAV wheels bundle their native libs, keeping the image slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /srv/app

# CPU-only torch wheels keep the image ~2GB smaller than the CUDA default.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

ARG MODEL_ID=audeering/wav2vec2-large-robust-6-ft-age-gender
ENV VIS_MODEL_ID=${MODEL_ID}

# Bake model weights into the image so `docker compose up` needs no network.
# (~500MB for the 6-layer variant; the 24-layer variant is ~1.3GB.)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

RUN useradd -m -u 10001 appuser && chown -R appuser /opt/hf /srv/app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
