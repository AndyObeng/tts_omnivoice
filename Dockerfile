FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VOICES_DIR=/data/voices

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only PyTorch (≈200 MB instead of several GB of CUDA libraries)
RUN pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Bake the weights into the image so a cold start doesn't download ~3.5 GB.
# Loading once also fetches any extra files the model pulls on first use.
ARG BAKE_MODEL=1
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu')"; \
    fi

COPY app.py .
EXPOSE 8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 75
