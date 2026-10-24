#!/bin/bash
# Runs on every boot: installs Docker once, then (re)starts the TTS container.
set -e
md() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
if ! command -v docker >/dev/null; then
  apt-get update && apt-get install -y docker.io
fi
IMAGE="$(md tts-image)"; KEY="$(md tts-api-key)"; DTYPE="$(md tts-dtype)"
REGISTRY="${IMAGE%%/*}"
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY}"
docker pull "$IMAGE"
mkdir -p /var/omnivoice/voices
docker rm -f omnivoice 2>/dev/null || true
docker run -d --name omnivoice --restart unless-stopped -p 8080:8080 \
  -e TTS_API_KEY="$KEY" -e OMNIVOICE_DTYPE="${DTYPE:-float32}" -e OMNIVOICE_NUM_STEP=16 \
  -e TORCH_THREADS="$(nproc)" -v /var/omnivoice/voices:/data/voices "$IMAGE"
