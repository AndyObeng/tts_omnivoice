#!/usr/bin/env bash
# Deploy OmniVoice TTS to Cloud Run (CPU only).
# Usage: PROJECT_ID=my-project REGION=europe-west1 TTS_API_KEY=$(openssl rand -hex 24) ./deploy_cloud_run.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-europe-west1}"          # nearest to Ghana with broad support; africa-south1 also works
SERVICE="${SERVICE:-omnivoice-tts}"
REPO="${REPO:-tts}"
TTS_API_KEY="${TTS_API_KEY:?set TTS_API_KEY (any long random string; Django uses it as OMNIVOICE_API_KEY)}"
BUCKET="${BUCKET:-${PROJECT_ID}-omnivoice-voices}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-16Gi}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com

gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

# Build in Cloud Build (the image is ~5 GB with weights baked in, so use a bigger builder)
gcloud builds submit --tag "$IMAGE" --machine-type=e2-highcpu-8 --disk-size=100 --timeout=3600 .

# API key as a secret
if gcloud secrets describe omnivoice-api-key >/dev/null 2>&1; then
  printf "%s" "$TTS_API_KEY" | gcloud secrets versions add omnivoice-api-key --data-file=-
else
  printf "%s" "$TTS_API_KEY" | gcloud secrets create omnivoice-api-key --data-file=-
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding omnivoice-api-key \
  --member="serviceAccount:${RUNTIME_SA}" --role=roles/secretmanager.secretAccessor >/dev/null

# Bucket that persists saved voices across instances and restarts
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --location "$REGION" --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" --role=roles/storage.objectAdmin >/dev/null

# concurrency=1: each instance gives all its CPUs to one synthesis; more videos = more instances.
# min-instances=0: scales to zero, so you pay nothing while idle (first request then waits for a cold start).
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" \
  --cpu "$CPU" --memory "$MEMORY" --cpu-boost \
  --concurrency 1 --min-instances 0 --max-instances "$MAX_INSTANCES" \
  --timeout 3600 --execution-environment gen2 \
  --allow-unauthenticated \
  --set-secrets "TTS_API_KEY=omnivoice-api-key:latest" \
  --set-env-vars "OMNIVOICE_NUM_STEP=16,TORCH_THREADS=${CPU},VOICES_DIR=/data/voices" \
  --add-volume "name=voices,type=cloud-storage,bucket=${BUCKET}" \
  --add-volume-mount "volume=voices,mount-path=/data/voices"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "Deployed: $URL"
echo "Add to your Django .env:"
echo "  OMNIVOICE_SERVICE_URL=$URL"
echo "  OMNIVOICE_API_KEY=<the key you used>"
echo
echo "Measure speed:  curl -s -X POST -H \"Authorization: Bearer \$TTS_API_KEY\" $URL/benchmark"
