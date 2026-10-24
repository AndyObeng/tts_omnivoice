#!/usr/bin/env bash
# Alternative: run the same container on a Compute Engine VM (CPU only).
# Good when you generate a lot: a Spot VM costs a fraction of on-demand, and c4/c3 machines have
# Intel AMX, which can speed up bfloat16 inference (try OMNIVOICE_DTYPE=bfloat16 and compare /benchmark).
# Usage: PROJECT_ID=my-project ZONE=europe-west1-b TTS_API_KEY=... ./create_vm.sh
set -euo pipefail
PROJECT_ID="${PROJECT_ID:?}"
ZONE="${ZONE:-europe-west1-b}"
REGION="${ZONE%-*}"
NAME="${NAME:-omnivoice-vm}"
MACHINE="${MACHINE:-c4-standard-8}"     # fall back to c3-standard-8 or n2-standard-8 if unavailable in the zone
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/tts/omnivoice-tts:latest}"
TTS_API_KEY="${TTS_API_KEY:?}"
DTYPE="${DTYPE:-float32}"

gcloud compute instances create "$NAME" \
  --project "$PROJECT_ID" --zone "$ZONE" --machine-type "$MACHINE" \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=50GB \
  --scopes=cloud-platform --tags=omnivoice \
  --metadata="tts-api-key=${TTS_API_KEY},tts-image=${IMAGE},tts-dtype=${DTYPE}" \
  --metadata-from-file=startup-script="$(dirname "$0")/startup.sh"

# Only open the port to your own IP (replace with your server/home address)
MYIP="$(curl -s https://ifconfig.me)/32"
gcloud compute firewall-rules create allow-omnivoice --project "$PROJECT_ID" \
  --allow tcp:8080 --target-tags omnivoice --source-ranges "$MYIP" 2>/dev/null || true

echo "When it's up (a few minutes): curl http://$(gcloud compute instances describe "$NAME" --zone "$ZONE" \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)'):8080/health"
