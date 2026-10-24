# OmniVoice TTS service (CPU only) — optional, non-commercial

> The default engine is now Kokoro-82M (`tts_service/`, Apache 2.0). Use this service only for non-commercial work or for languages Kokoro and OpenVoice don't cover.

A small HTTP service around [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) that the Django studio calls to narrate videos. No GPU needed.

## License first

The OmniVoice **code** is Apache 2.0, but the **model weights are CC-BY-NC (non-commercial)** because of their training data. If your videos earn money (ads, sponsorships, client work), that's commercial use the weights don't allow. The Django side only talks to this service through `trends/services/tts_client.py`, so you can swap in a commercially licensed engine there without touching the video pipeline.

Only clone voices you own or have permission to use. The clone endpoint refuses uploads without `consent=true`.

## What to expect on CPU

OmniVoice is fast on GPUs; on CPU it's slower than real time. Expect roughly 2–6 seconds of compute per second of speech on an 8-vCPU instance at 16 diffusion steps, depending on the CPU. A 60-second video's narration therefore takes a few minutes. That's fine for background video production, but not for live voice.

Levers, in order of impact:
- `OMNIVOICE_NUM_STEP`: 16 is the default here; 8 is about twice as fast with some quality loss; 32 is best quality.
- More vCPUs per instance (8 is the Cloud Run maximum).
- On a Compute Engine `c4`/`c3` VM (Intel AMX), try `OMNIVOICE_DTYPE=bfloat16` and compare with `/benchmark`.

Run `/benchmark` after deploying to get your real numbers.

## Deploy to Cloud Run

Needs the gcloud CLI, logged in, with billing enabled on the project.

```bash
cd tts_omnivoice
export PROJECT_ID=your-project REGION=europe-west1
export TTS_API_KEY=$(openssl rand -hex 24)   # keep this; Django needs the same value
./deploy_cloud_run.sh
```

The script:
1. Builds the image in Cloud Build with CPU-only PyTorch and the model weights baked in (~5 GB), so cold starts don't download anything. The first build takes a while.
2. Stores the API key in Secret Manager.
3. Creates a Cloud Storage bucket mounted at `/data/voices`, so saved voices survive restarts and are shared by all instances.
4. Deploys with 8 vCPU, 16 GiB, **concurrency 1** (each instance gives all its CPUs to one job), **min instances 0** (scales to zero, so you pay nothing while idle), and a 60-minute request timeout.

Then add to your Django `.env`:

```
OMNIVOICE_SERVICE_URL=https://omnivoice-tts-xxxxx.run.app
OMNIVOICE_API_KEY=<same key>
```

The first request after idle waits for a cold start (container start plus loading weights, typically 1–2 minutes). The Django client retries while Cloud Run spins up. If you want instant responses, set `--min-instances 1`, but then you pay for an always-on 8-vCPU instance.

## Or run on a Compute Engine VM

Better if you produce a lot: a Spot VM is much cheaper than on-demand and runs continuously.

```bash
# after running deploy_cloud_run.sh once (it builds and pushes the image)
PROJECT_ID=your-project ZONE=europe-west1-b TTS_API_KEY=... ./vm/create_vm.sh
```

That creates a `c4-standard-8` Spot VM (change `MACHINE=` if the zone lacks it), installs Docker, runs the container on port 8080, and opens the port only to your current IP. Spot VMs can be stopped by Google at any time; the startup script restarts the service when the VM comes back.

## API

All endpoints except `/health` need `Authorization: Bearer $TTS_API_KEY`.

| Method | Path | Body |
|---|---|---|
| GET | `/health` | — |
| POST | `/tts` | `{"text": "...", "voice_id": "..."}` or `{"text": "...", "instruct": "female, british accent"}` → WAV |
| POST | `/tts/segments` | `{"segments": ["scene 1", "scene 2"], ...}` → JSON, one WAV per segment, same voice throughout |
| GET | `/voices` | — |
| POST | `/voices/design` | `{"name": "Narrator", "instruct": "male, low pitch, american accent"}` |
| POST | `/voices` | multipart: `name`, `file` (3–10 s clip), `ref_text` (optional), `consent=true` |
| DELETE | `/voices/{id}` | — |
| POST | `/benchmark` | — → real-time factor on this machine |

**Consistent voice:** voice descriptions (`instruct`) are random each time, so the service designs the voice once for the first sentence, then clones that sample for every sentence after. Saving a designed voice (`/voices/design`) and using its `voice_id` keeps the same voice across all your videos.
# tts_omnivoice
