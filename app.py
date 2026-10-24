"""OmniVoice text-to-speech as a CPU-only HTTP service (Cloud Run or any VM).

Endpoints (all but /health need `Authorization: Bearer $TTS_API_KEY` when TTS_API_KEY is set):
  GET    /health              model status, threads, dtype
  POST   /tts                 {text, voice_id|instruct, speed, num_step} -> audio/wav
  POST   /tts/segments        {segments: [...], ...} -> JSON with one wav per segment, same voice throughout
  GET    /voices              saved voices
  POST   /voices              multipart clone from a reference clip (consent required)
  POST   /voices/design       {name, instruct} -> designs a voice once and saves it for consistent reuse
  DELETE /voices/{id}
  POST   /benchmark           quick real-time-factor measurement on this machine

Model license: OmniVoice weights are CC-BY-NC (non-commercial). Code here is yours to use.
"""
import base64
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

API_KEY = os.getenv("TTS_API_KEY", "")
MODEL_ID = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DTYPE_NAME = os.getenv("OMNIVOICE_DTYPE", "float32")  # try bfloat16 on CPUs with AMX (GCE c3/c4)
DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}[DTYPE_NAME]
DEFAULT_STEPS = int(os.getenv("OMNIVOICE_NUM_STEP", "16"))  # 32 = best quality, 16 = ~2x faster, 8 = draft
THREADS = int(os.getenv("TORCH_THREADS") or os.cpu_count() or 4)
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/data/voices"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "220"))
SR = 24000
PAUSE = np.zeros(int(SR * 0.18), dtype=np.float32)

torch.set_num_threads(THREADS)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="OmniVoice CPU TTS", version="1.0")

_model = None
_model_error = None
_load_lock = threading.Lock()
_gen_lock = threading.Lock()  # one synthesis at a time per instance; scale out with more instances
_prompt_cache = {}
_loaded_at = None


def _load():
    global _model, _model_error, _loaded_at
    with _load_lock:
        if _model is not None:
            return _model
        try:
            from omnivoice import OmniVoice
            t = time.time()
            _model = OmniVoice.from_pretrained(MODEL_ID, device_map="cpu", dtype=DTYPE)
            _loaded_at = round(time.time() - t, 1)
            _model_error = None
        except Exception as e:  # surfaced via /health and request errors
            _model_error = f"{e.__class__.__name__}: {e}"
            raise
    return _model


@app.on_event("startup")
def _warm():
    # Start listening immediately (Cloud Run startup probe) and load weights in the background.
    threading.Thread(target=lambda: _safe_load(), daemon=True).start()


def _safe_load():
    try:
        _load()
    except Exception:
        pass


def auth(authorization: str = Header(default="")):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Missing or wrong bearer token")


# ------------------------------------------------------------------ helpers --

_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def chunk_text(text, limit=MAX_CHUNK_CHARS):
    """Split on sentence ends, break over-long sentences at a comma/space, then pack into <= limit chars."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    pieces = []
    for s in (x.strip() for x in _SENT.split(text)):
        while len(s) > limit:
            cut = max(s.rfind(",", 0, limit), s.rfind(" ", 0, limit))
            if cut < limit // 3:
                cut = limit - 1
            pieces.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            pieces.append(s)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def _wav_bytes(audio):
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _voice_meta(voice_id):
    p = VOICES_DIR / f"{voice_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _prompt(voice_id):
    if voice_id in _prompt_cache:
        return _prompt_cache[voice_id]
    path = VOICES_DIR / f"{voice_id}.pt"
    if not path.exists():
        raise HTTPException(404, f"Unknown voice_id {voice_id}")
    from omnivoice import VoiceClonePrompt
    _prompt_cache[voice_id] = VoiceClonePrompt.load(str(path))
    return _prompt_cache[voice_id]


def _gen(model, text, *, prompt=None, instruct=None, speed=1.0, num_step=None):
    kwargs = {"text": text, "num_step": num_step or DEFAULT_STEPS, "speed": speed}
    if prompt is not None:
        kwargs["voice_clone_prompt"] = prompt
    elif instruct:
        kwargs["instruct"] = instruct
    return np.asarray(model.generate(**kwargs)[0], dtype=np.float32)


def _prompt_from_audio(model, audio, text):
    """Turn a generated sample into a clone prompt so every later sentence uses the same voice."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, audio, SR)
        return model.create_voice_clone_prompt(ref_audio=f.name, ref_text=text)


def synthesize_segments(segments, voice_id=None, instruct=None, speed=1.0, num_step=None):
    """Synthesize several texts in one consistent voice. Returns (list of float32 arrays, seconds spent)."""
    model = _load()
    t0 = time.time()
    with _gen_lock:
        prompt = _prompt(voice_id) if voice_id else None
        outputs = []
        for seg in segments:
            pieces = []
            for chunk in chunk_text(seg):
                if prompt is None:
                    # First chunk sets the voice (designed or automatic); clone it for everything after.
                    audio = _gen(model, chunk, instruct=instruct, speed=speed, num_step=num_step)
                    prompt = _prompt_from_audio(model, audio, chunk)
                else:
                    audio = _gen(model, chunk, prompt=prompt, speed=speed, num_step=num_step)
                pieces += [audio, PAUSE]
            outputs.append(np.concatenate(pieces) if pieces else PAUSE.copy())
    return outputs, time.time() - t0


# ---------------------------------------------------------------- endpoints --

class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    voice_id: str | None = None
    instruct: str | None = Field(default=None, description='e.g. "female, low pitch, british accent"')
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    num_step: int | None = Field(default=None, ge=4, le=64)


class SegmentsRequest(BaseModel):
    segments: list[str] = Field(min_length=1, max_length=40)
    voice_id: str | None = None
    instruct: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    num_step: int | None = Field(default=None, ge=4, le=64)


class DesignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    instruct: str = Field(min_length=3, max_length=200)
    sample_text: str = Field(default="Welcome back. Here is what everyone is talking about today, and why it matters.",
                             max_length=300)


@app.get("/health")
def health():
    return {"loaded": _model is not None, "load_seconds": _loaded_at, "error": _model_error,
            "device": "cpu", "dtype": DTYPE_NAME, "threads": THREADS, "default_num_step": DEFAULT_STEPS,
            "model": MODEL_ID, "license_note": "OmniVoice weights are CC-BY-NC (non-commercial)"}


@app.post("/tts", dependencies=[Depends(auth)])
def tts(req: TTSRequest):
    audios, spent = synthesize_segments([req.text], req.voice_id, req.instruct, req.speed, req.num_step)
    duration = len(audios[0]) / SR
    return Response(_wav_bytes(audios[0]), media_type="audio/wav",
                    headers={"X-Audio-Duration": f"{duration:.2f}", "X-RTF": f"{spent / max(duration, 0.01):.2f}"})


@app.post("/tts/segments", dependencies=[Depends(auth)])
def tts_segments(req: SegmentsRequest):
    audios, spent = synthesize_segments(req.segments, req.voice_id, req.instruct, req.speed, req.num_step)
    total = sum(len(a) for a in audios) / SR
    return {"sample_rate": SR, "seconds_spent": round(spent, 1), "rtf": round(spent / max(total, 0.01), 2),
            "segments": [{"duration": round(len(a) / SR, 3), "wav_b64": base64.b64encode(_wav_bytes(a)).decode()}
                         for a in audios]}


@app.get("/voices", dependencies=[Depends(auth)])
def list_voices():
    return [json.loads(p.read_text()) for p in sorted(VOICES_DIR.glob("*.json"))]


@app.post("/voices", dependencies=[Depends(auth)])
async def clone_voice(name: str = Form(...), consent: bool = Form(False), ref_text: str = Form(""),
                      file: UploadFile = File(...)):
    if not consent:
        raise HTTPException(400, "Confirm you have the speaker's permission to clone this voice (consent=true).")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Reference clip is too large; use 3–10 seconds of audio.")
    model = _load()
    suffix = Path(file.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        with _gen_lock:
            prompt = model.create_voice_clone_prompt(ref_audio=f.name, ref_text=ref_text.strip() or None)
    return _save_voice(prompt, name, kind="cloned")


@app.post("/voices/design", dependencies=[Depends(auth)])
def design_voice(req: DesignRequest):
    model = _load()
    with _gen_lock:
        audio = _gen(model, req.sample_text, instruct=req.instruct)
        prompt = _prompt_from_audio(model, audio, req.sample_text)
    meta = _save_voice(prompt, req.name, kind="designed", instruct=req.instruct)
    (VOICES_DIR / f"{meta['id']}.wav").write_bytes(_wav_bytes(audio))
    return meta


def _save_voice(prompt, name, **extra):
    vid = uuid.uuid4().hex[:12]
    prompt.save(str(VOICES_DIR / f"{vid}.pt"))
    meta = {"id": vid, "name": name, "created": int(time.time()), **extra}
    (VOICES_DIR / f"{vid}.json").write_text(json.dumps(meta))
    _prompt_cache[vid] = prompt
    return meta


@app.delete("/voices/{voice_id}", dependencies=[Depends(auth)])
def delete_voice(voice_id: str):
    if not re.fullmatch(r"[0-9a-f]{12}", voice_id):
        raise HTTPException(400, "Bad voice id")
    for ext in (".pt", ".json", ".wav"):
        (VOICES_DIR / f"{voice_id}{ext}").unlink(missing_ok=True)
    _prompt_cache.pop(voice_id, None)
    return {"deleted": voice_id}


@app.post("/benchmark", dependencies=[Depends(auth)])
def benchmark(num_step: int | None = None):
    text = ("Prediction markets moved sharply overnight, as traders reacted to the latest budget vote "
            "and new inflation figures released this morning.")
    audios, spent = synthesize_segments([text], instruct="female, british accent", num_step=num_step)
    dur = len(audios[0]) / SR
    return {"audio_seconds": round(dur, 2), "seconds_spent": round(spent, 2), "rtf": round(spent / dur, 2),
            "num_step": num_step or DEFAULT_STEPS, "threads": THREADS, "dtype": DTYPE_NAME}
