"""
Vaani — AI Speech-to-Text (Powered by Groq Whisper-Large-v3)
=============================================================
Fast, high-accuracy serverless speech transcription with state-of-the-art
whisper-large-v3 running on Groq LPUs.
Supports English, Hindi, and 90+ languages with timestamped segments.
Deployable on Vercel, Render, Koyeb, Docker, or local machine.
"""

import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3").strip()
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB hard limit on Groq free tier

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac",
    ".mp4", ".mpeg", ".mpga", ".oga",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaani")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vaani — AI Speech-to-Text",
    version="2.0.0",
    description="Blazing fast speech-to-text with Whisper-Large-v3 on Groq LPUs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Files & Frontend Routing
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the frontend application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "name": "Vaani — AI Speech-to-Text",
        "engine": "Groq LPU",
        "model": GROQ_MODEL,
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Health Probe
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Readiness and configuration probe."""
    return {
        "status": "ok",
        "engine": "groq",
        "model": GROQ_MODEL,
        "api_key_configured": bool(GROQ_API_KEY),
        "max_file_size_mb": 25,
    }


# ---------------------------------------------------------------------------
# Transcription Endpoint
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    lang: str | None = Query(None),
    x_groq_api_key: str | None = Header(None, alias="X-Groq-Api-Key"),
):
    """
    Accept an audio/video file (up to 25 MB), send to Groq's whisper-large-v3,
    and return timestamped transcript JSON.
    """
    # 1. Check API Key
    api_key = (
        (x_groq_api_key or "").strip()
        or GROQ_API_KEY
        or os.getenv("GROQ_API_KEY", "").strip()
    )
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "GROQ_API_KEY is not configured on the server. "
                "Get a free API key at https://console.groq.com/keys and set it in your environment."
            ),
        )

    # 2. Validate Extension
    filename = file.filename or "audio.mp3"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 3. Resolve Language
    target_lang = (language or lang or "auto").strip().lower()
    whisper_language = None if target_lang in ("auto", "", "none") else target_lang

    # 4. Stream & Check Size (25 MB max)
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(audio_bytes) > MAX_FILE_BYTES:
        size_mb = len(audio_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size ({size_mb:.1f} MB) exceeds the 25 MB limit for Groq Whisper. Please upload a smaller clip.",
        )

    # 5. Call Groq Whisper API
    try:
        client = Groq(api_key=api_key)
        logger.info(
            "Transcribing '%s' (%.2f MB, lang=%s) with %s on Groq...",
            filename,
            len(audio_bytes) / (1024 * 1024),
            whisper_language or "auto",
            GROQ_MODEL,
        )

        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=GROQ_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=whisper_language,
        )

    except Exception as exc:
        logger.exception("Groq transcription request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Groq API error: {exc}") from exc

    # 6. Parse Segments
    segments = []
    raw_segments = getattr(transcription, "segments", None) or []
    for seg in raw_segments:
        start_t = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        end_t = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", 0.0)
        text_val = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
        segments.append({
            "start": round(float(start_t or 0.0), 2),
            "end": round(float(end_t or 0.0), 2),
            "text": str(text_val or "").strip(),
        })

    # If Groq returned text without segment breakdown
    full_text = getattr(transcription, "text", "") or ""
    duration = getattr(transcription, "duration", None)
    if not segments and full_text.strip():
        segments.append({
            "start": 0.0,
            "end": round(float(duration or 0.0), 2),
            "text": full_text.strip(),
        })

    detected_lang = getattr(transcription, "language", None) or target_lang

    response_data = {
        "text": full_text.strip(),
        "language": detected_lang,
        "detected_language": detected_lang,
        "language_probability": 1.0,
        "duration": round(float(duration or 0.0), 2) if duration else 0.0,
        "segments": segments,
        "model": GROQ_MODEL,
        "engine": "groq",
    }

    return JSONResponse(
        content=response_data,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
