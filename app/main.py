"""
Speech-to-Text Uploader — FastAPI Backend
==========================================
Accepts audio/video uploads, extracts & normalizes audio via FFmpeg,
and transcribes with Faster-Whisper running locally.
Supports English, Hindi, and 90+ languages with tunable accuracy levers.
"""

import os
import uuid
import logging
import tempfile
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration & Accuracy Levers
# ---------------------------------------------------------------------------
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")       # int8 for CPU, float16 for GPU
DEVICE = os.getenv("DEVICE", "cpu")                     # "cpu" or "cuda"
MAX_FILE_BYTES = 500 * 1024 * 1024                      # 500 MB
MAX_DURATION_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "180"))  # 0 = no limit

# Accuracy Levers
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
VAD_FILTER = os.getenv("VAD_FILTER", "true").lower() in ("true", "1", "yes")
WORD_TIMESTAMPS = os.getenv("WORD_TIMESTAMPS", "false").lower() in ("true", "1", "yes")
CONDITION_ON_PREVIOUS_TEXT = os.getenv("CONDITION_ON_PREVIOUS_TEXT", "true").lower() in ("true", "1", "yes")
INITIAL_PROMPT = os.getenv("INITIAL_PROMPT", "")
TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Pinned Detection Thresholds
NO_SPEECH_THRESHOLD = float(os.getenv("NO_SPEECH_THRESHOLD", "0.6"))
LOG_PROB_THRESHOLD = float(os.getenv("LOG_PROB_THRESHOLD", "-1.0"))
COMPRESSION_RATIO_THRESHOLD = float(os.getenv("COMPRESSION_RATIO_THRESHOLD", "2.4"))


def _default_threads() -> int:
    """Return a container-aware CPU thread count.

    Prefers os.sched_getaffinity() which respects cpuset masks set by
    Docker's ``--cpus`` / ``cpus:`` limits. Falls back to os.cpu_count()
    on platforms that lack sched_getaffinity (Windows, macOS, exotic).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


# Set CPU_THREADS=0 to let CTranslate2 auto-tune thread count
CPU_THREADS = int(os.getenv("CPU_THREADS", str(_default_threads())))

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mkv", ".avi", ".mov"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-to-text")

# ---------------------------------------------------------------------------
# Whisper model (loaded once at startup)
# ---------------------------------------------------------------------------
whisper_model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Whisper model once at startup and release on shutdown."""
    global whisper_model

    # Capability warning check
    logger.info("=== Model Capabilities Matrix ===")
    logger.info("base:           English OK, Hindi poor — do not use for Hindi")
    logger.info("small:          Hindi usable, accents will be approximate")
    logger.info("medium:         Hindi good (recommended for CPU deployments with 8GB+ RAM)")
    logger.info("large-v3-turbo: Hindi very good, ~6GB RAM, best speed/accuracy tradeoff")
    logger.info("large-v3:       Hindi best, ~10GB RAM")

    if MODEL_SIZE in ("base", "tiny"):
        logger.warning(
            "WARNING: Current MODEL_SIZE is '%s'. Hindi transcription accuracy is poor on base/tiny models. "
            "For Hindi, use at least 'small' or 'medium'.",
            MODEL_SIZE,
        )

    logger.info(
        "Loading Faster-Whisper model '%s' (device=%s, compute_type=%s, threads=%d) …",
        MODEL_SIZE, DEVICE, COMPUTE_TYPE, CPU_THREADS,
    )
    whisper_model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
    )
    logger.info("Model '%s' loaded successfully.", MODEL_SIZE)
    yield
    logger.info("Shutting down — releasing model resources.")
    whisper_model = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vaani — AI Speech-to-Text",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Readiness / liveness probe."""
    return {"status": "ok", "model": MODEL_SIZE, "compute_type": COMPUTE_TYPE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    lang: str | None = Query(None),
):
    """
    Accept an audio or video file, extract & normalize audio with FFmpeg,
    run Faster-Whisper with configurable accuracy levers, and return
    timestamped transcript JSON.
    """
    # --- Validate extension ------------------------------------------------
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # --- Language parameter resolution ------------------------------------
    target_lang = (language or lang or "auto").strip().lower()
    whisper_language = None if target_lang in ("auto", "", "none") else target_lang

    if whisper_language == "hi" and MODEL_SIZE in ("base", "tiny"):
        logger.warning(
            "Hindi transcription requested on model '%s'. Accuracy may be poor.",
            MODEL_SIZE,
        )

    # --- Stream file to disk & enforce size limit -------------------------
    tmp_dir = tempfile.mkdtemp()
    job_id = uuid.uuid4().hex[:12]
    input_path = Path(tmp_dir) / f"{job_id}_input{ext}"
    wav_path = Path(tmp_dir) / f"{job_id}.wav"

    try:
        # Write upload to disk in chunks (never hold full file in RAM)
        CHUNK_SIZE = 1024 * 1024  # 1 MB
        bytes_written = 0
        with open(input_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds 500 MB limit.",
                    )
                f.write(chunk)

        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # --- FFmpeg: convert to 16 kHz mono WAV + loudness normalization ---
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",                                      # discard video
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",     # loudness normalization
            "-acodec", "pcm_s16le",                     # 16-bit PCM
            "-ar", "16000",                             # 16 kHz
            "-ac", "1",                                 # mono
            str(wav_path),
        ]
        logger.info("Running FFmpeg: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("FFmpeg stderr: %s", result.stderr)
            raise HTTPException(status_code=422, detail="FFmpeg failed to extract audio.")

        # --- Duration guard ------------------------------------------------
        if MAX_DURATION_SECONDS > 0:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of", "csv=p=0", str(wav_path)],
                    capture_output=True, text=True, timeout=30,
                )
                audio_duration = float(probe.stdout.strip())
                if audio_duration > MAX_DURATION_SECONDS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio duration ({audio_duration:.0f}s) exceeds the {MAX_DURATION_SECONDS}s limit. "
                               f"Please upload a shorter clip.",
                    )
            except (ValueError, subprocess.TimeoutExpired):
                logger.warning("Could not determine audio duration; skipping duration guard.")

        # --- Transcribe ----------------------------------------------------
        if whisper_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")

        segments_gen, info = whisper_model.transcribe(
            str(wav_path),
            language=whisper_language,
            task="transcribe",                          # transcribe (not translate)
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            word_timestamps=WORD_TIMESTAMPS,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            initial_prompt=INITIAL_PROMPT if INITIAL_PROMPT else None,
            temperature=TEMPERATURE,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            log_prob_threshold=LOG_PROB_THRESHOLD,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        )
        logger.info(
            "Detected language: %s (prob=%.2f), duration=%.1fs, beam=%d, vad=%s",
            info.language, info.language_probability, info.duration, BEAM_SIZE, VAD_FILTER,
        )

        segments = []
        full_text_parts = []
        for seg in segments_gen:
            seg_dict = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            if WORD_TIMESTAMPS and getattr(seg, "words", None):
                seg_dict["words"] = [
                    {
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "word": w.word,
                        "probability": round(w.probability, 2),
                    }
                    for w in seg.words
                ]
            segments.append(seg_dict)
            full_text_parts.append(seg.text.strip())

        response_data = {
            "text": " ".join(full_text_parts),
            "language": info.language,
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 4) if info.language_probability is not None else 1.0,
            "duration": round(info.duration, 2),
            "segments": segments,
        }
        return JSONResponse(
            content=response_data,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Cleanup temp files
        for p in (input_path, wav_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            Path(tmp_dir).rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Serve the frontend (index.html + static assets)
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
