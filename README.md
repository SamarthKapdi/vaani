---
title: Vaani
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🎙️ Vaani — AI Speech-to-Text

AI-powered local speech-to-text transcription with interactive audio/video playback. Upload media files and get timestamped transcripts — all running locally with [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper).

## Quick Start (Local)

```bash
docker compose up --build -d
```

Open **http://localhost:8000** in your browser.

## Features

- **Drag & drop upload** for MP3, WAV, MP4, MKV, and more (up to 500 MB)
- **Built-in Media Player** with speed controls (0.75x to 2x) & instant playback
- **Interactive Timestamps** (click any segment timestamp to seek and play media)
- **Language Detection & Selection** (Auto, English, Hindi, and 90+ languages)
- **Audio Preprocessing** with FFmpeg loudness normalization (`loudnorm`)
- **Tunable Accuracy Levers** (Beam search, VAD filter, word timestamps, initial prompt)
- **Duration Guard** — configurable max audio length (default 180s, set `MAX_DURATION_SECONDS=0` to disable)
- **100% Local & Private** — no external APIs, no data leaves your machine

## ⚠️ Model Capability Matrix (Hindi & Multilingual Support)

The default model size is `small`. **`base` is NOT recommended for Hindi.**

| Model Size | RAM Footprint | English Accuracy | Hindi Accuracy & Recommendations |
| :--- | :---: | :---: | :--- |
| **`base`** | ~250 MB | Good (0–15% WER) | **Poor (100% WER)** — Do NOT use for Hindi (transcribes into Urdu or transliterated English). |
| **`small`** (default) | ~800 MB | Very Good (0–15% WER) | **Usable (~13–18% WER)** — Devanagari output; phonetic accents/matras approximate. |
| **`medium`** | ~2.5 GB | Excellent (0–9% WER) | **Good (~13% WER)** — Recommended for CPU servers with 8GB+ RAM. |
| **`large-v3-turbo`** | ~6 GB | State-of-the-art | **Very Good** — Best speed/accuracy tradeoff on modern servers or GPUs. |
| **`large-v3`** | ~10 GB | State-of-the-art | **Best** — Highest accuracy across all dialects (requires 12GB+ RAM or GPU). |

> To change model size, set `MODEL_SIZE=medium` in `docker-compose.yml` or as a build arg.

## Accuracy Levers

Tuned in `docker-compose.yml` or container environment without rebuilding:

| Lever | Env Var | Default | Description |
| :--- | :--- | :---: | :--- |
| **Beam Search** | `BEAM_SIZE` | `5` | `1` is greedy/fastest; `5`–`10` improves accuracy on difficult audio. |
| **VAD Pre-Filter** | `VAD_FILTER` | `true` | Silero VAD skips silence and prevents hallucinations on gaps. |
| **Word Timestamps** | `WORD_TIMESTAMPS` | `false` | Enables per-word start/end timing (useful for subtitles). |
| **Context Continuity** | `CONDITION_ON_PREVIOUS_TEXT` | `true` | Conditions each segment on prior output for better phrasing. |
| **Domain Vocabulary** | `INITIAL_PROMPT` | `""` | Prime model with custom brand names, acronyms, or proper nouns. |
| **Compute Type** | `COMPUTE_TYPE` | `int8` | `int8` for fast CPU inference; `float16` for GPU. |
| **Duration Limit** | `MAX_DURATION_SECONDS` | `180` | Max audio length in seconds (~3 mins). Set `0` to disable. |

## Audio Preprocessing

Before transcription, FFmpeg converts all uploaded audio/video into 16 kHz mono PCM WAV while applying ITU-R BS.1770 loudness normalization:
```bash
-af loudnorm=I=-16:TP=-1.5:LRA=11
```
This boosts low-volume voices and balances uneven phone/mic recordings for higher recognition accuracy.

## API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check (`{"status": "ok", "model": "small"}`) |
| `POST` | `/transcribe` | Upload media (`multipart/form-data`), returns JSON transcript |

### Request Parameters (`POST /transcribe`)
- `file`: Media file (MP3, WAV, MP4, MKV, etc., max 500 MB)
- `language`: `auto` (default), `en`, `hi`, or any BCP-47 language code

### Response Format
```json
{
  "text": "नमस्ते, मेरा नाम राहुल है...",
  "language": "hi",
  "detected_language": "hi",
  "language_probability": 0.985,
  "duration": 4.99,
  "segments": [
    { "start": 0.0, "end": 4.0, "text": "नमस्ते, मेरा नाम राहुल है..." }
  ]
}
```

## 🚀 Deployment

### Hugging Face Spaces (Free, Recommended)

HF Spaces provides **16 GB RAM / 2 vCPU** on the free tier — enough for the `small` or `medium` model.

1. **Create a GitHub repo** and push this project
2. **Go to** [huggingface.co/new-space](https://huggingface.co/new-space)
3. **Select**: SDK → **Docker**, Hardware → **CPU basic (Free)**
4. **Link** your GitHub repo — HF auto-deploys on every push
5. (Optional) In your Space **Settings → Variables and secrets**, you can explicitly configure:
   - `MODEL_SIZE` = `small` (or `medium`)
   - `MAX_DURATION_SECONDS` = `180`
   - `BEAM_SIZE` = `5`
6. The app will be live at `https://huggingface.co/spaces/<your-username>/<space-name>`

> **Runtime Verified**:
> - Container startup takes **~23.5s on 2 vCPUs** (safely within HF's strict 60s startup timeout) because the Faster-Whisper model is pre-downloaded and baked into the image at build time.
> - Real speech inference has been validated on both English and Hindi audio with accurate Devanagari script output and automatic language detection.
> - The `README.md` frontmatter (`sdk: docker`, `app_port: 7860`) tells HF everything it needs. No other config required.

### Local Docker

```bash
# Default (small model, port 8000)
docker compose up --build -d

# With medium model for better Hindi
MODEL_SIZE=medium docker compose up --build -d
```

## GPU Support

Uncomment the `deploy` block in `docker-compose.yml` and set:
```yaml
environment:
  - DEVICE=cuda
  - COMPUTE_TYPE=float16
```

