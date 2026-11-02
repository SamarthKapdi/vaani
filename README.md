# 🎙️ Vaani (वाणी) — AI Speech-to-Text

Ultra-fast, high-accuracy speech-to-text transcription with interactive audio/video playback. Powered by OpenAI's **Whisper-Large-v3** running on **Groq LPUs** for sub-second to few-second inference.

Deployable for **free on Vercel**, **Render**, **Koyeb**, or locally via **Docker**.

---

## ✨ Features

- **⚡ Blazing Fast**: Sub-second transcription powered by Groq LPU inference.
- **🎯 State-of-the-Art Accuracy**: Uses `whisper-large-v3` (~15.7% Hindi WER vs. 42%+ on small models).
- **🌐 90+ Languages Supported**: Automatic detection or explicit language lock (English, Hindi, etc.).
- **▶️ Interactive Media Player**: Built-in player with speed controls (0.75x to 2x).
- **⏱️ Clickable Timestamps**: Click any segment timestamp to seek and play media instantly.
- **📁 Drag & Drop**: Accepts MP3, WAV, M4A, MP4, WebM, OGG, FLAC (up to 25 MB).
- **🚀 1-Click Serverless Deploy**: Deploys free on Vercel with zero credit card required.

---

## 🚀 1-Click Deployment on Vercel (Free)

1. Get a free API key at **[console.groq.com/keys](https://console.groq.com/keys)**.
2. Go to **[vercel.com/new](https://vercel.com/new)**.
3. Import your GitHub repository (`SamarthKapdi/vaani`).
4. Under **Environment Variables**, add:
   - **`GROQ_API_KEY`**: *Your Groq API key (`gsk_...`)*
5. Click **Deploy**. Your app will be live on a free `*.vercel.app` domain in seconds!

---

## 💻 Local Quick Start

### Option 1: Python Directly

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Groq API key
export GROQ_API_KEY="gsk_..."      # Linux / macOS
$env:GROQ_API_KEY="gsk_..."       # Windows PowerShell

# 3. Run the development server
uvicorn app.main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser.

### Option 2: Docker Compose

```bash
GROQ_API_KEY="gsk_..." docker compose up --build -d
```
Open **`http://localhost:8000`** in your browser.

---

## 📡 API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health & API key probe (`{"status": "ok", "engine": "groq"}`) |
| `POST` | `/transcribe` | Upload media (`multipart/form-data`), returns JSON transcript |

### Request Parameters (`POST /transcribe`)
- `file`: Audio or video file (max 25 MB)
- `language`: `auto` (default), `en`, `hi`, or any ISO language code

### Response Format
```json
{
  "text": "नमस्ते, भारत एक सुंदर देश है और तकनीक बहुत तेजी से आगे बढ़ रही है।",
  "language": "hi",
  "detected_language": "hi",
  "language_probability": 1.0,
  "duration": 5.66,
  "segments": [
    {
      "start": 0.0,
      "end": 5.66,
      "text": "नमस्ते, भारत एक सुंदर देश है और तकनीक बहुत तेजी से आगे बढ़ रही है।"
    }
  ],
  "model": "whisper-large-v3",
  "engine": "groq"
}
```

---

## 🛠️ Architecture Comparison

| Metric | Local Faster-Whisper | Groq Whisper-Large-v3 (Current) |
| :--- | :--- | :--- |
| **Model** | `small` | `whisper-large-v3` (OpenAI flagship) |
| **Hindi Accuracy** | ~42% WER (phonetic/approximate) | **~15.7% WER** (accurate Devanagari) |
| **Inference Time** | 20–40s per clip | **0.5s – 2s** (instantaneous) |
| **Startup / Cold Start** | 23.5s container startup | **< 1s** serverless cold start |
| **RAM Required** | 16 GB RAM | **< 50 MB** |
| **Max File Size** | 500 MB (slow on CPU) | **25 MB** (approx. 25–30 mins audio) |
| **Hosting** | Requires paid VM or HF Pro | **Free on Vercel** / Render / Koyeb |
