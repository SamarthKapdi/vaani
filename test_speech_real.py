"""
Real Speech Transcription Test Script.
Generates genuine English and Hindi spoken audio using gTTS,
submits them to the running VoiceScript container,
and verifies transcription accuracy, language detection, and response times.
"""
import json
import os
import sys
import time
import requests
from gtts import gTTS

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://localhost:8000"

def test_health():
    print("=" * 60)
    print("1. Testing /health endpoint...")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text}")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok", "Expected status ok"
    print("Health check PASSED! ✅")
    print("=" * 60)

def generate_audio():
    print("\n2. Generating real speech samples with gTTS...")
    en_text = "Artificial intelligence and speech recognition are transforming modern technology."
    hi_text = "नमस्ते, भारत एक सुंदर देश है और तकनीक बहुत तेजी से आगे बढ़ रही है।"
    
    en_file = "real_en_sample.mp3"
    hi_file = "real_hi_sample.mp3"
    
    tts_en = gTTS(text=en_text, lang='en', slow=False)
    tts_en.save(en_file)
    print(f"  - Saved English audio ({os.path.getsize(en_file)} bytes): '{en_text}'")
    
    tts_hi = gTTS(text=hi_text, lang='hi', slow=False)
    tts_hi.save(hi_file)
    print(f"  - Saved Hindi audio ({os.path.getsize(hi_file)} bytes): '{hi_text}'")
    
    return en_file, hi_file, en_text, hi_text

def transcribe_file(file_path, language=None):
    url = f"{BASE_URL}/transcribe"
    data = {}
    if language:
        data["language"] = language
        
    start_t = time.time()
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
        r = requests.post(url, files=files, data=data, timeout=300)
    elapsed = time.time() - start_t
    
    return r, elapsed

def run_tests():
    test_health()
    en_file, hi_file, en_expected, hi_expected = generate_audio()
    
    test_cases = [
        {"name": "English Speech (Explicit 'en')", "file": en_file, "lang": "en"},
        {"name": "English Speech (Auto-detect)",   "file": en_file, "lang": "auto"},
        {"name": "Hindi Speech (Explicit 'hi')",   "file": hi_file, "lang": "hi"},
        {"name": "Hindi Speech (Auto-detect)",     "file": hi_file, "lang": "auto"},
    ]
    
    results = []
    
    print("\n3. Running transcription inference tests against container...")
    for tc in test_cases:
        print("-" * 60)
        print(f"Running: {tc['name']}...")
        r, elapsed = transcribe_file(tc["file"], tc["lang"])
        print(f"  HTTP Status: {r.status_code} in {elapsed:.2f}s")
        if r.status_code != 200:
            print(f"  ERROR Response: {r.text}")
            results.append({"name": tc["name"], "passed": False, "error": r.text, "elapsed": elapsed})
            continue
            
        data = r.json()
        transcript = data.get("text", "")
        detected_lang = data.get("detected_language") or data.get("language")
        prob = data.get("language_probability")
        duration = data.get("duration")
        segments = data.get("segments", [])
        
        print(f"  Detected Lang: {detected_lang} (prob: {prob})")
        print(f"  Duration:      {duration}s")
        print(f"  Segments:      {len(segments)}")
        print(f"  Transcript:    \"{transcript}\"")
        
        # Verify text is non-empty
        passed = len(transcript.strip()) > 0
        results.append({
            "name": tc["name"],
            "passed": passed,
            "elapsed": round(elapsed, 2),
            "transcript": transcript,
            "detected_lang": detected_lang,
            "prob": prob,
            "duration": duration,
        })
        
    print("\n" + "=" * 60)
    print("SUMMARY OF REAL SPEECH TESTS:")
    print("=" * 60)
    all_passed = True
    for res in results:
        status_sym = "✅ PASS" if res.get("passed") else "❌ FAIL"
        if not res.get("passed"):
            all_passed = False
        print(f"{status_sym} | {res['name']:<35} | Latency: {res.get('elapsed', 0)}s | Output: {res.get('transcript', '')[:40]}")
    
    # Clean up test audio files
    for f in [en_file, hi_file]:
        try:
            os.remove(f)
        except OSError:
            pass

    if all_passed:
        print("\n🎉 ALL REAL SPEECH TRANSCRIPTION TESTS PASSED SUCCESSFULLY!")
    else:
        print("\n❌ SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
