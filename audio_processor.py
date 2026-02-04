from redis import Redis

import soundfile as sf

import whisper
import base64
import orjson
import math
import time
import io


CONFIDENCE_THRESHOLD = 0.6
WINDOW_SIZE = 50
MULTIPLIER = 3

# or "medium" if on VPS
model = whisper.load_model("small")
cache = Redis(decode_responses=True)


def _process_audio(model, audio_data: str, language_iso: str) -> float:
    try:
        # Decode base64 → buffer → waveform
        decoded_audio = base64.b64decode(audio_data)
        buffer = io.BytesIO(decoded_audio)

        # Load and resample if needed
        signal, fs = sf.read(buffer, dtype="float32")

        # Convert to log-mel spectrogram
        audio = whisper.pad_or_trim(signal)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect language
        _, probs = model.detect_language(mel)
        top_lang, top_prob = max(probs.items(), key=lambda x: x[1])

        # Logging / debug
        print(f"Top detected language: {top_lang} ({top_prob:.3f})")
        print(f"Expected: {language_iso} ({probs.get(language_iso, 0.0):.3f})")

        # Check expected language probability
        target_prob = probs.get(language_iso, 0.0)

        if target_prob >= CONFIDENCE_THRESHOLD:
            print("Confident detection")
            duration_sec = len(signal) / 16000
            return float(duration_sec) * MULTIPLIER
        
        else:
            print("Not confident")
            return 0.0

    except Exception as e:
        print(f"Error reading or processing audio: {e}")
        return 0.0


def _TEST():
    with open("result_speech_only.wav", "rb") as file:
        file_data = file.read()
        b64_data = base64.b64encode(file_data).decode("utf-8")  # base64 as string
        print(_process_audio(model, b64_data, "nl"))

_TEST()


while True:
    cache.set("worker:status", "idle")

    result = cache.brpop("tasks", timeout=0) # type: ignore
    if not result: continue

    cache.set("worker:status", "working")

    _, task_data = result # type: ignore
    task: dict = orjson.loads(task_data)

    start_time = time.perf_counter()
    duration = _process_audio(model, task["d"], task["l"])
    process_time = time.perf_counter() - start_time

    cache.lpush("stats:processing_times", process_time)
    cache.ltrim("stats:processing_times", 0, WINDOW_SIZE - 1)

    cache.incr("stats:total_tasks")
    if duration > 0:
        cache.incr("stats:successful_tasks")
        cache.incrby(f"p:{task['u']}", math.ceil(duration))
        
    else:
        cache.incr("stats:skipped_tasks")
