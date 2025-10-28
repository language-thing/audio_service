import io
import base64
import torch
import torchaudio
import whisper
import soundfile as sf
import numpy as np

CONFIDENCE_THRESHOLD = 0.6

model = whisper.load_model("small")  # or "medium" if on VPS

def _process_audio(model, audio_data: str, language_iso: str) -> float:
    """
    Detects the spoken language of a WAV clip using Whisper.
    Returns duration (s) if detection probability for language_iso exceeds threshold, else 0.0

    🧠 If you ever do want to use raw PCM

    If you intentionally want to skip the header and handle raw samples (for custom pipelines), you must specify the format manually:

    sf.read(buffer, dtype="float32", format="RAW", samplerate=16000, channels=1, subtype="PCM_16")


    But that’s only needed for low-level processing — not for normal WAV audio.
    """
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
            print("Confident detection ✅")
            duration_sec = len(signal) / 16000
            return float(duration_sec)
        else:
            print("Not confident ❌")
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

