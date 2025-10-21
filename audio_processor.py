from pymemcache.client.base import Client
from speechbrain.inference import EncoderClassifier

import soundfile as sf
import torchaudio
import config
import base64
import torch
import time
import io


def _connect_cache() -> Client | None:
    cache = Client(config.MEMCACHED_SERVER)

    try:
        cache.set("audio_processor", "connected", expire=0)

        print("Connected to Memcached.")
        return cache
    
    except Exception as e:
        print(f"Error connecting to Memcached: {e}")
        return None
    

def _load_model() -> EncoderClassifier | None:
    try:
        model = EncoderClassifier.from_hparams(source=config.MODEL_NAME, run_opts={"device": config.DEVICE})

        print("Model loaded successfully.")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def _process_audio(model: EncoderClassifier, audio_data: str, language_iso: str) -> float:
    try:
        decoded_audio = base64.b64decode(audio_data)
        buffer = io.BytesIO(decoded_audio)

        signal, fs = sf.read(buffer, dtype="float32") # ASSUMES WAV FORMAT
        signal = torch.from_numpy(signal).unsqueeze(0)

    except Exception as e:
        print(f"Error reading audio data: {e}")
        return 0.0
    
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)

    model.eval()
    with torch.no_grad():
        out_prob, _, __, ___ = model.classify_batch(signal)

    probabilities = out_prob[0]
    lang_index = model.hparams.label_encoder.get_ind_from_label(language_iso)
    target_lang_probs = probabilities[:, lang_index]

    confident_frames = torch.sum(target_lang_probs > config.CONFIDENCE_THRESHOLD).item()
    frame_duration_sec = 0.02

    return confident_frames * frame_duration_sec


def main_loop() -> None:
    """
    A = AUDIO
    AT = AUDIO_TASK
    P = PROGRESS
    """

    cache = _connect_cache()
    if not cache: return

    model = _load_model()
    if not model: return

    print("Audio processing started, polling for tasks...")

    task_id = 0 # TODO: WHAT IF SERVICES OUT OF SYNC
    while True:
        task: dict | None = cache.get(f"AT:{task_id}")
        if not task:
            time.sleep(0.5)
            continue

        audio: str | None = cache.get(f"A:{task_id}")
        if not audio:
            time.sleep(0.5)
            continue
        
        duration = _process_audio(
            model=model,
            audio_data=audio,
            language_iso=task.get("language")
        )

        user_id, goal_id = f"P:{task.get("user_id")}", f"P:{task.get("goal_id")}"

        progress: dict = cache.get(f"P:{user_id}", {})
        progress[goal_id] = progress.get(goal_id, 0) + duration
        cache.set(f"P:{user_id}", progress, expire=0)

        cache.delete(f"AT:{task_id}")
        cache.delete(f"A:{task_id}")

        task_id += 1


if __name__ == "__main__":
    main_loop()