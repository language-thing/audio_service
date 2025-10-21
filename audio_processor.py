from pymemcache.client.base import Client
from speechbrain.inference import EncoderClassifier

import config
import time


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

    print("Audio processing started, Polling for tasks...")

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
        
        duration = process_audio(
            model=model,
            audio_data=audio,
            sample_rate=task.get("sample_rate"),
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