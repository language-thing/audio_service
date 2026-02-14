from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from redis import Redis

import numpy as np
import orjson
import torch
import math
import time


CONFIDENCE_THRESHOLD = 0.45
WINDOW_SIZE = 50
MULTIPLIER = 2


class AccentRobustDetector:
    def __init__(self, model_size="126"):
        model_id = f"facebook/mms-lid-{model_size}"
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps = utils[0]

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("[DETECTOR] Using GPU")
        else:
            print("[DETECTOR] Using CPU")

    def remove_silence(self, audio_np: np.ndarray) -> np.ndarray:
        """Uses Silero VAD to strip out all non-speech segments."""
        audio_tensor = torch.from_numpy(audio_np)
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=16000
        )

        if not speech_timestamps:
            return np.array([], dtype=np.float32)

        speech_segments = [audio_np[ts['start']:ts['end']] for ts in speech_timestamps]
        return np.concatenate(speech_segments)

    def process_audio(self, audio_data: bytes, language_iso3: str) -> float:
        """Returns True if language detected, False otherwise"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.float32).copy()

            if len(audio_np) == 0:
                return 0.0

            audio_np = self.remove_silence(audio_np)
            if len(audio_np) < 8000:
                print("Skipping: Not enough speech detected.")
                return 0.0

            # Process with MMS-LID
            inputs = self.processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]

            top_probs, top_indices = torch.topk(probs, k=5)
            top_langs = [self.model.config.id2label[i.item()] for i in top_indices]
    
            print(f"Top Predictions: {list(zip(top_langs, top_probs.tolist()))}")
        
            if language_iso3 in top_langs:
                target_idx = top_langs.index(language_iso3)
                conf = top_probs[target_idx].item()
                
                print(f"Detected: {language_iso3}) with confidence {conf}")

                if conf >= CONFIDENCE_THRESHOLD:
                    return (len(audio_np) / 16000) * MULTIPLIER
            
            return 0.0

        except Exception as e:
            print(f"Error processing audio: {e}")
            return 0.0


# Main worker loop
print("[WORKER] Initializing detector...")
detector = AccentRobustDetector(model_size="126")

cache = Redis(decode_responses=True)
audio_cache = Redis(decode_responses=False)

print("[WORKER] Started processing queue")

while True:
    cache.set("worker:status", "idle")
    
    result = cache.brpop("tasks", timeout=0) # type: ignore
    if not result:
        continue
    
    cache.set("worker:status", "working")
    
    _, task_raw = result # type: ignore
    task: dict = orjson.loads(task_raw)
    
    data: bytes = audio_cache.get(task["k"]) # type: ignore
    audio_cache.delete(task["k"])
    
    start_time = time.perf_counter()
    duration = detector.process_audio(data, task["l"])
    process_time = time.perf_counter() - start_time
    
    # Stats tracking
    cache.lpush("stats:processing_times", process_time)
    cache.ltrim("stats:processing_times", 0, WINDOW_SIZE - 1)
    cache.incr("stats:total_tasks")
    
    if duration > 0:
        cache.incr("stats:successful_tasks")
        cache.incrby(f"p:{task['u']}", math.ceil(duration))
        print(f"With length: {math.ceil(duration)} sec")

    else:
        cache.incr("stats:skipped_tasks")
        cache.set(f"result:{task['u']}", "false", ex=3600)
