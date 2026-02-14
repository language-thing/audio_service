from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from redis import Redis

import numpy as np
import ffmpeg
import orjson
import torch
import math
import time


CONFIDENCE_THRESHOLD = 0.75
WINDOW_SIZE = 50
MULTIPLIER = 3


class AccentRobustDetector:
    def __init__(self, model_size="126"):
        model_id = f"facebook/mms-lid-{model_size}"
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("[DETECTOR] Using GPU")
        else:
            print("[DETECTOR] Using CPU")
    
    def process_audio(self, audio_data: bytes, language_iso3: str) -> float:
        """Returns True if language detected, False otherwise"""
        try:
            # FFmpeg conversion
            out, _ = (
                ffmpeg.input("pipe:0")
                .output("pipe:1", format="f32le", acodec="pcm_f32le", ac=1, ar=16000)
                .run(input=audio_data, capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            audio_np = np.frombuffer(out, np.float32)
            
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

            top_probs, top_indices = torch.topk(probs, k=3)
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
