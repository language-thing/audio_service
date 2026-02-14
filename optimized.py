from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from redis import Redis

import numpy as np
import ffmpeg
import orjson
import torch
import math
import time


CONFIDENCE_THRESHOLD = 0.5
WINDOW_SIZE = 50
MULTIPLIER = 2


class AccentRobustDetector:
    def __init__(self, model_size="126", use_half_precision=False):
        model_id = f"facebook/mms-lid-{model_size}"
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        
        # Optimization 1: Enable eval mode BEFORE moving to GPU
        self.model.eval()
        
        # Optimization 2: Move to GPU and optionally use half precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if use_half_precision and torch.cuda.is_available():
            self.model = self.model.half()
            self.use_fp16 = True
            print("[DETECTOR] Using GPU with FP16 (half precision)")
        else:
            self.use_fp16 = False
            if torch.cuda.is_available():
                print("[DETECTOR] Using GPU with FP32")
            else:
                print("[DETECTOR] Using CPU")
        
        # Optimization 3: Set inference-only optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Optimization 4: Compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[DETECTOR] Model compiled with torch.compile")
            except Exception as e:
                print(f"[DETECTOR] Could not compile model: {e}")
        
        # Load VAD model
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        self.get_speech_timestamps = utils[0]
        
        # Optimization 5: Move VAD model to same device
        self.vad_model = self.vad_model.to(self.device)

    def remove_silence(self, audio_np: np.ndarray) -> np.ndarray:
        """Uses Silero VAD to strip out all non-speech segments."""
        # Optimization 6: Create tensor directly on target device
        audio_tensor = torch.from_numpy(audio_np).to(self.device)
        
        # Optimization 7: Disable gradient computation for VAD
        with torch.no_grad():
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model, 
                sampling_rate=16000,
                threshold=0.5,  # Explicitly set threshold
                return_seconds=False
            )
        
        if not speech_timestamps:
            return np.array([], dtype=np.float32)
        
        # Optimization 8: Move back to CPU for numpy operations
        audio_np_cpu = audio_tensor.cpu().numpy() if audio_tensor.is_cuda else audio_np
        speech_segments = [audio_np_cpu[ts['start']:ts['end']] for ts in speech_timestamps]
        return np.concatenate(speech_segments) if speech_segments else np.array([], dtype=np.float32)
    
    def process_audio(self, audio_data: bytes, language_iso3: str) -> float:
        """Returns duration if language detected, 0.0 otherwise"""
        try:
            # Optimization 9: Use numpy's buffer protocol efficiently
            audio_np = np.frombuffer(audio_data, dtype=np.float32)

            if len(audio_np) == 0:
                return 0.0
            
            audio_np = self.remove_silence(audio_np)
            
            # Optimization 10: Adjusted minimum speech threshold
            # For 5-second clips, 0.5s of speech might be enough for accent detection
            min_samples = 8000 if len(audio_data) > 80000 else 4000  # Adaptive threshold
            
            if len(audio_np) < min_samples: 
                print("Skipping: Not enough speech detected.")
                return 0.0
            
            # Optimization 11: Process with MMS-LID - batch size 1, optimized padding
            inputs = self.processor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            
            # Optimization 12: Move inputs to device (handles both CPU and GPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Optimization 13: Use half precision if enabled
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Optimization 14: Use inference mode (better than no_grad for inference)
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                # Optimization 15: Use in-place softmax on GPU
                probs = torch.softmax(logits, dim=-1)[0]
                
                # Optimization 16: Only compute top-5, keep on GPU until needed
                top_probs, top_indices = torch.topk(probs, k=5)

            # Optimization 17: Move to CPU only once, after all GPU operations
            top_probs = top_probs.cpu()
            top_indices = top_indices.cpu()
            
            top_langs = [self.model.config.id2label[i.item()] for i in top_indices]
    
            print(f"Top Predictions: {list(zip(top_langs, top_probs.tolist()))}")
        
            if language_iso3 in top_langs:
                target_idx = top_langs.index(language_iso3)
                conf = top_probs[target_idx].item()
                
                print(f"Detected: {language_iso3} with confidence {conf}")

                if conf >= CONFIDENCE_THRESHOLD:
                    return (len(audio_np) / 16000) * MULTIPLIER
            
            return 0.0
        
        except ffmpeg.Error as e:
            print(f"FFmpeg Stderr: {e.stderr.decode()}")
            return 0.0
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


# Main worker loop
print("[WORKER] Initializing detector...")

# Optimization 18: Enable half precision for GPU (2x faster inference)
use_fp16 = torch.cuda.is_available()
detector = AccentRobustDetector(model_size="126", use_half_precision=use_fp16)

# Optimization 19: Set number of threads for CPU inference
if not torch.cuda.is_available():
    # Use all available cores for CPU inference
    torch.set_num_threads(torch.get_num_threads())
    print(f"[WORKER] Using {torch.get_num_threads()} CPU threads")

cache = Redis(decode_responses=True)
audio_cache = Redis(decode_responses=False)

print("[WORKER] Started processing queue")

while True:
    cache.set("worker:status", "idle")
    
    result = cache.brpop("tasks", timeout=0)
    if not result:
        continue
    
    cache.set("worker:status", "working")
    
    _, task_raw = result
    task: dict = orjson.loads(task_raw)
    
    data: bytes = audio_cache.get(task["k"])
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
