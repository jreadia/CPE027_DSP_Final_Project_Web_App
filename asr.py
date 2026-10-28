import os
import re
import threading
import numpy as np
import torch
from jiwer import wer

# Available models
LARGE_MODEL_ID = "sapinsapin/whisper-large-v3-pld-ceb-norm"
LARGE_PROCESSOR_ID = "openai/whisper-large-v3"

SMALL_MODEL_ID = "sapinsapin/whisper-small-pld-ceb"
SMALL_PROCESSOR_ID = "openai/whisper-small"

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Automatically pick Large for GPU and Small for CPU laptops
DEFAULT_MODEL_ID = LARGE_MODEL_ID if _device == "cuda" else SMALL_MODEL_ID
DEFAULT_PROCESSOR_ID = LARGE_PROCESSOR_ID if _device == "cuda" else SMALL_PROCESSOR_ID

_model = None
_processor = None
_current_model_id = None
_loading_lock = threading.Lock()

def normalize_text(text: str) -> str:
    """Matches text normalization used in documentation.ipynb."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def is_model_loaded() -> bool:
    return _model is not None and _processor is not None

def get_device_info() -> dict:
    has_gpu = torch.cuda.is_available()
    return {
        "device": _device,
        "device_name": torch.cuda.get_device_name(0) if has_gpu else "Standard CPU",
        "has_gpu": has_gpu,
        "active_model": _current_model_id or DEFAULT_MODEL_ID,
        "is_loaded": is_model_loaded()
    }

def load_asr_model(model_name: str = "auto"):
    """
    Loads Whisper fine-tuned for Cebuano.
    - If CUDA GPU is available: uses Whisper Large v3 (FP16).
    - If CPU only (laptop): automatically uses Whisper Small (FP32) to prevent freezing and save memory.
    """
    global _model, _processor, _current_model_id

    # Resolve target model based on hardware and preference
    if model_name == "large" or (model_name == "auto" and _device == "cuda"):
        target_model = LARGE_MODEL_ID
        target_proc = LARGE_PROCESSOR_ID
    else:
        target_model = SMALL_MODEL_ID
        target_proc = SMALL_PROCESSOR_ID

    if _model is not None and _current_model_id == target_model:
        return _model, _processor

    with _loading_lock:
        if _model is None or _current_model_id != target_model:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            print(f"[ASR] Device: {_device.upper()} ({get_device_info()['device_name']})")
            print(f"[ASR] Loading processor from '{target_proc}'...")
            _processor = WhisperProcessor.from_pretrained(target_proc)

            print(f"[ASR] Loading Cebuano Whisper model '{target_model}'...")
            dtype = torch.float16 if _device == "cuda" else torch.float32
            
            # Limit CPU threads to prevent maxing out laptop cores
            if _device == "cpu":
                torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

            _model = WhisperForConditionalGeneration.from_pretrained(
                target_model,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(_device)
            _model.eval()
            _current_model_id = target_model
            print(f"[ASR] Model '{target_model}' loaded successfully on {_device.upper()}!")

    return _model, _processor

def transcribe_audio_chunked(audio: np.ndarray, sr: int = 16000, model_choice: str = "auto") -> str:
    """
    Transcribes audio using Whisper with chunking.
    Safely runs on either CUDA GPU or CPU.
    """
    model, processor = load_asr_model(model_name=model_choice)

    chunk_samples = 25 * sr
    transcripts = []

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < sr * 0.5:
            continue

        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        dtype = torch.float16 if _device == "cuda" else torch.float32
        input_features = inputs.input_features.to(_device, dtype=dtype)

        with torch.inference_mode():
            predicted_ids = model.generate(
                input_features,
                task="transcribe",
                max_new_tokens=225
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        if text.strip():
            transcripts.append(text.strip())

    return " ".join(transcripts).strip()

def calculate_wer(reference_text: str, hypothesis_text: str) -> float:
    """Calculates WER % using jiwer between normalized texts."""
    ref_norm = normalize_text(reference_text)
    hyp_norm = normalize_text(hypothesis_text)
    if not ref_norm:
        return 0.0
    return round(float(wer(ref_norm, hyp_norm) * 100), 2)
