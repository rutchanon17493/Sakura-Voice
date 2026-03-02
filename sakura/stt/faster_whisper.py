import numpy as np
from faster_whisper import WhisperModel


class FasterWhisperSTT:
    """Speech-to-Text using Faster-Whisper (local, CPU/GPU)."""

    def __init__(self, model_size: str = "tiny.en", device: str = "cpu",
                 compute_type: str = "int8"):
        print(f"Loading Whisper model ({model_size})...")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper ready.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe a float32 numpy audio array. Returns text."""
        segments, _ = self._model.transcribe(audio_data, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()
