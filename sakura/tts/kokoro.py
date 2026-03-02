import urllib.request

import numpy as np
from kokoro_onnx import Kokoro

from sakura.config import MODELS_DIR

_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
)
_MODEL_FILE = "kokoro-v1.0.int8.onnx"
_VOICES_FILE = "voices-v1.0.bin"


class KokoroTTS:
    """Text-to-Speech using the Kokoro ONNX model (local inference)."""

    def __init__(self):
        model_path = MODELS_DIR / _MODEL_FILE
        voices_path = MODELS_DIR / _VOICES_FILE
        self._download_if_needed(model_path, voices_path)
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self.sample_rate = 24000

    def _download_if_needed(self, model_path, voices_path):
        if not model_path.exists():
            print("Downloading Kokoro ONNX model...")
            urllib.request.urlretrieve(_BASE_URL + _MODEL_FILE, str(model_path))
        if not voices_path.exists():
            print("Downloading Kokoro voices...")
            urllib.request.urlretrieve(_BASE_URL + _VOICES_FILE, str(voices_path))

    def generate(self, text: str, voice: str = "af_heart",
                 speed: float = 1.0) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Returns:
            Tuple of (audio_int16 numpy array, sample_rate).
        """
        wav, sr = self._kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        return (wav * 32767).astype(np.int16), sr
