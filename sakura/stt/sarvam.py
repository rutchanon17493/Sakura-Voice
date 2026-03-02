import io
import wave

import numpy as np
import requests


class SarvamSTT:
    """Speech-to-Text using Sarvam AI's Saarika / Saaras model."""

    _BASE_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: str, model: str = "saarika:v2.5",
                 language_code: str = "hi-IN"):
        self.api_key = api_key
        self.model = model
        self.language_code = language_code
        self.detected_language: str = language_code  # updated after each transcription
        print(f"Sarvam STT ready (model={model}, lang={language_code})")

    def transcribe(self, audio_np: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio.

        Args:
            audio_np: float32 numpy array in range [-1, 1].
            sample_rate: sample rate of the audio.

        Returns:
            Transcribed text string.
        """
        wav_bytes = self._to_wav_bytes(audio_np, sample_rate)
        response = requests.post(
            self._BASE_URL,
            headers={"api-subscription-key": self.api_key},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": self.model, "language_code": self.language_code},
            timeout=30,
        )
        if not response.ok:
            print(f"STT error {response.status_code}: {response.text}")
            response.raise_for_status()
        result = response.json()
        transcript = result.get("transcript", "").strip()
        detected = result.get("language_code", "").strip()
        if detected and detected != "unknown":
            self.detected_language = detected
        if not transcript:
            print(f"STT returned empty transcript. Full response: {result}")
        return transcript

    @staticmethod
    def _to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
        audio_int16 = (audio_np * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()
