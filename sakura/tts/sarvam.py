import base64
import io
import wave

import numpy as np
import requests


class SarvamTTS:
    """Text-to-Speech using Sarvam AI's Bulbul model."""

    _BASE_URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self, api_key: str, model: str = "bulbul:v3",
                 speaker: str = "Shubh", target_language_code: str = "hi-IN",
                 pace: float = 1.0):
        self.api_key = api_key
        self.model = model
        self.speaker = speaker
        self.target_language_code = target_language_code
        self.pace = pace
        self.sample_rate = 22050
        print(f"Sarvam TTS ready (model={model}, speaker={speaker}, lang={target_language_code})")

    def generate(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Returns:
            Tuple of (audio_int16 numpy array, sample_rate).
        """
        response = requests.post(
            self._BASE_URL,
            headers={
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "inputs": [text],
                "target_language_code": self.target_language_code,
                "model": self.model,
                "speaker": self.speaker,
                "pace": self.pace,
                "output_audio_codec": "wav",
            },
            timeout=30,
        )
        if not response.ok:
            print(f"TTS error {response.status_code}: {response.text}")
            response.raise_for_status()

        audios = response.json().get("audios", [])
        if not audios:
            return np.array([], dtype=np.int16), self.sample_rate

        wav_bytes = base64.b64decode(audios[0])
        return self._wav_to_numpy(wav_bytes)

    @staticmethod
    def _wav_to_numpy(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16), sr
