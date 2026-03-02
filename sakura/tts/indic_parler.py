"""Indic Parler-TTS — HuggingFace, Apache-2.0, CPU-only."""

import numpy as np


class IndicParlerTTS:
    """Text-to-Speech using ai4bharat/indic-parler-tts-mini.

    Parler-TTS takes text and a natural-language voice description. The voice
    description is resolved from the language registry on every generate() call
    so that switching target_language_code takes effect immediately.

    Args:
        model_id: HuggingFace model ID.
        target_language_code: Initial BCP-47 language code. The agent updates
            this attribute directly when the user switches language.
    """

    _DEFAULT_MODEL = "ai4bharat/indic-parler-tts-mini"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        target_language_code: str = "hi-IN",
    ):
        # Deferred imports: users running local/sarvam agents don't need torch.
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        print(f"Loading Indic Parler-TTS ({model_id}) on CPU...")
        print("  First load downloads ~1.8 GB from HuggingFace Hub.")

        self._device = "cpu"
        self._torch  = torch

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Read sample rate from model config (Parler-TTS mini = 44100 Hz).
        self.sample_rate: int = self._model.config.sampling_rate

        # Public; the agent updates this directly (same pattern as SarvamTTS).
        self.target_language_code = target_language_code

        print(f"Indic Parler-TTS ready (sr={self.sample_rate} Hz).")

    def generate(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech from text using the current target language.

        Args:
            text: Text to synthesize.

        Returns:
            Tuple of (audio_int16 numpy array, sample_rate int).
        """
        from sakura.languages.indic import get_voice_description

        if not text.strip():
            return np.array([], dtype=np.int16), self.sample_rate

        voice_desc = get_voice_description(self.target_language_code)

        input_ids = self._tokenizer(
            voice_desc, return_tensors="pt"
        ).input_ids.to(self._device)

        prompt_ids = self._tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self._device)

        with self._torch.no_grad():
            generation = self._model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_ids,
            )

        # generation: (1, sequence_length) float32 in [-1, 1]
        audio_float32 = generation.cpu().numpy().squeeze()

        # Clip then convert to int16
        audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)

        return audio_int16, self.sample_rate
