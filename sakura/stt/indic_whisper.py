"""IndicWhisper STT — HuggingFace Transformers pipeline, CPU-only, MIT license."""

import numpy as np


class IndicWhisperSTT:
    """Speech-to-Text using ai4bharat/indicwhisper-small (or configurable size).

    Uses the HuggingFace automatic-speech-recognition pipeline with
    torch.float32 for CPU compatibility. Language is auto-detected per
    utterance; the result is stored in self.detected_language as a BCP-47 code.

    Args:
        model_id: HuggingFace model ID. Defaults to the small variant.
        language: Seed BCP-47 language code. Used only for the initial value
            of detected_language; actual detection is always automatic.
    """

    _DEFAULT_MODEL = "ai4bharat/indicwhisper-small"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        language: str = "hi-IN",
    ):
        # Deferred import: users running local/sarvam agents don't need PyTorch.
        import torch
        from transformers import pipeline

        print(f"Loading IndicWhisper ({model_id}) on CPU...")
        print("  First load downloads ~950 MB from HuggingFace Hub.")

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch.float32,  # float16 not supported on CPU
            device="cpu",
        )
        # Updated after every transcribe() call.
        self.detected_language: str = language
        print("IndicWhisper STT ready.")

    def transcribe(self, audio_np: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a float32 audio array and auto-detect language.

        Args:
            audio_np: Float32 numpy array in range [-1, 1], mono.
            sample_rate: Sample rate; IndicWhisper expects 16 000 Hz.

        Returns:
            Transcribed text string (empty string if nothing detected).
        """
        from sakura.languages.indic import whisper_to_bcp47

        # return_timestamps=True is required — without it the pipeline omits the
        # "chunks" key and detected language is never returned. The full
        # transcription is still in result["text"] regardless of this flag.
        result = self._pipe(
            {"array": audio_np, "sampling_rate": sample_rate},
            generate_kwargs={"task": "transcribe", "language": None},
            return_timestamps=True,
        )

        text = result.get("text", "").strip() if isinstance(result, dict) else ""

        raw_lang = self._extract_language(result)
        if raw_lang:
            bcp47 = whisper_to_bcp47(raw_lang)
            if bcp47:
                self.detected_language = bcp47

        return text

    @staticmethod
    def _extract_language(result) -> str:
        """Pull the detected language name from the pipeline output dict.

        HuggingFace ASR pipeline places language in different locations
        depending on the transformers version. Tries all known paths.

        Returns:
            Lowercase language name string, or empty string if not found.
        """
        if not isinstance(result, dict):
            return ""
        # Newer transformers: result["chunks"][0]["language"]
        chunks = result.get("chunks")
        if chunks and isinstance(chunks, list) and chunks[0]:
            lang = chunks[0].get("language", "")
            if lang:
                return lang.lower().strip()
        # Older / alternate path: result["language"]
        lang = result.get("language", "")
        if lang:
            return lang.lower().strip()
        return ""
