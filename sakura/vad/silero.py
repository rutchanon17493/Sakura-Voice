"""Silero VAD — neural Voice Activity Detection using the official ONNX model."""

import numpy as np

from sakura.config import MODELS_DIR


class SileroVAD:
    """Neural Voice Activity Detection using the Silero VAD v4/v5 ONNX model.

    Uses a lightweight LSTM-based neural network to distinguish speech from
    silence/noise. Far more accurate than energy thresholding, especially for
    quiet speech and noisy environments.

    The model is stateful: LSTM hidden states are preserved between chunks so
    detection improves within a continuous stream. Call reset() between
    separate conversations if needed.

    Args:
        threshold: Speech probability threshold (0–1). Default 0.5 is a
            balanced operating point. Raise to reduce false positives;
            lower to catch quieter speech.
        sample_rate: Audio sample rate in Hz. Silero VAD supports 8000 and
            16000 Hz. Agents use 16000 Hz.
    """

    _MODEL_FILE = "silero_vad.onnx"

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        import onnxruntime as ort

        model_path = MODELS_DIR / self._MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found: {model_path}\n"
                "The file should be at models/silero_vad.onnx."
            )

        self.threshold   = threshold
        self.sample_rate = sample_rate

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), sess_options=opts)

        # Discover actual input names from the model (handles v4 and v5).
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        self._reset_state()
        print(f"VAD ready (Silero ONNX, threshold={threshold})")

    def _reset_state(self):
        """Initialise (or reset) the LSTM hidden and cell states."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        """Reset LSTM state. Call between separate recording sessions."""
        self._reset_state()

    def is_speech(self, audio_chunk) -> bool:
        """Return True if the chunk contains speech.

        Args:
            audio_chunk: Raw audio as bytes (int16 PCM) or a float32 numpy
                array in range [-1, 1]. Must be exactly one VAD window
                (chunk_size samples as configured in AudioEngine).

        Returns:
            bool: True if the neural network assigns speech probability
                above self.threshold.
        """
        if isinstance(audio_chunk, bytes):
            wav = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            wav = np.asarray(audio_chunk, dtype=np.float32)

        # Model expects shape (1, N)
        chunk = wav.reshape(1, -1)

        ort_inputs = {
            "input": chunk,
            "sr":    np.array(self.sample_rate, dtype=np.int64),
            "h":     self._h,
            "c":     self._c,
        }

        output, self._h, self._c = self._session.run(None, ort_inputs)

        # output shape is (1, 1) or scalar — squeeze to float
        prob = float(np.squeeze(output))
        return prob > self.threshold
