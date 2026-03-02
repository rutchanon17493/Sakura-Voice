import queue
import numpy as np
import pyaudio


class AudioEngine:
    """PyAudio-backed microphone input and speaker output engine."""

    def __init__(self, input_rate: int = 16000, output_rate: int = 24000,
                 chunk: int = 512, channels: int = 1,
                 input_device_index: int | None = None):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.chunk = chunk
        self.channels = channels
        self.input_device_index = input_device_index
        self._p = pyaudio.PyAudio()
        self._stream_in = None
        self._stream_out = None
        self._input_queue: queue.Queue = queue.Queue()
        self.is_running = False

    @staticmethod
    def list_devices():
        """Print all available audio input devices."""
        p = pyaudio.PyAudio()
        print("\n=== Available Microphones ===")
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["maxInputChannels"] > 0:
                print(f"  [{i}] {d['name']}")
        print("Set AUDIO_INPUT_DEVICE=<index> in .env to select one.\n")
        p.terminate()

    def start(self):
        self.is_running = True

        if self.input_device_index is not None:
            d = self._p.get_device_info_by_index(self.input_device_index)
            print(f"  Microphone: [{self.input_device_index}] {d['name']}")
        else:
            print("  Microphone: system default")

        self._stream_in = self._p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.input_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self._input_callback,
        )
        self._stream_out = self._p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.output_rate,
            output=True,
        )
        self._stream_in.start_stream()
        self._stream_out.start_stream()

    def _input_callback(self, in_data, frame_count, time_info, status):
        self._input_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def read_chunk(self) -> bytes:
        return self._input_queue.get()

    def write_chunk(self, audio_data):
        """Write int16 numpy array or raw bytes to the speaker."""
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.tobytes()
        self._stream_out.write(audio_data)

    def stop(self):
        self.is_running = False
        for stream in (self._stream_in, self._stream_out):
            if stream:
                stream.stop_stream()
                stream.close()
        self._p.terminate()
