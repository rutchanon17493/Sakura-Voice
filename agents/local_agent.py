"""Local voice agent using Faster-Whisper STT, NVIDIA NIM LLM, and Kokoro TTS."""

import os
import time

import numpy as np
from dotenv import load_dotenv

from sakura.audio import AudioEngine
from sakura.vad import SileroVAD
from sakura.stt import FasterWhisperSTT
from sakura.llm import NvidiaNIM
from sakura.tts import KokoroTTS

load_dotenv()

SYSTEM_PROMPT      = os.getenv("SYSTEM_PROMPT", "You are a helpful voice assistant. Be concise.")
_dev               = os.getenv("AUDIO_INPUT_DEVICE")
AUDIO_INPUT_DEVICE = int(_dev) if _dev is not None else None
MAX_HISTORY_TURNS  = int(os.getenv("MAX_HISTORY_TURNS", "20"))  # user+assistant pairs to keep


class LocalVoiceAgent:
    """Voice agent powered entirely by local / NVIDIA NIM models."""

    def __init__(self):
        print("Initializing Local Voice Agent...")
        self.audio = AudioEngine(input_rate=16000, output_rate=24000, chunk=512,
                               input_device_index=AUDIO_INPUT_DEVICE)
        self.vad   = SileroVAD(threshold=0.5)
        self.stt   = FasterWhisperSTT(model_size="tiny.en")
        self.llm   = NvidiaNIM(api_key=os.getenv("NVIDIA_API_KEY"))
        self.tts   = KokoroTTS()

        self._is_speaking   = False
        self._speech_buffer = bytearray()
        self._silence_start = None
        self._history       = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _greet(self):
        """Generate and speak an opening greeting."""
        print("Sakura is thinking...")
        stream       = self.llm.chat_completion(
            self._history + [{"role": "user", "content": "Greet the user warmly and introduce yourself briefly."}],
            stream=True,
        )
        full_text    = ""
        current_sent = ""
        print("Sakura: ", end="", flush=True)
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
                full_text    += token
                current_sent += token
                if token in (".", "!", "?", "\n"):
                    self._speak(current_sent)
                    current_sent = ""
        if current_sent.strip():
            self._speak(current_sent)
        print()
        self._history.append({"role": "assistant", "content": full_text})

    def run(self):
        print("Starting audio loop...")
        self.audio.start()
        print("\nLocal Agent Ready.")

        self._greet()
        print("Listening...")

        try:
            while True:
                chunk     = self.audio.read_chunk()
                is_speech = self.vad.is_speech(chunk)

                if is_speech:
                    if not self._is_speaking:
                        print("User started speaking...")
                        self._is_speaking = True
                    self._speech_buffer.extend(chunk)
                    self._silence_start = None
                else:
                    if self._is_speaking:
                        if self._silence_start is None:
                            self._silence_start = time.time()
                        elif time.time() - self._silence_start > 0.6:
                            print("User finished speaking. Processing...")
                            self._process(bytes(self._speech_buffer))
                            self._is_speaking   = False
                            self._speech_buffer = bytearray()
                            self._silence_start = None
                            print("Listening...")

        except KeyboardInterrupt:
            print("\nStopping...")
            self.audio.stop()

    def _trim_history(self):
        """Keep system message + last MAX_HISTORY_TURNS user/assistant pairs."""
        system = self._history[:1]
        turns  = self._history[1:]
        if len(turns) > MAX_HISTORY_TURNS * 2:
            turns = turns[-(MAX_HISTORY_TURNS * 2):]
        self._history = system + turns

    def _process(self, audio_bytes: bytes):
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            text = self.stt.transcribe(audio_np)
        except Exception as e:
            print(f"STT error: {e}")
            return
        if not text:
            return
        print(f"User: {text}")
        self._history.append({"role": "user", "content": text})

        print("Assistant: ", end="", flush=True)
        full_text    = ""
        current_sent = ""
        try:
            stream = self.llm.chat_completion(self._history, stream=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="", flush=True)
                    full_text    += token
                    current_sent += token
                    if token in (".", "!", "?", "\n"):
                        self._speak(current_sent)
                        current_sent = ""
            if current_sent.strip():
                self._speak(current_sent)
        except Exception as e:
            print(f"\nLLM error: {e}")
            self._history.pop()  # remove the user turn so history stays consistent
            self._speak("Sorry, I had trouble responding. Please try again.")
            return

        print("\n")
        self._history.append({"role": "assistant", "content": full_text})
        self._trim_history()

    def _speak(self, text: str):
        if not text.strip():
            return
        try:
            audio_int16, _ = self.tts.generate(text)
            self.audio.write_chunk(audio_int16)
        except Exception as e:
            print(f"TTS error: {e}")
