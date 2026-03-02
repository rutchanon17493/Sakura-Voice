"""Voice agent powered by Sarvam AI — supports 23 Indian languages."""

import os
import time
from collections import Counter, deque

import numpy as np
from dotenv import load_dotenv

from sakura.audio import AudioEngine
from sakura.vad import SileroVAD
from sakura.stt import SarvamSTT
from sakura.llm import SarvamLLM
from sakura.tts import SarvamTTS
from sakura.languages import LANGUAGES, best_stt_model, tts_language

load_dotenv()

SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY")
SYSTEM_PROMPT      = os.getenv("SYSTEM_PROMPT", "You are a helpful voice assistant. Be concise.")
LANGUAGE_CODE      = os.getenv("SARVAM_LANGUAGE", "hi-IN")
SPEAKER            = os.getenv("SARVAM_SPEAKER", "Shubh")
_dev               = os.getenv("AUDIO_INPUT_DEVICE")
AUDIO_INPUT_DEVICE = int(_dev) if _dev is not None else None
MAX_HISTORY_TURNS  = int(os.getenv("MAX_HISTORY_TURNS", "20"))  # user+assistant pairs to keep

# How many consecutive utterances must agree before switching language.
_SWITCH_WINDOW = 3   # window size
_SWITCH_THRESH = 2   # votes needed within window


class SarvamVoiceAgent:
    """
    Voice agent using Sarvam AI's Saarika STT, Sarvam-M LLM, and Bulbul TTS.

    Set SARVAM_LANGUAGE in .env to any supported BCP-47 code.
    Run  python run.py --list-languages  to see all options.
    """

    def __init__(self, language: str = LANGUAGE_CODE, speaker: str = SPEAKER):
        if language not in LANGUAGES:
            raise ValueError(
                f"Unknown language '{language}'. "
                "Run  python run.py --list-languages  to see options."
            )

        if not SARVAM_API_KEY:
            raise ValueError("SARVAM_API_KEY not set in .env")

        self.language  = language
        lang_name  = LANGUAGES[language][0]
        stt_model  = best_stt_model(language)
        tts_lang   = tts_language(language)
        tts_note   = " [fallback]" if tts_lang != language else ""

        print("Initializing Sarvam Voice Agent...")
        print(f"  Start lang: {lang_name} ({language})")
        print(f"  STT model : saaras:v3 (auto-detect, 23 languages)")
        print(f"  TTS lang  : {tts_lang}{tts_note}")
        print(f"  Speaker   : {speaker}")

        self.audio = AudioEngine(input_rate=16000, output_rate=22050, chunk=512,
                               input_device_index=AUDIO_INPUT_DEVICE)
        self.vad   = SileroVAD(threshold=0.5)
        # saaras:v3 + unknown = auto-detect across all 23 Indian languages.
        # saarika:v2.5 only covers 11 — not enough for full multilingual switching.
        self.stt   = SarvamSTT(api_key=SARVAM_API_KEY, model="saaras:v3",
                               language_code="unknown")
        self.llm   = SarvamLLM(api_key=SARVAM_API_KEY, model="sarvam-m",
                               temperature=0.7, max_tokens=256)
        self.tts   = SarvamTTS(api_key=SARVAM_API_KEY, model="bulbul:v3",
                               speaker=speaker.lower(), target_language_code=tts_lang,
                               pace=1.0)

        self._is_speaking    = False
        self._speech_buffer  = bytearray()
        self._silence_start  = None
        self._history        = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Language tracking
        self._active_language = language          # current confirmed language
        self._lang_window     = deque(maxlen=_SWITCH_WINDOW)

    def _greet(self):
        """Generate and speak an opening greeting."""
        print("Sakura is thinking...")
        greeting_messages = self._build_llm_messages() + [
            {"role": "user", "content": "Greet the user warmly and introduce yourself briefly."}
        ]
        result = self.llm.chat_completion(greeting_messages, stream=False,)
        greeting = result["choices"][0]["message"]["content"]
        # Sarvam API requires first non-system message to be from user,
        # so store the prompt that triggered the greeting in history.
        self._history.append({"role": "user", "content": "Greet the user warmly and introduce yourself briefly."})
        self._history.append({"role": "assistant", "content": greeting})
        print(f"Sakura: {greeting}")
        self._speak(greeting)

    def run(self):
        print("Starting audio loop...")
        self.audio.start()
        lang_name = LANGUAGES[self.language][0]
        print(f"\nSarvam Agent Ready [{lang_name}].")

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

    def _update_language(self, detected: str):
        """Apply smoothing window and switch active language if votes cross threshold."""
        # Only count known languages with TTS support (or en-IN / hi-IN)
        if detected not in LANGUAGES or detected == "unknown":
            return
        self._lang_window.append(detected)
        counts  = Counter(self._lang_window)
        dominant, votes = counts.most_common(1)[0]
        if dominant != self._active_language and votes >= _SWITCH_THRESH:
            old_name = LANGUAGES.get(self._active_language, (self._active_language,))[0]
            new_name = LANGUAGES.get(dominant, (dominant,))[0]
            self._active_language         = dominant
            self.tts.target_language_code = tts_language(dominant)
            print(f"  [Language: {old_name} → {new_name}]")

    def _build_llm_messages(self) -> list[dict]:
        """Return history with a language-aware system prompt for this turn."""
        lang_name = LANGUAGES.get(self._active_language, (self._active_language,))[0]
        augmented_system = (
            f"{SYSTEM_PROMPT}\n\n"
            f"IMPORTANT: The user is currently speaking {lang_name}. "
            f"You MUST reply in {lang_name} only. Do not switch languages."
        )
        return [{"role": "system", "content": augmented_system}] + self._history[1:]

    def _trim_history(self):
        """Keep system message + last MAX_HISTORY_TURNS user/assistant pairs."""
        system = self._history[:1]
        turns  = self._history[1:]
        if len(turns) > MAX_HISTORY_TURNS * 2:
            turns = turns[-(MAX_HISTORY_TURNS * 2):]
        self._history = system + turns

    def _process(self, audio_bytes: bytes):
        audio_np   = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = len(audio_np) / 16000
        rms        = float(np.sqrt(np.mean(audio_np ** 2)))
        print(f"  Audio: {duration_s:.2f}s, RMS={rms:.4f}")

        try:
            text = self.stt.transcribe(audio_np, sample_rate=16000)
        except Exception as e:
            print(f"STT error: {e}")
            return
        if not text:
            print("(no speech detected)")
            return

        # Detect and potentially switch language based on this utterance.
        self._update_language(self.stt.detected_language)

        lang_name = LANGUAGES.get(self._active_language, (self._active_language,))[0]
        print(f"User [{lang_name}]: {text}")
        self._history.append({"role": "user", "content": text})

        # Stream token-by-token and speak sentence-by-sentence so TTS starts
        # as soon as the first sentence is ready instead of waiting for the
        # full response.
        print("Assistant: ", end="", flush=True)
        full_text    = ""
        current_sent = ""
        try:
            stream = self.llm.chat_completion(self._build_llm_messages(), stream=True)
            for chunk in stream:
                token = (
                    chunk.get("choices", [{}])[0]
                         .get("delta", {})
                         .get("content") or ""
                )
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
        sentences, current = [], ""
        for ch in text:
            current += ch
            if ch in (".", "!", "?", "\n") and current.strip():
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        for sentence in sentences:
            try:
                audio_int16, _ = self.tts.generate(sentence)
                if audio_int16.size > 0:
                    self.audio.write_chunk(audio_int16)
            except Exception as e:
                print(f"TTS error: {e}")
