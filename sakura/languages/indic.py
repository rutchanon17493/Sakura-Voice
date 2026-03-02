"""
Indic language registry for the fully-local indic agent.

IndicWhisper returns detected language as a lowercase English name
(e.g. "hindi", "tamil"). WHISPER_LANG_MAP converts those to BCP-47
codes used everywhere else in the system.

Indic Parler-TTS (ai4bharat/indic-parler-tts-mini) supports 21 Indic
languages. TTS_VOICE_DESCRIPTIONS maps each BCP-47 code to a natural-
language voice-description prompt understood by Parler-TTS.
"""

# BCP-47 → (display_name, parler_tts_supported)
INDIC_LANGUAGES: dict[str, tuple[str, bool]] = {
    "hi-IN":  ("Hindi",            True),
    "bn-IN":  ("Bengali",          True),
    "kn-IN":  ("Kannada",          True),
    "ml-IN":  ("Malayalam",        True),
    "mr-IN":  ("Marathi",          True),
    "od-IN":  ("Odia",             True),
    "pa-IN":  ("Punjabi",          True),
    "ta-IN":  ("Tamil",            True),
    "te-IN":  ("Telugu",           True),
    "en-IN":  ("English (India)",  True),
    "gu-IN":  ("Gujarati",         True),
    "as-IN":  ("Assamese",         True),
    "ur-IN":  ("Urdu",             True),
    "ne-IN":  ("Nepali",           True),
    "sa-IN":  ("Sanskrit",         False),  # limited Parler-TTS support
    "kok-IN": ("Konkani",          False),
    "mai-IN": ("Maithili",         False),
}

# Whisper language name (lowercase) → BCP-47 code.
# Both "odia" and "oriya" are included — Whisper uses either depending on version.
WHISPER_LANG_MAP: dict[str, str] = {
    "hindi":     "hi-IN",
    "bengali":   "bn-IN",
    "kannada":   "kn-IN",
    "malayalam": "ml-IN",
    "marathi":   "mr-IN",
    "odia":      "od-IN",
    "oriya":     "od-IN",
    "punjabi":   "pa-IN",
    "tamil":     "ta-IN",
    "telugu":    "te-IN",
    "english":   "en-IN",
    "gujarati":  "gu-IN",
    "assamese":  "as-IN",
    "urdu":      "ur-IN",
    "nepali":    "ne-IN",
    "sanskrit":  "sa-IN",
    "konkani":   "kok-IN",
    "maithili":  "mai-IN",
    "sindhi":    "ur-IN",   # closest supported fallback (shared script)
}

# BCP-47 → Parler-TTS natural language voice description.
# Kept short: longer prompts slow down CPU inference significantly.
TTS_VOICE_DESCRIPTIONS: dict[str, str] = {
    "hi-IN":  "A female speaker with a warm, clear voice speaks in Hindi at a moderate pace.",
    "bn-IN":  "A female speaker with a warm, clear voice speaks in Bengali at a moderate pace.",
    "kn-IN":  "A female speaker with a clear, pleasant voice speaks in Kannada at a moderate pace.",
    "ml-IN":  "A female speaker with a clear voice speaks in Malayalam at a moderate pace.",
    "mr-IN":  "A female speaker with a warm voice speaks in Marathi at a moderate pace.",
    "od-IN":  "A female speaker with a clear voice speaks in Odia at a moderate pace.",
    "pa-IN":  "A female speaker with a warm voice speaks in Punjabi at a moderate pace.",
    "ta-IN":  "A female speaker with a clear voice speaks in Tamil at a moderate pace.",
    "te-IN":  "A female speaker with a warm voice speaks in Telugu at a moderate pace.",
    "en-IN":  "A female speaker with a clear Indian English accent speaks at a moderate pace.",
    "gu-IN":  "A female speaker with a clear voice speaks in Gujarati at a moderate pace.",
    "as-IN":  "A female speaker with a clear voice speaks in Assamese at a moderate pace.",
    "ur-IN":  "A female speaker with a warm voice speaks in Urdu at a moderate pace.",
    "ne-IN":  "A female speaker with a clear voice speaks in Nepali at a moderate pace.",
    "sa-IN":  "A female speaker with a clear voice speaks in Hindi at a moderate pace.",   # fallback
    "kok-IN": "A female speaker with a clear voice speaks in Hindi at a moderate pace.",   # fallback
    "mai-IN": "A female speaker with a clear voice speaks in Hindi at a moderate pace.",   # fallback
}

_TTS_FALLBACK = "hi-IN"


def whisper_to_bcp47(whisper_lang: str) -> str | None:
    """Convert a Whisper-returned language name to a BCP-47 code.

    Args:
        whisper_lang: Lowercase language name as returned by IndicWhisper
            (e.g. "hindi", "tamil").

    Returns:
        BCP-47 code string, or None if the language is not recognised.
    """
    return WHISPER_LANG_MAP.get(whisper_lang.lower())


def indic_tts_language(code: str, fallback: str = _TTS_FALLBACK) -> str:
    """Return the TTS language code, falling back when Parler-TTS lacks support.

    Args:
        code: BCP-47 language code.
        fallback: Code to use when TTS is not supported.

    Returns:
        A BCP-47 code guaranteed to have a TTS voice description.
    """
    entry = INDIC_LANGUAGES.get(code)
    if entry and entry[1]:
        return code
    return fallback


def get_voice_description(code: str) -> str:
    """Return the Parler-TTS voice description for a BCP-47 code.

    Args:
        code: BCP-47 language code.

    Returns:
        Natural-language voice prompt string.
    """
    return TTS_VOICE_DESCRIPTIONS.get(
        code,
        TTS_VOICE_DESCRIPTIONS[_TTS_FALLBACK],
    )


def print_indic_languages():
    """Print a formatted table of all indic agent supported languages."""
    print("\n== Indic Agent Supported Languages =================================")
    print(f"  {'Code':<10} {'Language':<22} {'TTS'}")
    print("  " + "-" * 50)
    for code, (name, tts) in INDIC_LANGUAGES.items():
        tts_mark = "yes" if tts else f"no ({_TTS_FALLBACK} fallback)"
        print(f"  {code:<10} {name:<22} {tts_mark}")
    print()
