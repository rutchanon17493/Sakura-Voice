"""
Sarvam AI language and voice registry.

STT model coverage:
  saarika:v2.5  — 11 Indian languages + auto-detect
  saaras:v3     — 23 Indian languages (superset of v2.5)

TTS model coverage:
  bulbul:v3 / bulbul:v2  — 11 languages (same set as saarika:v2.5)

Languages only in saaras:v3 (as-IN … doi-IN) have no TTS support;
the agent falls back to hi-IN for speech output.
"""

# code → (display_name, stt_v2.5, stt_v3, tts_supported)
LANGUAGES: dict[str, tuple[str, bool, bool, bool]] = {
    "hi-IN":  ("Hindi",           True,  True,  True),
    "bn-IN":  ("Bengali",         True,  True,  True),
    "kn-IN":  ("Kannada",         True,  True,  True),
    "ml-IN":  ("Malayalam",       True,  True,  True),
    "mr-IN":  ("Marathi",         True,  True,  True),
    "od-IN":  ("Odia",            True,  True,  True),
    "pa-IN":  ("Punjabi",         True,  True,  True),
    "ta-IN":  ("Tamil",           True,  True,  True),
    "te-IN":  ("Telugu",          True,  True,  True),
    "en-IN":  ("English (India)", True,  True,  True),
    "gu-IN":  ("Gujarati",        True,  True,  True),
    # saaras:v3 only — no TTS support
    "as-IN":  ("Assamese",        False, True,  False),
    "ur-IN":  ("Urdu",            False, True,  False),
    "ne-IN":  ("Nepali",          False, True,  False),
    "kok-IN": ("Konkani",         False, True,  False),
    "ks-IN":  ("Kashmiri",        False, True,  False),
    "sd-IN":  ("Sindhi",          False, True,  False),
    "sa-IN":  ("Sanskrit",        False, True,  False),
    "sat-IN": ("Santali",         False, True,  False),
    "mni-IN": ("Manipuri",        False, True,  False),
    "brx-IN": ("Bodo",            False, True,  False),
    "mai-IN": ("Maithili",        False, True,  False),
    "doi-IN": ("Dogri",           False, True,  False),
    "unknown":("Auto-detect",     True,  True,  False),
}

SPEAKERS_V3: list[str] = [
    "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "ashutosh", "advait",
    "amelia", "sophia", "anand", "tanya", "tarun", "sunny", "mani", "gokul",
    "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
]

SPEAKERS_V2: list[str] = [
    "anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh",
]


def best_stt_model(code: str) -> str:
    """Return the most appropriate STT model for a language code."""
    entry = LANGUAGES.get(code)
    if entry and entry[1]:
        return "saarika:v2.5"
    return "saaras:v3"


def tts_language(code: str, fallback: str = "hi-IN") -> str:
    """Return the TTS language code, falling back if TTS doesn't support it."""
    entry = LANGUAGES.get(code)
    if entry and entry[3]:
        return code
    return fallback


def print_languages():
    """Print a formatted table of all supported languages."""
    print("\n== Sarvam AI Supported Languages ==================================")
    print(f"  {'Code':<10} {'Language':<22} {'STT Model':<16} {'TTS'}")
    print("  " + "-" * 60)
    for code, (name, v25, v3, tts) in LANGUAGES.items():
        stt_model = "saarika:v2.5" if v25 else "saaras:v3"
        tts_mark = "yes" if tts else "no (hi-IN fallback)"
        print(f"  {code:<10} {name:<22} {stt_model:<16} {tts_mark}")
    print()
