"""Tests for the indic language registry (no ML models required)."""

from sakura.languages.indic import (
    INDIC_LANGUAGES,
    WHISPER_LANG_MAP,
    whisper_to_bcp47,
    indic_tts_language,
    get_voice_description,
)


def test_whisper_to_bcp47_known():
    assert whisper_to_bcp47("hindi")     == "hi-IN"
    assert whisper_to_bcp47("tamil")     == "ta-IN"
    assert whisper_to_bcp47("telugu")    == "te-IN"
    assert whisper_to_bcp47("english")   == "en-IN"
    assert whisper_to_bcp47("bengali")   == "bn-IN"
    assert whisper_to_bcp47("kannada")   == "kn-IN"
    assert whisper_to_bcp47("malayalam") == "ml-IN"
    assert whisper_to_bcp47("marathi")   == "mr-IN"
    assert whisper_to_bcp47("punjabi")   == "pa-IN"
    assert whisper_to_bcp47("gujarati")  == "gu-IN"
    assert whisper_to_bcp47("assamese")  == "as-IN"
    assert whisper_to_bcp47("urdu")      == "ur-IN"
    assert whisper_to_bcp47("nepali")    == "ne-IN"


def test_whisper_to_bcp47_odia_both_spellings():
    # Whisper uses "odia" or "oriya" depending on model version — both must map
    assert whisper_to_bcp47("odia")  == "od-IN"
    assert whisper_to_bcp47("oriya") == "od-IN"


def test_whisper_to_bcp47_case_insensitive():
    assert whisper_to_bcp47("HINDI")  == "hi-IN"
    assert whisper_to_bcp47("Tamil")  == "ta-IN"
    assert whisper_to_bcp47("ENGLISH") == "en-IN"


def test_whisper_to_bcp47_unknown_returns_none():
    assert whisper_to_bcp47("klingon") is None
    assert whisper_to_bcp47("") is None
    assert whisper_to_bcp47("unknown") is None


def test_indic_tts_language_supported():
    # Languages with TTS support should return themselves
    for code, (_, tts_ok) in INDIC_LANGUAGES.items():
        if tts_ok:
            assert indic_tts_language(code) == code, f"{code} should not fall back"


def test_indic_tts_language_fallback():
    # Languages without TTS support should fall back to hi-IN
    assert indic_tts_language("sa-IN")  == "hi-IN"
    assert indic_tts_language("kok-IN") == "hi-IN"
    assert indic_tts_language("mai-IN") == "hi-IN"


def test_indic_tts_language_unknown_code_fallback():
    assert indic_tts_language("xx-XX") == "hi-IN"


def test_get_voice_description_returns_string():
    for code in INDIC_LANGUAGES:
        desc = get_voice_description(code)
        assert isinstance(desc, str) and len(desc) > 0, f"Empty description for {code}"


def test_get_voice_description_unknown_fallback():
    desc = get_voice_description("xx-XX")
    assert isinstance(desc, str) and len(desc) > 0


def test_all_whisper_map_values_in_indic_languages():
    # Every BCP-47 code in the Whisper map should exist in INDIC_LANGUAGES
    for whisper_name, bcp47 in WHISPER_LANG_MAP.items():
        assert bcp47 in INDIC_LANGUAGES, (
            f"WHISPER_LANG_MAP['{whisper_name}'] = '{bcp47}' "
            f"not found in INDIC_LANGUAGES"
        )


if __name__ == "__main__":
    # Run manually: python tests/test_indic_languages.py
    tests = [
        test_whisper_to_bcp47_known,
        test_whisper_to_bcp47_odia_both_spellings,
        test_whisper_to_bcp47_case_insensitive,
        test_whisper_to_bcp47_unknown_returns_none,
        test_indic_tts_language_supported,
        test_indic_tts_language_fallback,
        test_indic_tts_language_unknown_code_fallback,
        test_get_voice_description_returns_string,
        test_get_voice_description_unknown_fallback,
        test_all_whisper_map_values_in_indic_languages,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
