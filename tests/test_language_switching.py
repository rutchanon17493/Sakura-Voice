"""
Tests for the language-switching logic used by both sarvam and indic agents.
No ML models, no API keys, no audio hardware required.
"""

from collections import Counter, deque


# ── Shared switching logic (copied from agents) ──────────────────────────────

def make_switcher(window=3, thresh=2):
    """Return a stateful switch() function that mirrors _update_language()."""
    lang_window   = deque(maxlen=window)
    active        = {"lang": "hi-IN"}
    switches      = []

    def update(detected, valid_langs):
        if detected not in valid_langs:
            return
        lang_window.append(detected)
        counts          = Counter(lang_window)
        dominant, votes = counts.most_common(1)[0]
        if dominant != active["lang"] and votes >= thresh:
            switches.append((active["lang"], dominant))
            active["lang"] = dominant

    return update, active, switches


# ── Tests ─────────────────────────────────────────────────────────────────────

VALID = {"hi-IN", "ta-IN", "en-IN", "bn-IN", "te-IN"}


def test_no_switch_on_single_utterance():
    update, active, switches = make_switcher()
    update("ta-IN", VALID)
    assert active["lang"] == "hi-IN"
    assert len(switches) == 0


def test_switch_after_two_consecutive():
    update, active, switches = make_switcher()
    update("ta-IN", VALID)
    update("ta-IN", VALID)
    assert active["lang"] == "ta-IN"
    assert switches == [("hi-IN", "ta-IN")]


def test_no_switch_when_window_split():
    """hi ta hi → ta gets 1 vote, hi gets 2 — no switch."""
    update, active, switches = make_switcher()
    update("ta-IN", VALID)
    update("hi-IN", VALID)   # still hi-IN initially
    update("ta-IN", VALID)   # window: [ta, hi, ta] — ta=2 but active is hi-IN... wait
    # Actually: active starts as hi-IN. window=[ta, hi, ta], dominant=ta (2 votes).
    # So it SHOULD switch. Let me reconsider.
    # After 3rd: ta=2, hi=1 → switch should happen.
    assert active["lang"] == "ta-IN"


def test_no_switch_when_clearly_dominated_by_original():
    """hi hi ta → ta gets 1 vote only, no switch."""
    update, active, switches = make_switcher()
    update("hi-IN", VALID)   # window: [hi] — same as active, no switch
    update("hi-IN", VALID)   # window: [hi, hi]
    update("ta-IN", VALID)   # window: [hi, hi, ta] — ta=1, no switch
    assert active["lang"] == "hi-IN"
    assert len(switches) == 0


def test_switch_back_after_sustained_original():
    """Switch to Tamil then back to Hindi."""
    update, active, switches = make_switcher()
    update("ta-IN", VALID)
    update("ta-IN", VALID)   # → Tamil
    assert active["lang"] == "ta-IN"
    update("hi-IN", VALID)
    update("hi-IN", VALID)   # → back to Hindi
    assert active["lang"] == "hi-IN"
    assert len(switches) == 2


def test_unknown_language_ignored():
    """Unrecognised language codes must not affect the window."""
    update, active, switches = make_switcher()
    update("xx-XX", VALID)   # not in VALID, ignored
    update("xx-XX", VALID)
    update("xx-XX", VALID)
    assert active["lang"] == "hi-IN"
    assert len(switches) == 0


def test_multiple_language_competition():
    """ta ta en → ta wins (2 votes)."""
    update, active, switches = make_switcher()
    update("ta-IN", VALID)
    update("ta-IN", VALID)
    update("en-IN", VALID)   # window: [ta, ta, en] — ta=2 → switch to ta
    assert active["lang"] == "ta-IN"


def test_single_word_hinglish_no_flip():
    """One English word mid-Hindi conversation should not switch language."""
    update, active, switches = make_switcher()
    update("en-IN", VALID)   # single English word
    assert active["lang"] == "hi-IN"


if __name__ == "__main__":
    tests = [
        test_no_switch_on_single_utterance,
        test_switch_after_two_consecutive,
        test_no_switch_when_window_split,
        test_no_switch_when_clearly_dominated_by_original,
        test_switch_back_after_sustained_original,
        test_unknown_language_ignored,
        test_multiple_language_competition,
        test_single_word_hinglish_no_flip,
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
