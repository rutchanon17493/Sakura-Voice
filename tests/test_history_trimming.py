"""
Tests for _trim_history() across all three agents.
No ML models, no API keys, no audio hardware required.
"""


def make_history(n_turns: int) -> list[dict]:
    """Build a fake history: 1 system + n_turns user/assistant pairs."""
    h = [{"role": "system", "content": "You are a test assistant."}]
    for i in range(n_turns):
        h.append({"role": "user",      "content": f"User turn {i}"})
        h.append({"role": "assistant", "content": f"Assistant turn {i}"})
    return h


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """Mirror of _trim_history() shared across all agents."""
    system = history[:1]
    turns  = history[1:]
    if len(turns) > max_turns * 2:
        turns = turns[-(max_turns * 2):]
    return system + turns


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_trim_when_under_limit():
    h = make_history(5)
    result = trim_history(h, max_turns=20)
    assert len(result) == len(h)


def test_no_trim_at_exact_limit():
    h = make_history(20)
    result = trim_history(h, max_turns=20)
    assert len(result) == len(h)


def test_trims_one_extra_turn():
    h = make_history(21)           # 1 system + 42 turn messages
    result = trim_history(h, max_turns=20)
    assert len(result) == 1 + 40   # system + 40 messages (20 pairs)


def test_system_message_always_preserved():
    h = make_history(100)
    result = trim_history(h, max_turns=5)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a test assistant."


def test_keeps_most_recent_turns():
    h = make_history(10)
    result = trim_history(h, max_turns=3)
    # After trimming to 3 pairs, the last messages should be from turns 7, 8, 9
    assert result[-1]["content"] == "Assistant turn 9"
    assert result[-2]["content"] == "User turn 9"
    assert result[1]["content"]  == "User turn 7"


def test_oldest_turns_dropped():
    h = make_history(10)
    result = trim_history(h, max_turns=3)
    contents = [m["content"] for m in result]
    assert "User turn 0" not in contents
    assert "User turn 6" not in contents


def test_empty_history_unchanged():
    h = [{"role": "system", "content": "sys"}]
    result = trim_history(h, max_turns=20)
    assert result == h


def test_max_turns_1():
    h = make_history(5)
    result = trim_history(h, max_turns=1)
    assert len(result) == 3   # system + 1 user + 1 assistant
    assert result[1]["content"] == "User turn 4"
    assert result[2]["content"] == "Assistant turn 4"


if __name__ == "__main__":
    tests = [
        test_no_trim_when_under_limit,
        test_no_trim_at_exact_limit,
        test_trims_one_extra_turn,
        test_system_message_always_preserved,
        test_keeps_most_recent_turns,
        test_oldest_turns_dropped,
        test_empty_history_unchanged,
        test_max_turns_1,
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
