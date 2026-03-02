"""
Smoke test for OllamaLLM.
Requires Ollama running locally: ollama serve
And the model pulled:          ollama pull qwen2.5:7b

Skip this test if Ollama is not running.
"""

import pytest


def ollama_available():
    """Return True if Ollama is reachable."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False


@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_ollama_non_streaming():
    from sakura.llm import OllamaLLM
    llm    = OllamaLLM(max_tokens=32)
    result = llm.chat_completion(
        [{"role": "user", "content": "Reply with exactly the word: hello"}],
        stream=False,
    )
    text = result.choices[0].message.content.strip().lower()
    assert len(text) > 0, "Empty response from Ollama"
    print(f"  Response: {text}")


@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_ollama_streaming():
    from sakura.llm import OllamaLLM
    llm    = OllamaLLM(max_tokens=32)
    stream = llm.chat_completion(
        [{"role": "user", "content": "Say hi in one word."}],
        stream=True,
    )
    tokens = []
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            tokens.append(token)
    assert len(tokens) > 0, "No tokens streamed from Ollama"
    print(f"  Streamed {len(tokens)} tokens: {''.join(tokens)[:60]}")


@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_ollama_hindi_response():
    from sakura.llm import OllamaLLM
    llm    = OllamaLLM(max_tokens=64)
    result = llm.chat_completion(
        [
            {"role": "system",  "content": "You MUST reply in Hindi only."},
            {"role": "user",    "content": "नमस्ते, आप कैसे हैं?"},
        ],
        stream=False,
    )
    text = result.choices[0].message.content.strip()
    assert len(text) > 0
    print(f"  Hindi response: {text}")


if __name__ == "__main__":
    if not ollama_available():
        print("SKIP — Ollama not running (start with: ollama serve)")
    else:
        tests = [
            test_ollama_non_streaming,
            test_ollama_streaming,
            test_ollama_hindi_response,
        ]
        passed = 0
        for t in tests:
            try:
                t()
                print(f"  PASS  {t.__name__}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {t.__name__}: {e}")
        print(f"\n{passed}/{len(tests)} passed")
