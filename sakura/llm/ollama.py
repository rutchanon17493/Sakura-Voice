"""OllamaLLM — local LLM via Ollama's OpenAI-compatible API."""

from openai import OpenAI


class OllamaLLM:
    """Chat completion via a locally-running Ollama instance.

    Ollama exposes an OpenAI-compatible HTTP API at http://localhost:11434/v1.
    Any model pulled with ``ollama pull <model>`` is accessible.

    The default model is qwen2.5:7b which has strong multilingual Indic
    support at a manageable ~4 GB Q4_K_M RAM footprint on CPU.

    Args:
        model: Ollama model tag (e.g. "qwen2.5:7b", "llama3.2:3b").
        base_url: Ollama API base URL. Override for remote Ollama instances.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_tokens: Maximum tokens to generate per response.
    """

    _DEFAULT_MODEL    = "qwen2.5:7b"
    _DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ):
        # Ollama ignores the API key; the openai client requires a non-empty string.
        self._client     = OpenAI(base_url=base_url, api_key="ollama")
        self.model       = model
        self.temperature = temperature
        self.top_p       = top_p
        self.max_tokens  = max_tokens
        print(f"Ollama LLM ready (model={model}, url={base_url})")

    def chat_completion(self, messages: list[dict], stream: bool = False):
        """Return a completion or streaming iterator.

        Args:
            messages: List of role/content dicts (OpenAI format).
            stream: If True, returns a streaming iterator whose chunks expose
                ``chunk.choices[0].delta.content`` (identical to NvidiaNIM).

        Returns:
            OpenAI ChatCompletion object (stream=False) or streaming iterator.
        """
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=stream,
        )
