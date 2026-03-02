from openai import OpenAI


class NvidiaNIM:
    """LLM chat completion via NVIDIA NIM (OpenAI-compatible API)."""

    _BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str, model: str = "meta/llama-3.1-8b-instruct",
                 temperature: float = 0.7, top_p: float = 0.9):
        self._client = OpenAI(base_url=self._BASE_URL, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def chat_completion(self, messages: list[dict], stream: bool = False):
        """Return a completion or streaming iterator."""
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=stream,
        )
