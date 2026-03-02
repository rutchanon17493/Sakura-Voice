import json

import requests


class SarvamLLM:
    """Chat completion using Sarvam AI's Sarvam-M model."""

    _BASE_URL = "https://api.sarvam.ai/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "sarvam-m",
                 temperature: float = 0.7, max_tokens: int = 256,
                 wiki_grounding: bool = False):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.wiki_grounding = wiki_grounding
        print(f"Sarvam LLM ready (model={model})")

    def chat_completion(self, messages: list[dict], stream: bool = False):
        """
        Call the Sarvam chat completions API.

        Args:
            messages: List of role/content dicts.
            stream: If True returns an SSE chunk iterator, else a parsed dict.
        """
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            "wiki_grounding": self.wiki_grounding,
        }
        response = requests.post(
            self._BASE_URL,
            headers={
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            stream=stream,
            timeout=60,
        )
        if not response.ok:
            print(f"LLM error {response.status_code}: {response.text}")
            response.raise_for_status()

        if not stream:
            return response.json()
        return self._iter_sse(response)

    @staticmethod
    def _iter_sse(response):
        for line in response.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
