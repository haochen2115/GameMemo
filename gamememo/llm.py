# -*- coding: utf-8 -*-
"""LLM backends.

Everything in GameMemo talks to a model through the small ``LLMClient``
protocol below, so tests can swap in ``FakeLLM`` and experiments can swap
models without touching memory logic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import requests


Message = Dict[str, str]


class LLMClient(Protocol):
    def chat(self,
             prompt: Optional[str] = None,
             system: str = "",
             temperature: float = 0.7,
             max_tokens: Optional[int] = None,
             json_schema: Optional[Dict[str, Any]] = None,
             messages: Optional[List[Message]] = None) -> str:
        ...


class OllamaClient:
    """Minimal Ollama client with structured output and embeddings."""

    def __init__(self,
                 model: str = "deepseek-v3.1:671b-cloud",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self,
             prompt: Optional[str] = None,
             system: str = "",
             temperature: float = 0.7,
             max_tokens: Optional[int] = None,
             json_schema: Optional[Dict[str, Any]] = None,
             messages: Optional[List[Message]] = None) -> str:
        """Send a chat request.

        Either ``prompt`` (a single user turn) or ``messages`` (a full
        history) must be given. ``json_schema`` turns on Ollama structured
        output so the reply is guaranteed to be JSON of that shape.
        """
        msgs: List[Message] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if messages:
            msgs.extend(messages)
        if prompt is not None:
            msgs.append({"role": "user", "content": prompt})
        if len(msgs) == (1 if system else 0):
            raise ValueError("chat() needs a prompt or messages")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if json_schema is not None:
            payload["format"] = json_schema

        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def embed(self, texts: Sequence[str], model: str = "bge-m3") -> List[List[float]]:
        data = self._post("/api/embed", {"model": model, "input": list(texts)})
        return data.get("embeddings", [])

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            data = requests.get(f"{self.base_url}/api/tags", timeout=5).json()
            return [m["name"] for m in data.get("models", [])]
        except (requests.RequestException, ValueError):
            return []

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url} (try: ollama serve)")
        except (requests.RequestException, ValueError) as e:
            raise RuntimeError(f"Ollama request failed: {e}")


class FakeLLM:
    """Deterministic stand-in for tests.

    ``handler`` receives the final user prompt (and the system prompt) and
    returns the reply. Every call is recorded in ``calls``.
    """

    def __init__(self, handler: Callable[[str, str], Any]):
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    def chat(self, prompt=None, system="", temperature=0.7, max_tokens=None,
             json_schema=None, messages=None) -> str:
        text = prompt if prompt is not None else (messages[-1]["content"] if messages else "")
        self.calls.append({"prompt": text, "system": system, "messages": messages,
                           "json_schema": json_schema})
        reply = self.handler(text, system)
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def parse_json(text: str) -> Optional[Any]:
    """Parse a model reply as JSON, tolerating code fences and chatter."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            return None
    return None
