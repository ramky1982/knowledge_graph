from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List
from urllib import error as urlerror
from urllib import request as urlrequest


@dataclass
class OllamaClient:
    """Small client for optional Ollama enrichment."""

    base_url: str = "http://localhost:11434"
    model: str = "mistral"

    def generate(self, prompt: str) -> str:
        """Send a single-turn prompt to Ollama and return the generated text response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
        }
        req = urlrequest.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw).get("response", "")

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            req = urlrequest.Request(url=f"{self.base_url}/api/tags", method="GET")
            with urlrequest.urlopen(req, timeout=5):
                return True
        except (urlerror.URLError, TimeoutError, ValueError):
            return False

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat with Ollama using message history."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        req = urlrequest.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
        message = json.loads(raw).get("message", {})
        return message.get("content", "")
