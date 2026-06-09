"""Ollama provider for local LLM generation (no API key required)."""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from providers.base_provider import BaseProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = 600


def get_ollama_url() -> str:
    """Read Ollama base URL from the environment."""
    return os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def get_default_model() -> str:
    """Read default Ollama model from the environment."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def strip_thinking_tags(text: str) -> str:
    """Remove Qwen-style thinking blocks from model output."""
    think_pattern = re.compile(
        "<" + "think" + ">" + r".*?" + "<" + "/" + "think" + ">" + r"\s*",
        re.DOTALL | re.IGNORECASE,
    )
    text = think_pattern.sub("", text)
    text = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


def model_is_available(requested: str, available: set[str]) -> bool:
    """Return True if the requested model name matches an installed Ollama model."""
    if requested in available:
        return True
    requested_base = requested.split(":")[0]
    for name in available:
        if name.split(":")[0] == requested_base:
            return True
    return False


class OllamaProvider(BaseProvider):
    """Generate notebook content using a local Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.model = model or get_default_model()
        self.base_url = (base_url or get_ollama_url()).rstrip("/")
        self.timeout = timeout

    def _post_json(self, path: str, payload: dict) -> dict:
        """Send a JSON POST request to the Ollama API."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and "model" in body.lower():
                raise RuntimeError(
                    f"Ollama model '{self.model}' not found.\n"
                    f"Pull it with:\n"
                    f"  ollama pull {self.model}"
                ) from exc
            raise RuntimeError(
                f"Ollama API error ({exc.code}): {body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            if "Connection refused" in reason or "nodename nor servname" in reason:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}.\n"
                    "Install Ollama:\n"
                    "  brew install ollama\n"
                    "Start the server:\n"
                    "  ollama serve\n"
                    "Pull a model:\n"
                    f"  ollama pull {self.model}"
                ) from exc
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "Ollama request timed out. The model may still be loading — try again."
            ) from exc

    def check_ready(self) -> None:
        """Verify Ollama is running and the model is available."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                "Start the server with:\n"
                "  ollama serve"
            ) from exc

        model_names = {model.get("name", "") for model in data.get("models", [])}
        if not model_is_available(self.model, model_names):
            available = ", ".join(sorted(model_names)[:5]) or "(none)"
            raise RuntimeError(
                f"Ollama model '{self.model}' is not available.\n"
                f"Pull it with:\n"
                f"  ollama pull {self.model}\n"
                f"Models currently available: {available}"
            )

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the model response."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7},
        }

        response = self._post_json("/api/chat", payload)
        message = response.get("message", {})
        content = message.get("content", "")

        if not content:
            raise RuntimeError("Ollama returned an empty response.")

        return strip_thinking_tags(content)
