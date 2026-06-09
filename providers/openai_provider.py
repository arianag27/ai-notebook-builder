"""OpenAI provider for text generation and embeddings."""

import os
from pathlib import Path

from providers.base_provider import BaseProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100


def get_api_key() -> str:
    """Read the OpenAI API key from the environment."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def require_api_key() -> str:
    """Return the API key or raise a clear error."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY.\n"
            "Create a .env file in the project root with:\n"
            "  OPENAI_API_KEY=your_key_here"
        )
    return api_key


def create_openai_client():
    """Create an OpenAI client with environment-based auth."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is not installed.\n"
            "Install AI dependencies with:\n"
            "  pip install -r requirements-llm.txt\n"
            "On macOS, use a virtual environment:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements-llm.txt"
        ) from exc

    return OpenAI(api_key=require_api_key())


class OpenAIProvider(BaseProvider):
    """Generate notebook content using the OpenAI Chat Completions API."""

    def __init__(self, model: str = DEFAULT_CHAT_MODEL):
        self.model = model
        self.client = create_openai_client()

    def generate(self, prompt: str) -> str:
        """Send a prompt to OpenAI and return the model response."""
        from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except AuthenticationError as exc:
            raise ValueError(
                "Invalid OPENAI_API_KEY. Check the key in your .env file."
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI rate limit reached. Wait a moment and try again."
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to the OpenAI API: {exc}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(
                f"OpenAI API request failed: {exc}"
            ) from exc


def embed_texts(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL):
    """
    Create embeddings for a list of texts using OpenAI.

    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    import numpy as np
    from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    client = create_openai_client()
    vectors = []

    try:
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(model=model, input=batch)
            batch_vectors = [item.embedding for item in response.data]
            vectors.extend(batch_vectors)
    except AuthenticationError as exc:
        raise ValueError(
            "Invalid OPENAI_API_KEY. Check the key in your .env file."
        ) from exc
    except RateLimitError as exc:
        raise RuntimeError(
            "OpenAI rate limit reached while building embeddings. Try again later."
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to the OpenAI API: {exc}"
        ) from exc
    except APIError as exc:
        raise RuntimeError(
            f"OpenAI embedding request failed: {exc}"
        ) from exc

    embeddings = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms
