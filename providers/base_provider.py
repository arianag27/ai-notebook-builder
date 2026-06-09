"""Common interface for LLM providers."""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Base class for text-generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Args:
            prompt: Full prompt string to send to the model.

        Returns:
            Generated text from the model.
        """
        raise NotImplementedError
