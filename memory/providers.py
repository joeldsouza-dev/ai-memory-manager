"""
memory/providers.py
-------------------
Provider abstraction for Large Language Models.

Enables swappable LLM backends (OpenAI, Anthropic, Gemini, local/Ollama, Mock)
without coupling extraction logic to vendor SDKs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

from memory.models import RetryableError, TerminalError


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for all LLM provider errors."""
    pass


class LLMAPIError(RetryableError, LLMError):
    """Network failure, timeout, rate limit, or service 5xx error.
    
    Subclasses RetryableError so background job processing automatically
    applies exponential backoff retries.
    """
    pass


class LLMAuthError(TerminalError, LLMError):
    """Missing or invalid API credentials."""
    pass


class LLMResponseError(TerminalError, LLMError):
    """Provider returned an unparseable or structurally invalid response."""
    pass


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Interface for LLM text and structured generation providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Send prompt to the LLM and return the generated raw string.

        Parameters
        ----------
        prompt : str
            The user prompt containing message and context.
        system_prompt : Optional[str]
            Instruction guiding task, taxonomy, and output format.
        temperature : float
            Sampling temperature (0.0 for deterministic extraction).

        Returns
        -------
        str
            Raw output from the model (typically a JSON string).

        Raises
        ------
        LLMAPIError
            For transient network, timeout, or server errors.
        LLMAuthError
            For authentication / authorization failures.
        LLMError
            For general provider errors.
        """
        pass


# ---------------------------------------------------------------------------
# Mock Provider (for deterministic testing & offline runs)
# ---------------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """Mock provider with canned responses, response queues, and error simulation."""

    def __init__(
        self,
        responses: Optional[str | Sequence[str | Exception]] = None,
        callback: Optional[Callable[[str, Optional[str]], str]] = None,
    ):
        if responses is None:
            self._responses: list[str | Exception] = []
        elif isinstance(responses, str):
            self._responses = [responses]
        else:
            self._responses = list(responses)

        self._callback = callback
        self.calls: list[dict] = []

    def queue_response(self, response: str | Exception) -> None:
        """Queue an additional response or exception to be yielded."""
        self._responses.append(response)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )

        if self._callback is not None:
            return self._callback(prompt, system_prompt)

        if not self._responses:
            raise LLMResponseError("MockLLMProvider has no responses left in queue.")

        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------------------
# OpenAI-Compatible Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI and compatible HTTP API endpoints (Ollama, vLLM, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        if not self.api_key:
            raise LLMAuthError(
                "OPENAI_API_KEY is not set. Configure it via environment variable or provider constructor."
            )

        endpoint = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            endpoint, data=data, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_body = resp.read().decode("utf-8")
                parsed = json.loads(res_body)
                choices = parsed.get("choices", [])
                if not choices:
                    raise LLMResponseError("Provider response contained no choices.")
                return choices[0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise LLMAuthError(f"LLM Authentication failed (HTTP {exc.code}): {exc.reason}") from exc
            if exc.code in (429, 500, 502, 503, 504):
                raise LLMAPIError(f"LLM transient API error (HTTP {exc.code}): {exc.reason}") from exc
            raise LLMError(f"LLM HTTP error {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMAPIError(f"Network error communicating with LLM provider: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Malformed JSON envelope from LLM provider: {exc}") from exc

