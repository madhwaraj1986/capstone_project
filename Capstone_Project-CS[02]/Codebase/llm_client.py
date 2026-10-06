"""
OpenAI-compatible chat-completions client used by both LLM stages.

The same HTTP interface covers OpenAI, Groq, Together, Azure-compatible
gateways, and local Ollama (`--api-base http://localhost:11434/v1`).
API keys are never hard-coded; they are supplied at runtime.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple


_NON_RETRYABLE_STATUS = {400, 401, 403, 404}


class LLMClient:
    """Thin wrapper around the OpenAI Python SDK chat.completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> None:
        """
        Create a client bound to a single model name.

        Parameters
        ----------
        api_key:
            Secret provided via CLI or environment. Empty values are rejected.
        model:
            Model identifier understood by the chosen provider.
        api_base:
            Optional OpenAI-compatible base URL. When omitted the official
            OpenAI endpoint (or OPENAI_BASE_URL) is used.
        timeout:
            Per-request timeout in seconds.
        max_retries:
            Number of attempts for transient HTTP/JSON failures.
        temperature:
            Sampling temperature. Extraction uses 0.0; ranking may be slightly higher.
        """
        if not api_key or not api_key.strip():
            raise ValueError(
                "An API key is required. Pass --api-key or set OPENAI_API_KEY / GROQ_API_KEY."
            )
        if not model or not model.strip():
            raise ValueError("A model name is required.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install it with: pip install openai"
            ) from exc

        kwargs: Dict[str, Any] = {"api_key": api_key.strip(), "timeout": timeout}
        if api_base:
            kwargs["base_url"] = api_base.rstrip("/")

        self._client = OpenAI(**kwargs)
        self.model = model.strip()
        self.max_retries = max_retries
        self.temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
        """
        Run a single chat completion and return the assistant text.

        json_mode requests structured JSON when the provider supports it.
        Providers that reject response_format are automatically retried without it.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Optional[Exception] = None
        use_json_mode = json_mode

        for attempt in range(1, self.max_retries + 1):
            try:
                request: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if use_json_mode:
                    request["response_format"] = {"type": "json_object"}

                response = self._client.chat.completions.create(**request)
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise RuntimeError("LLM returned an empty response.")
                return content
            except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                message = str(exc).lower()
                # Some local/open-source servers do not implement json_object mode.
                if use_json_mode and "response_format" in message:
                    use_json_mode = False
                    last_error = exc
                    continue
                last_error = exc
                # Bad key, unknown model, or malformed request will never succeed on retry.
                status = getattr(exc, "status_code", None)
                if status in _NON_RETRYABLE_STATUS:
                    hint = ""
                    if "model_not_found" in message or "does not have access" in message:
                        hint = (
                            "\nHint: this key's project cannot use this model. Enable it under "
                            "Project settings > Limits > Model usage for the SAME project the "
                            "key was created in, or pass --extractor-model/--ranker-model "
                            "with models that key can access."
                        )
                    raise RuntimeError(
                        f"LLM call failed for model '{self.model}' (HTTP {status}): {exc}{hint}"
                    ) from exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(
                    f"LLM call failed for model '{self.model}' after {self.max_retries} attempts: {exc}"
                ) from last_error

        raise RuntimeError(f"LLM call failed for model '{self.model}': {last_error}")

    def complete_json(
        self, system_prompt: str, user_prompt: str
    ) -> Dict[str, Any]:
        """
        Run a completion and parse the result as a JSON object.

        The parser is tolerant of markdown fences and leading/trailing prose,
        which is common with smaller open-source models.
        """
        raw = self.complete(system_prompt, user_prompt, json_mode=True)
        try:
            return parse_json_object(raw)
        except ValueError:
            # One repair pass: ask the same model to return valid JSON only.
            repair_prompt = (
                "Convert the following text into a single valid JSON object. "
                "Do not add commentary.\n\n"
                f"{raw}"
            )
            repaired = self.complete(system_prompt, repair_prompt, json_mode=True)
            return parse_json_object(repaired)


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Extract and deserialize the first JSON object found in `text`.

    Raises ValueError when no object can be parsed.
    """
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1)

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(candidate[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse a JSON object from the LLM response.")


def resolve_api_key_with_source(cli_key: Optional[str]) -> Tuple[str, str]:
    """
    Resolve the API key and report where it came from (never the key itself).

    Order: --api-key, OPENAI_API_KEY, GROQ_API_KEY, LLM_API_KEY. Reporting the
    source matters because a stale environment key silently wins over nothing,
    and keys from different OpenAI projects have different model access.
    """
    if cli_key and cli_key.strip():
        return cli_key.strip(), "--api-key"
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "LLM_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    return "", ""


def resolve_api_key(cli_key: Optional[str]) -> str:
    """Resolve the API key from CLI first, then from common environment variables."""
    return resolve_api_key_with_source(cli_key)[0]


def mask_key(key: str) -> str:
    """Show only the last 4 characters so users can tell keys apart safely."""
    return f"...{key[-4:]}" if len(key) > 8 else "(short key)"
