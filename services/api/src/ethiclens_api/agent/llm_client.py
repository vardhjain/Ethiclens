"""Multi-provider LLM client with structured (JSON) output and automatic fallback.

Design constraints from the build plan:

- The LLM is an interpretation layer only. It never computes metrics or thresholds.
- Every call is schema-validated (Pydantic); malformed JSON is retried once, then
  raises so the caller can degrade gracefully instead of trusting a bad response.
- Groq (Llama 3.3 70B) is primary since it is free-tier and fast; Gemini Flash is the
  fallback. Both SDKs are imported lazily so the rest of the agent package (and its
  tests) work without either dependency installed — tests inject a stub ``LLMClient``.
"""

from __future__ import annotations

import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMCallError(RuntimeError):
    """Raised when every configured provider failed to produce valid structured output."""


class LLMClient(Protocol):
    """Anything that can turn a prompt into raw text. Implementations are provider-specific."""

    def complete(self, system: str, prompt: str) -> str: ...


def complete_json(client: LLMClient, system: str, prompt: str, schema: type[SchemaT]) -> SchemaT:
    """Call ``client``, parse the response as JSON, and validate it against ``schema``.

    Retries once with a stricter instruction if the first response isn't valid JSON matching
    ``schema`` (the LLM sometimes wraps JSON in prose or code fences). Raises ``LLMCallError``
    if both attempts fail, so the caller can degrade instead of trusting a bad structure.
    """
    schema_prompt = (
        f"{prompt}\n\nRespond with ONLY a single JSON object matching this schema "
        f"(no prose, no markdown fences):\n{json.dumps(schema.model_json_schema())}"
    )
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            raw = client.complete(system, schema_prompt)
            return schema.model_validate_json(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            schema_prompt = (
                f"{schema_prompt}\n\nYour previous response was invalid JSON or did not match "
                f"the schema ({exc}). Return ONLY the corrected JSON object."
            )
    raise LLMCallError(f"LLM did not return valid {schema.__name__} JSON after 2 attempts") from (
        last_error
    )


def _extract_json(raw: str) -> str:
    """Strip markdown code fences some models wrap JSON in."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


class GroqClient:
    """Wraps the Groq SDK. Imported lazily so ``groq`` is only required if actually used."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model

    def complete(self, system: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""


class GeminiClient:
    """Wraps the Gemini SDK. Imported lazily so ``google-genai`` is only required if used."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, system: str, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={"system_instruction": system, "temperature": 0},
        )
        return response.text or ""


class FallbackLLMClient:
    """Tries each client in order; falls through to the next on any exception."""

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackLLMClient requires at least one client")
        self._clients = clients

    def complete(self, system: str, prompt: str) -> str:
        last_error: Exception | None = None
        for client in self._clients:
            try:
                return client.complete(system, prompt)
            except Exception as exc:
                last_error = exc
        raise LLMCallError("All configured LLM providers failed") from last_error


def build_default_client(groq_api_key: str | None, gemini_api_key: str | None) -> LLMClient:
    """Build the Groq-primary / Gemini-fallback client from settings.

    Raises ``ValueError`` if neither key is configured — callers should treat that as
    "narrative/inference unavailable" and degrade rather than let this bubble up.
    """
    clients: list[LLMClient] = []
    if groq_api_key:
        clients.append(GroqClient(groq_api_key))
    if gemini_api_key:
        clients.append(GeminiClient(gemini_api_key))
    if not clients:
        raise ValueError("No LLM provider configured (set GROQ_API_KEY and/or GEMINI_API_KEY)")
    return FallbackLLMClient(clients)


def build_default_client_or_none(
    groq_api_key: str | None, gemini_api_key: str | None
) -> LLMClient | None:
    """Same as :func:`build_default_client` but returns ``None`` instead of raising.

    For callers (e.g. the narrative endpoint) that should degrade gracefully rather
    than fail outright when no LLM provider is configured.
    """
    try:
        return build_default_client(groq_api_key, gemini_api_key)
    except ValueError:
        return None
