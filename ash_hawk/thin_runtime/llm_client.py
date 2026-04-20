from __future__ import annotations

import asyncio
import importlib
import json
import os
from typing import TypeVar, cast

import pydantic as pd

T = TypeVar("T", bound=pd.BaseModel)


def call_model_text(system_prompt: str, user_prompt: str) -> str | None:
    try:
        config_module = importlib.import_module("dawn_kestrel.base.config")
        client_module = importlib.import_module("dawn_kestrel.provider.llm_client")
    except ImportError:
        return None

    try:
        dk_config = config_module.load_agent_config()
        provider = (
            dk_config.get("runtime.provider")
            or os.environ.get("DAWN_KESTREL_PROVIDER")
            or "anthropic"
        )
        model = (
            dk_config.get("runtime.model")
            or os.environ.get("DAWN_KESTREL_MODEL")
            or "claude-sonnet-4-20250514"
        )
        api_key = config_module.get_config_api_key(provider) or None

        llm_client_type = getattr(client_module, "LLMClient")
        client = llm_client_type(provider_id=provider, model=model, api_key=api_key)
        result = asyncio.run(
            client.chat_completion(system_prompt=system_prompt, user_message=user_prompt)
        )
        return cast(str | None, result)
    except Exception:
        return None


def call_model_structured(model_type: type[T], system_prompt: str, user_prompt: str) -> T | None:
    response = call_model_text(system_prompt, user_prompt)
    if not isinstance(response, str) or not response.strip():
        return None
    extracted = _extract_json_object(response)
    if extracted is None:
        return None
    try:
        return model_type.model_validate(extracted)
    except pd.ValidationError:
        return None


def _extract_json_object(text: str) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    candidates.append(parsed)
    return candidates[-1] if candidates else None
