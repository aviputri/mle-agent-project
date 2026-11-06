"""
Provider selection.
"""

import os

from pydantic_ai.models import Model, infer_model
from pydantic_ai.settings import ModelSettings

PROVIDERS = {
    "groq": ("groq", "openai/gpt-oss-20b"),
    "mistral": ("mistral", "ministral-14b-latest"),
    "openrouter": ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "google": ("google", "gemma-4-26b-a4b-it"),
    "openai": ("openai", "gpt-6-luna"),
}


def model_from_env(provider: str) -> Model | None:
    """None means dry-run (scripted planner). Missing keys fail loudly, never fall back."""

    if provider == "dry-run":
        return None
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be dry-run or one of {', '.join(PROVIDERS)}.")
    if not os.getenv(f"{provider.upper()}_API_KEY", "").strip():
        raise ValueError(f"Set {provider.upper()}_API_KEY for LLM_PROVIDER={provider}.")
    prefix, default = PROVIDERS[provider]
    return infer_model(f"{prefix}:{os.getenv(f'{provider.upper()}_MODEL', default).strip()}")


def model_settings(provider: str) -> ModelSettings:
    settings = ModelSettings(timeout=30, max_tokens=2048, max_retries=0)
    if provider == "openrouter":
        backup = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/free").strip()
        settings["extra_body"] = {"provider": {"require_parameters": True}, **({"models": [backup]} if backup else {})}
    return settings