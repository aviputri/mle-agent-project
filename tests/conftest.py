import pytest

from src.config import load_settings


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Tests never call an API, whatever is in .env."""
    monkeypatch.setenv("LLM_PROVIDER", "dry-run")
    monkeypatch.setenv("MIN_SCORE", "0.225")


@pytest.fixture
def settings():
    return load_settings(llm_provider="dry-run", min_score=0.225, top_k=3, max_agent_steps=8, max_tool_calls=6)