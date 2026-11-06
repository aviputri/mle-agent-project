"""
Step 1 - Runtime settings loaded from .env (exported variables win).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    llm_provider: str
    top_k: int
    min_score: float
    max_agent_steps: int  # model requests (live) / planner turns (dry-run)
    max_tool_calls: int  # application tool calls per case

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def knowledge_base_dir(self) -> Path:
        return self.data_dir / "knowledge_base"

    @property
    def operations_dir(self) -> Path:
        return self.data_dir / "operations"

    @property
    def cases_path(self) -> Path:
        return self.data_dir / "evaluation" / "agent_cases.json"

    @property
    def storage_dir(self) -> Path:
        return self.project_root / "storage"


def load_settings(root: Path = PROJECT_ROOT, **overrides) -> Settings:
    load_dotenv(root / ".env", override=False)
    settings = Settings(
        project_root=root,
        llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        top_k=int(os.getenv("TOP_K", "3")),
        min_score=float(os.getenv("MIN_SCORE", "0.18")),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "8")),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "6")),
    )
    settings = Settings(**{**settings.__dict__, **overrides})
    if settings.max_agent_steps < 1 or settings.max_tool_calls < 1:
        raise ValueError("MAX_AGENT_STEPS and MAX_TOOL_CALLS must be positive.")
    if not 0 <= settings.min_score <= 1:
        raise ValueError("MIN_SCORE must be between 0 and 1.")
    return settings