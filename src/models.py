"""
Step 2 - Typed contracts for data fixtures, tool inputs/outputs and results.
"""

from typing import Literal
import re

from pydantic import BaseModel, Field, field_validator, model_validator

Status = Literal["answered", "escalated", "abstained", "review_required"]


# ---- Fixtures -------------------------------------------------------------
class Ticket(BaseModel):
    ticket_id: str = Field(pattern=r"^T-\d+$")
    customer_id: str = Field(pattern=r"^C-\d+$")
    category: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class Customer(BaseModel):
    customer_id: str = Field(pattern=r"^C-\d+$")
    name: str
    email: str
    plan: str
    verified: bool


class QueueAvailability(BaseModel):
    queue: str
    owner: str
    channel: str
    available: bool


class EvalCase(BaseModel):
    case_id: str
    ticket_id: str | None
    prompt: str
    expected_status: Status
    expected_tools: list[str]
    expected_sources: list[str]
    must_use_memory: bool
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_safety_events: list[str] = Field(default_factory=list)


# ---- Retrieval ------------------------------------------------------------
class Chunk(BaseModel):
    chunk_id: str  # e.g. "refunds-1"
    source_id: str  # stable ID from "Source ID:" line, e.g. "refunds"
    title: str
    text: str


class Evidence(BaseModel):
    """What search_policy returns to the planner (no hidden fields)."""

    source_id: str
    chunk_id: str
    score: float
    shared_terms: list[str] = Field(default_factory=list)
    text: str


class PolicySearchResult(BaseModel):
    query: str
    evidence: list[Evidence]
    removed_chunks: list[str] = Field(default_factory=list)
    supported: bool


# ---- Draft tools ----------------------------------------------------------
class TicketUpdateDraft(BaseModel):
    ticket_id: str
    customer_name: str
    next_steps: list[str]
    citations: list[str]
    text: str
    sent: bool = False  # drafts are never sent


class EscalationDraft(BaseModel):
    ticket_id: str
    queue: str
    owner: str
    channel: str
    reason: str
    evidence: list[str]
    citations: list[str]
    text: str
    sent: bool = False


# ---- Trace + result -------------------------------------------------------
class ToolCall(BaseModel):
    name: str
    args: dict
    ok: bool = True
    error: str | None = None


class AgentDecision(BaseModel):
    """Structured output the planner (scripted or LLM) must return."""

    status: Status
    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def normalize_citations(cls, value):
        """Accept '[refunds]', ' refunds ' or 'refunds, privacy' - format only; IDs are still checked."""

        if isinstance(value, str):
            value = value.split(",")
        cleaned = [str(v).strip().strip("[]").strip() for v in value or []]
        return list(dict.fromkeys(c for c in cleaned if c))
    
    @model_validator(mode="after")
    def citations_from_answer(self):
        """Models often cite inline ('[refunds]') but leave the list empty: copy them over.
        The guard still rejects any ID that was not retrieved."""

        if not self.citations:
            self.citations = list(dict.fromkeys(re.findall(r"\[([a-z0-9-]+)\]", self.answer)))
        return self
# class AgentDecision(BaseModel):
#     """Structured output the planner (scripted or LLM) must return."""

#     status: Status
#     answer: str = Field(min_length=1)
#     citations: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    case_id: str
    ticket_id: str | None
    status: Status
    answer: str
    citations: list[str]
    tool_calls: list[ToolCall]
    memory_used: bool
    safety_events: list[str] = Field(default_factory=list)
    model_requests: int = 0

    @property
    def tools_used(self) -> list[str]:
        return [call.name for call in self.tool_calls if call.ok]