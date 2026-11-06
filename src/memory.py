"""
Step 6 - Short-term conversation memory + durable case memory.

* ConversationMemory: lives for one conversation (last ticket/customer, turns,
  PydanticAI message history for live mode).
* CaseMemoryStore: JSON file in storage/ that survives processes. Notes are
  keyed by ticket AND customer, and are only ever returned for the customer of
  the ticket being handled -> no cross-customer leakage.

Memory may resolve *references* ("the same customer"); it never replaces the
policy search - the agent still retrieves evidence on every turn.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage

from src.data import load_json


class CaseRecord(BaseModel):
    ticket_id: str
    customer_id: str
    notes: list[str] = Field(default_factory=list)
    last_status: str | None = None
    citations: list[str] = Field(default_factory=list)
    updated_at: str = ""


class CaseMemoryStore:
    def __init__(self, path: Path | None) -> None:
        self.path = path  # None = in-memory only (used in evaluation)
        self.records: dict[str, CaseRecord] = {}
        if path and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.records = {k: CaseRecord.model_validate(v) for k, v in raw.items()}

    def notes_for(self, ticket_id: str, customer_id: str) -> list[str]:
        record = self.records.get(ticket_id)
        if record is None or record.customer_id != customer_id:
            return []
        return list(record.notes)

    def remember(self, ticket_id: str, customer_id: str, *, note: str | None = None,
                 status: str | None = None, citations: list[str] | None = None) -> None:
        record = self.records.setdefault(ticket_id, CaseRecord(ticket_id=ticket_id, customer_id=customer_id))
        if record.customer_id != customer_id:
            raise ValueError(f"{ticket_id} belongs to {record.customer_id}, not {customer_id}")
        if note and note not in record.notes:
            record.notes.append(note)
        record.last_status = status or record.last_status
        record.citations = citations if citations is not None else record.citations
        record.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.save()

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.model_dump() for k, v in self.records.items()}
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class ConversationMemory:
    conversation_id: str
    last_ticket_id: str | None = None
    last_customer_id: str | None = None
    turns: list[dict] = field(default_factory=list)
    message_history: list[ModelMessage] = field(default_factory=list)

    def remember_turn(self, request: str, ticket_id: str, customer_id: str, status: str) -> None:
        if customer_id != self.last_customer_id:
            self.message_history = []  # never carry one customer's chat into another's
        self.last_ticket_id, self.last_customer_id = ticket_id, customer_id
        self.turns.append({"request": request, "ticket_id": ticket_id, "status": status})


def load_seed(operations_dir: Path, store: CaseMemoryStore) -> ConversationMemory:
    """Seed short-term state and durable notes from data/operations/seed_memory.json."""

    seed = load_json(operations_dir / "seed_memory.json")
    memory = ConversationMemory(seed["conversation_id"], seed["last_ticket_id"], seed["last_customer_id"])
    for note in seed["notes"]:
        if seed["last_ticket_id"] not in store.records or note not in store.records[seed["last_ticket_id"]].notes:
            store.remember(seed["last_ticket_id"], seed["last_customer_id"], note=note)
    return memory