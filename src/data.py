"""
Step 3 - Load and validate every fixture BEFORE the agent runs.
"""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import Customer, EvalCase, QueueAvailability, Ticket

SOURCE_ID_RE = re.compile(r"^Source ID:\s*([a-z0-9-]+)\s*$", re.MULTILINE)


class DataValidationError(ValueError):
    """Raised with every problem found, not just the first one."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_source_ids(knowledge_base_dir: Path) -> dict[str, Path]:
    ids: dict[str, Path] = {}
    for path in sorted(knowledge_base_dir.glob("*.md")):
        match = SOURCE_ID_RE.search(path.read_text(encoding="utf-8"))
        if match:
            ids[match.group(1)] = path
    return ids


def validate_data(data_dir: Path) -> dict[str, int]:
    """Return counts if valid, otherwise raise DataValidationError listing all errors."""

    errors: list[str] = []
    ops, kb = data_dir / "operations", data_dir / "knowledge_base"

    # Tickets: required fields present and non-empty (pydantic enforces it).
    tickets: dict[str, Ticket] = {}
    for row in load_json(ops / "tickets.json"):
        try:
            ticket = Ticket.model_validate(row)
        except ValidationError as error:
            errors.append(f"ticket {row.get('ticket_id', '?')}: {error.errors()[0]['msg']}")
            continue
        if ticket.ticket_id in tickets:
            errors.append(f"duplicate ticket id {ticket.ticket_id}")
        tickets[ticket.ticket_id] = ticket

    customers = {
        c.customer_id: c
        for c in (Customer.model_validate(r) for r in load_json(ops / "customers.json"))
    }
    for ticket in tickets.values():
        if ticket.customer_id not in customers:
            errors.append(f"{ticket.ticket_id} references missing customer {ticket.customer_id}")

    queues = [QueueAvailability.model_validate(r) for r in load_json(ops / "availability.json")]

    # Policies: each markdown file must declare a unique Source ID.
    sources = policy_source_ids(kb)
    for path in kb.glob("*.md"):
        if path not in sources.values():
            errors.append(f"{path.name} has no 'Source ID:' line")

    # Evaluation cases: unique IDs, valid ticket refs, existing source IDs.
    seen: set[str] = set()
    cases = [EvalCase.model_validate(r) for r in load_json(data_dir / "evaluation" / "agent_cases.json")]
    for case in cases:
        if case.case_id in seen:
            errors.append(f"duplicate case id {case.case_id}")
        seen.add(case.case_id)
        if case.ticket_id is not None and case.ticket_id not in tickets:
            errors.append(f"{case.case_id} references missing ticket {case.ticket_id}")
        for source in case.expected_sources:
            if source not in sources:
                errors.append(f"{case.case_id} expects unknown source {source}")

    # Seed memory must only talk about its own customer/ticket.
    errors.extend(memory_leak_errors(load_json(ops / "seed_memory.json"), tickets, customers))

    if errors:
        raise DataValidationError("\n".join(errors))
    return {
        "tickets": len(tickets),
        "customers": len(customers),
        "queues": len(queues),
        "policies": len(sources),
        "cases": len(cases),
    }


def memory_leak_errors(
    memory: dict, tickets: dict[str, Ticket], customers: dict[str, Customer]
) -> list[str]:
    errors: list[str] = []
    owner = memory.get("last_customer_id")
    last_ticket = memory.get("last_ticket_id")
    if owner not in customers:
        return [f"seed memory customer {owner} does not exist"]
    if last_ticket and (last_ticket not in tickets or tickets[last_ticket].customer_id != owner):
        errors.append(f"seed memory ticket {last_ticket} does not belong to {owner}")
    for note in memory.get("notes", []):
        lowered = note.lower()
        for customer in customers.values():
            mentioned = (
                customer.name.lower() in lowered
                or customer.name.split()[0].lower() in re.findall(r"[a-z]+", lowered)
                or customer.email.lower() in lowered
                or customer.customer_id.lower() in lowered
            )
            if mentioned and customer.customer_id != owner:
                errors.append(f"seed memory note leaks {customer.customer_id}: {note!r}")
        for ticket_id in re.findall(r"T-\d+", note):
            if ticket_id in tickets and tickets[ticket_id].customer_id != owner:
                errors.append(f"seed memory note leaks ticket {ticket_id}: {note!r}")
    return errors