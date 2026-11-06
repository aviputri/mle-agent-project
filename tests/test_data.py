from src.data import load_json, memory_leak_errors, validate_data
from src.models import Customer, Ticket


def test_fixtures_are_valid(settings):
    counts = validate_data(settings.data_dir)
    assert counts == {"tickets": 7, "customers": 7, "queues": 4, "policies": 6, "cases": 7}


def test_memory_leak_is_detected(settings):
    ops = settings.operations_dir
    tickets = {t["ticket_id"]: Ticket.model_validate(t) for t in load_json(ops / "tickets.json")}
    customers = {c["customer_id"]: Customer.model_validate(c) for c in load_json(ops / "customers.json")}
    seed = load_json(ops / "seed_memory.json")
    assert memory_leak_errors(seed, tickets, customers) == []
    seed["notes"].append("Ari Patel also asked about T-1001.")
    errors = memory_leak_errors(seed, tickets, customers)
    assert any("C-001" in e for e in errors) and any("T-1001" in e for e in errors)