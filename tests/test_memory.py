from src.memory import CaseMemoryStore


def test_notes_are_scoped_to_customer_and_persist(tmp_path):
    path = tmp_path / "case_memory.json"
    store = CaseMemoryStore(path)
    store.remember("T-1002", "C-002", note="verified requester")
    assert store.notes_for("T-1002", "C-002") == ["verified requester"]
    assert store.notes_for("T-1002", "C-001") == []  # no cross-customer leak
    assert CaseMemoryStore(path).notes_for("T-1002", "C-002") == ["verified requester"]  # durable