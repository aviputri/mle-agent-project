import pytest

from src.retrieval import PolicyRetriever, load_chunks
from src.tools import ToolBudgetExceeded, ToolError, ToolRegistry


@pytest.fixture
def tools(settings):
    r = PolicyRetriever(load_chunks(settings.knowledge_base_dir), 3, 0.25)
    return ToolRegistry(settings.operations_dir, r, max_tool_calls=3)


def test_calls_are_traced_and_budgeted(tools):
    tools.get_ticket("T-1001")
    tools.get_customer_context("C-001")
    with pytest.raises(ToolError):
        tools.get_ticket("T-9999")
    assert [(c.name, c.ok) for c in tools.calls] == [("get_ticket", True), ("get_customer_context", True), ("get_ticket", False)]
    with pytest.raises(ToolBudgetExceeded):
        tools.search_policy("refund")


def test_drafts_require_retrieved_citations(tools):
    with pytest.raises(ToolError, match="Citations"):
        tools.draft_ticket_update("T-1001", ["x"], ["refunds"])  # nothing retrieved yet
    tools.search_policy("duplicate payment refund")
    draft = tools.draft_ticket_update("T-1001", ["Ask for the invoice number [refunds]"], ["refunds"])
    assert draft.sent is False and "[refunds]" in draft.text


def test_escalation_rejected_when_owner_unavailable(tools):
    tools.search_policy("MFA account owner identity review")
    with pytest.raises(ToolError, match="No available owner"):
        tools.draft_escalation("T-1007", "identity-review", "unverified", [], ["account-access"])