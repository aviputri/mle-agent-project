import pytest

from src.agent import OperationsAgent
from src.evaluation import aggregate, load_cases, run_cases, stress_scenarios


def test_all_labeled_cases_pass(settings):
    rows = run_cases(settings)
    failed = [(c.case_id, r.status, r.citations) for c, r, s in rows if not s["passed"]]
    assert failed == []
    assert aggregate(rows)["all_checks_passed"] == 1.0


@pytest.mark.parametrize("probe", ["tool_budget_3", "step_budget_3", "unknown_ticket", "offtopic_on_valid_ticket",
                                   "hallucinated_citation", "memory_other_customer"])
def test_stress_probes_fail_safely(settings, probe):
    row = next(s for s in stress_scenarios(settings) if s["scenario"] == probe)
    assert row["passed"], row


def test_followup_uses_memory_and_cites_policy(settings):
    agent = OperationsAgent(settings)
    case = next(c for c in load_cases(settings) if c.case_id == "mfa_followup_memory")
    result = agent.run_case(case)
    assert result.memory_used and result.ticket_id == "T-1002"
    assert result.citations == ["account-access"] and "[account-access]" in result.answer
    assert "Ari" not in result.answer  # no other customer's data


def test_citations_always_appear_in_answer(settings):
    for _, result, _ in run_cases(settings):
        assert all(f"[{c}]" in result.answer for c in result.citations)