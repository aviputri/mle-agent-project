"""
Step 7b - Bounded agent run + application-side guard.

The planner only *proposes* (status, answer, citations). The guard re-checks
the proposal against the observed tool trace and evidence, so a wrong or
manipulated model cannot turn into an unsupported 'answered'/'escalated'.
"""

import asyncio
import json
import re
#import time
from pathlib import Path

from pydantic_ai.exceptions import ModelAPIError, ModelRateLimitError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart

from src.config import Settings
from src.data import validate_data
from src.memory import CaseMemoryStore, ConversationMemory, load_seed
from src.models import AgentDecision, AgentResult, EvalCase
from src.planner import build_planner, required_queue
from src.provider import model_from_env, model_settings
from src.retrieval import PolicyRetriever, is_unsafe, load_chunks
from src.tools import ToolBudgetExceeded, ToolRegistry
from src.rate_limit import RateLimitRetryModel

CITATION_RE = re.compile(r"\[([a-z0-9-]+)\]")
# Request-vs-context gate; lower than MIN_SCORE because prompts are short (see report).
REQUEST_MIN_SCORE = 0.10

# def _retry_after_seconds(error: Exception, default: float) -> float:
#     """Use the provider's hint ('try again in 1.33s'), plus a small safety margin."""

#     hint = getattr(error, "retry_after", None)
#     if hint is None:
#         match = re.search(r"try again in ([\d.]+)s", str(error))
#         hint = float(match.group(1)) if match else None
#     return min((hint or default) + 1.0, 60.0)

class OperationsAgent:
    def __init__(self, settings: Settings, *, model=None, memory_path: Path | None = None) -> None:
        validate_data(settings.data_dir)  # refuse to run on broken fixtures
        self.settings = settings
        #self.model = model if model is not None else model_from_env(settings.llm_provider)
        self.model = model if model is not None else model_from_env(settings.llm_provider)
        if self.model is not None and settings.llm_provider != "dry-run":
            self.model = RateLimitRetryModel(self.model)  # retry single requests on 429
        self.retriever = PolicyRetriever(load_chunks(settings.knowledge_base_dir), settings.top_k, settings.min_score)
        self.tools = ToolRegistry(settings.operations_dir, self.retriever, settings.max_tool_calls)
        self.planner = build_planner(
            self.tools, self.model, None if self.model is None else model_settings(settings.llm_provider)
        )
        self.store = CaseMemoryStore(memory_path)
        self.memory: ConversationMemory = load_seed(settings.operations_dir, self.store)

    # ------------------------------------------------------------------
    def run_case(self, case: EvalCase) -> AgentResult:
        return self.run(case.prompt, case.ticket_id, case_id=case.case_id)

    def run(self, request: str, ticket_id: str | None = None, *, case_id: str = "ad-hoc") -> AgentResult:
        self.tools.reset()
        events: list[str] = []
        memory_used = ticket_id is None

        # --- resolve the ticket (memory only fills a missing reference) ---
        if ticket_id is None:
            ticket_id = self.memory.last_ticket_id
            if ticket_id is None:
                return self._result(case_id, None, "review_required", "No ticket in request or memory.", [], True, ["no_ticket_reference"])
        ticket = self.tools.tickets.get(ticket_id)
        if ticket and memory_used and self._mentions_other_customer(request, ticket.customer_id):
            return self._result(case_id, ticket_id, "review_required",
                                "The request names a different customer than conversation memory.",
                                [], True, ["memory_customer_mismatch"])
        same_customer = ticket is not None and ticket.customer_id == self.memory.last_customer_id
        notes = self.store.notes_for(ticket_id, ticket.customer_id) if ticket else []
        if ticket and is_unsafe(ticket.body):
            events.append("ticket_instruction_attack")

        payload = {"request": request, "ticket_id": ticket_id, "memory_notes": notes}
        history = self.memory.message_history if same_customer else []
        with capture_run_messages() as run_messages:  # kept even if the run raises
            try:
                result = asyncio.run(self._run_planner(json.dumps(payload), history))
            except Exception as error:  # noqa: BLE001 - every failure becomes a visible handoff
                return self._failed_run(error, run_messages, case_id, ticket_id, memory_used, events)
        """
        try:
            #result = self._run_with_rate_limit_retry(json.dumps(payload), history)
            result = asyncio.run(self._run_planner(json.dumps(payload), history))
            # result = self.planner.run_sync(
            #     json.dumps(payload),
            #     message_history=history,
            #     usage_limits=UsageLimits(request_limit=self.settings.max_agent_steps,
            #                              tool_calls_limit=self.settings.max_tool_calls),
            # )
        except (ToolBudgetExceeded, UsageLimitExceeded) as error:
            return self._result(case_id, ticket_id, "review_required", f"Execution budget exhausted: {error}",
                                [], memory_used, [*events, "execution_budget_exhausted"])
        except UnexpectedModelBehavior as error:
            return self._result(case_id, ticket_id, "review_required", f"Planner failed: {error}",
                                [], memory_used, [*events, "planner_error"])
        except ModelRateLimitError as error:
            return self._result(case_id, ticket_id, "review_required", f"Provider rate limit: {error}",
                                [], memory_used, [*events, "provider_rate_limited"])
        except ModelAPIError as error:  # auth, timeout, 5xx: visible, never a fake success
            return self._result(case_id, ticket_id, "review_required", f"Provider error: {error}",
                                [], memory_used, [*events, "provider_error"])
        """

        events += [f"removed_unsafe_chunk:{c}" for s in self.tools.searches for c in s.removed_chunks]
        status, answer, citations, guard_events = self.guard(request, ticket_id, result.output)
        out = self._result(case_id, ticket_id, status, answer, citations, memory_used, [*events, *guard_events],
                           requests=result.usage.requests)

        # --- update memory only after a run that inspected the real ticket ---
        if ticket and any(c.name == "get_ticket" and c.ok and c.args["ticket_id"] == ticket_id for c in self.tools.calls):
            self.memory.remember_turn(request, ticket_id, ticket.customer_id, status)
            self.memory.message_history = result.all_messages()
            self.store.remember(ticket_id, ticket.customer_id, status=status, citations=citations,
                                note=f"Last outcome: {status} citing {', '.join(citations) or 'no sources'}.")
        return out

    def _run_with_rate_limit_retry(self, prompt: str, history, attempts: int = 3):
        """Retry the WHOLE run on 429 (tokens-per-minute) with a fresh trace each time."""

        for attempt in range(1, attempts + 1):
            self.tools.reset()  # a retried run must not keep tool calls from the failed attempt
            try:
                return asyncio.run(self._run_planner(prompt, history))
            except ModelRateLimitError as error:
                if attempt == attempts:
                    raise
                wait = _retry_after_seconds(error, default=5.0 * attempt)
                print(f"  rate limited, waiting {wait:.1f}s (attempt {attempt}/{attempts})")
                time.sleep(wait)

    def _failed_run(self, error, messages, case_id, ticket_id, memory_used, events) -> AgentResult:
        """Map a crashed run to review_required AND say why (last validator/tool feedback)."""

        if isinstance(error, (ToolBudgetExceeded, UsageLimitExceeded)):
            event = "execution_budget_exhausted"
        elif isinstance(error, ModelRateLimitError):
            event = "provider_rate_limited"
        elif isinstance(error, ModelAPIError):
            event = "provider_error"
        elif isinstance(error, UnexpectedModelBehavior):
            event = "planner_error"
        else:
            raise error  # a real bug in our code: fail loudly
        feedback = [str(p.content if isinstance(p, ToolReturnPart) else p.model_response())[:200]
                    for m in messages for p in m.parts
                    if isinstance(p, RetryPromptPart) or (isinstance(p, ToolReturnPart) and p.outcome != "success")]
        detail = f" Last feedback to the model: {feedback[-1]}" if feedback else ""
        finals = [p for m in messages if isinstance(m, ModelResponse) for p in m.parts
                  if isinstance(p, ToolCallPart) and p.tool_name == "final_result"]
        if finals:  # what the planner proposed when it ran out of corrections
            last = finals[-1].args_as_dict()
            detail += f" Last proposal: status={last.get('status')}, citations={last.get('citations')}."
        requests = sum(isinstance(m, ModelResponse) for m in messages)
        return self._result(case_id, ticket_id, "review_required", f"{event}: {error}.{detail}", [],
                            memory_used, [*events, event], requests=requests)
    
    async def _run_planner(self, prompt: str, history):
        """Own the event loop explicitly (run_sync relies on a deprecated implicit loop)."""

        async with self.planner:  # opens/closes the provider HTTP client on this loop
            return await self.planner.run(
                prompt,
                message_history=history,
                usage_limits=UsageLimits(request_limit=self.settings.max_agent_steps,
                                         tool_calls_limit=self.settings.max_tool_calls),
            )

    # ------------------------------------------------------------------
    def guard(self, request: str, ticket_id: str, decision: AgentDecision) -> tuple[str, str, list[str], list[str]]:
        calls = self.tools.calls
        ok = lambda name, **match: any(  # noqa: E731
            c.name == name and c.ok and all(c.args.get(k) == v for k, v in match.items()) for c in calls)
        accepted = self.tools.accepted_sources
        if ticket_id not in self.tools.tickets:
            return "review_required", f"Ticket {ticket_id} could not be found; human review required.", [], ["missing_context"]
        ticket = self.tools.tickets[ticket_id]
        customer = self.tools.customers[ticket.customer_id]
        # A handoff carries the evidence it was based on, even if the planner forgot to list it.
        review = lambda msg, event, cites=(): ("review_required", msg,  # noqa: E731
                                               [c for c in (cites or sorted(accepted)) if c in accepted] or list(cites),
                                               [event])
        #review = lambda msg, event, cites=(): ("review_required", msg, list(cites), [event])  # noqa: E731

        # 1. citations must come from accepted evidence (both list and inline)
        inline = set(CITATION_RE.findall(decision.answer))
        if not set(decision.citations) <= accepted or not inline <= accepted:
            bad = sorted((set(decision.citations) | inline) - accepted)
            return review(f"Rejected citations not in retrieved evidence: {bad}.", "invalid_citations")

        # 2. required inspection steps for THIS ticket/customer
        if not (ok("get_ticket", ticket_id=ticket_id) and ok("get_customer_context", customer_id=ticket.customer_id)
                and ok("search_policy")):
            return review("Required ticket/customer/policy steps missing.", "incomplete_tool_trace")
        
        # 3a. the planner's own query found nothing: re-check with the ticket text before trusting
        #     the abstention (an LLM's short query can miss policy that does exist)
        if not accepted:
            check = self.retriever.search(f"{request} {ticket.subject} {ticket.body}")
            if check.evidence:
                found = sorted({e.source_id for e in check.evidence})
                return review(f"Planner found no evidence, but a ticket-based search found {found}; "
                              "human review required.", "abstention_overridden", found)
        
        # 3b. no evidence or request unrelated to ticket and evidence -> abstain
        if not accepted or not self._request_supported(request, ticket):
            return "abstained", ("No approved policy supports this request; abstaining." if decision.status != "abstained"
                                 else decision.answer), [], (["request_not_supported"] if accepted else [])
        if decision.status == "abstained":
            # Evidence exists, so abstaining is the planner giving up, not a safe no-answer.
            return review("Planner abstained although approved policy evidence was retrieved.",
                          "unjustified_abstention", sorted(accepted))
        # if decision.status == "abstained":
        #     return "abstained", decision.answer, [], []

        # 4. handoff rules (computed from fixtures, not from the model)
        queue = required_queue(ticket.category, customer.verified)
        if queue:
            owner = self.tools.queues.get(queue)
            if not ok("check_availability", queue=queue):
                return review(f"{queue} availability was not checked.", "availability_not_checked", decision.citations)
            if owner is None or not owner.available:
                return review(f"No available owner for {queue}; human review required. "
                              + " ".join(f"[{c}]" for c in decision.citations), "owner_unavailable", decision.citations)
            if not ok("draft_escalation", ticket_id=ticket_id, queue=queue) or decision.status != "escalated":
                return review(f"Ticket needs a {queue} handoff draft.", "missing_handoff", decision.citations)
            if ticket.category in {"incident", "privacy"} and "escalation-rules" not in decision.citations:
                return review("Escalation must cite [escalation-rules].", "missing_escalation_evidence", decision.citations)
            return "escalated", decision.answer, decision.citations, []

        missing = [why for why, failed in [
            (f"status was {decision.status}, not answered", decision.status != "answered"),
            (f"no successful draft_ticket_update for {ticket_id}", not ok("draft_ticket_update", ticket_id=ticket_id)),
            ("no citations", not decision.citations)] if failed]
        if missing:
            return review(f"Proposal lacks a cited ticket-update draft: {'; '.join(missing)}.",
                          "missing_update_draft", decision.citations)
        return "answered", decision.answer, decision.citations, []

    def _request_supported(self, request: str, ticket) -> bool:
        """The request itself must relate to the ticket or the evidence.
        Stops an off-topic ask riding on a valid ticket (e.g. cafeteria question on T-1001)."""

        texts = [f"{ticket.category} {ticket.subject} {ticket.body}",
                 *(e.text for s in self.tools.searches for e in s.evidence)]
        return any(self.retriever.similarity(request, t) >= REQUEST_MIN_SCORE for t in texts)

    def _mentions_other_customer(self, request: str, customer_id: str) -> bool:
        words = set(re.findall(r"[a-z]+", request.lower()))
        return any(c.customer_id != customer_id and c.name.split()[0].lower() in words for c in self.tools.customers.values())

    def _result(self, case_id, ticket_id, status, answer, citations, memory_used, events, requests=0) -> AgentResult:
        return AgentResult(case_id=case_id, ticket_id=ticket_id, status=status, answer=answer, citations=citations,
                           tool_calls=[c.model_copy() for c in self.tools.calls], memory_used=memory_used,
                           safety_events=events, model_requests=requests)