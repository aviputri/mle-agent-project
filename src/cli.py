"""
Step 10 - Command line entry points.

uv run python -m src inspect-data
uv run python -m src index
uv run python -m src run-case refund_duplicate_payment
uv run python -m src chat                       # multi-turn, durable memory in storage/
uv run python -m src evaluate [--output reports/evaluation_report.md]
"""

import argparse
import json
from pathlib import Path

from src.agent import OperationsAgent
from src.config import load_settings
from src.data import validate_data
from src.evaluation import load_cases, write_report
from src.retrieval import PolicyRetriever, load_chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="northstar-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect-data")
    index = sub.add_parser("index")
    index.add_argument("--query", default=None)
    run = sub.add_parser("run-case")
    run.add_argument("case_id")
    sub.add_parser("chat")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--output", type=Path, default=Path("reports/evaluation_report.md"))
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "inspect-data":
        print(json.dumps(validate_data(settings.data_dir), indent=2))
    elif args.command == "index":
        chunks = load_chunks(settings.knowledge_base_dir)
        if args.query:
            print(PolicyRetriever(chunks, settings.top_k, settings.min_score).search(args.query).model_dump_json(indent=2))
        for c in chunks:
            print(f"{c.chunk_id:<30} source={c.source_id:<26} {c.text[:60]}...")
    elif args.command == "run-case":
        case = next(c for c in load_cases(settings) if c.case_id == args.case_id)
        print(OperationsAgent(settings).run_case(case).model_dump_json(indent=2))
    elif args.command == "chat":
        agent = OperationsAgent(settings, memory_path=settings.storage_dir / "case_memory.json")
        print("Type a request (optionally 'T-1001: ...'). Empty line to quit.")
        while (line := input("> ").strip()):
            ticket, _, request = line.partition(":") if line.startswith("T-") else (None, "", line)
            r = agent.run(request.strip() or line, ticket)
            print(f"[{r.status}] tools={[c.name for c in r.tool_calls]} memory={r.memory_used}\n{r.answer}\n")
    elif args.command == "evaluate":
        label = "dry-run (scripted planner)" if settings.llm_provider == "dry-run" else f"live: {settings.llm_provider}"
        print(f"wrote {write_report(settings, args.output, label=label)}")


if __name__ == "__main__":
    main()