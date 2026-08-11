"""Run one explicitly confirmed live DocumentationAgent test and save its full trace."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lib import llm  # noqa: E402
from lib.agents import documentation_agent  # noqa: E402
from evals.documentation_cases import load_documentation_cases  # noqa: E402


CASES = load_documentation_cases()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-paid-calls",
        action="store_true",
        help="Required: makes live embedding, Pinecone, and chat-model calls.",
    )
    parser.add_argument(
        "--reflection-rounds",
        type=int,
        default=documentation_agent.MAX_REFLECTION_ROUNDS,
        help="Maximum critique/refine rounds; total chat calls are at most 1 + 2 * rounds.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(CASES),
        default="ly-eu-il-cancellation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "live-test-output" / "documentation_agent_live_test.json",
    )
    args = parser.parse_args()
    if not args.confirm_paid_calls:
        parser.error("Pass --confirm-paid-calls after reviewing the expected API-call scope.")
    if args.reflection_rounds < 1:
        parser.error("--reflection-rounds must be at least 1")

    documentation_agent.MAX_REFLECTION_ROUNDS = args.reflection_rounds

    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT.parent / ".env", override=False)

    case = CASES[args.scenario]
    request = dict(case["request"])

    started = time.monotonic()
    artifact: dict = {
        "started_at": datetime.now().astimezone().isoformat(),
        "model": llm.TEXT_MODEL,
        "embedding_model": llm.EMBEDDING_MODEL,
        "reflection_rounds": documentation_agent.MAX_REFLECTION_ROUNDS,
        "scenario": args.scenario,
        "evaluation_split": case["split"],
        "quality_contract": {
            key: case[key]
            for key in (
                "expected_namespaces",
                "must_cover_topics",
                "expected_findings",
                "forbidden_findings",
                "error_categories",
            )
        },
        "request": request,
    }
    try:
        payload, steps = documentation_agent.run(request, [])
        artifact.update({"status": "ok", "payload": payload, "steps": steps})
    except Exception as exc:
        artifact.update(
            {
                "status": "error",
                "error": str(exc),
                "steps": getattr(exc, "steps", []),
            }
        )
    artifact["elapsed_seconds"] = round(time.monotonic() - started, 3)
    artifact["chat_usage"] = dict(llm.usage)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "output": str(output),
                "elapsed_seconds": artifact["elapsed_seconds"],
                "chat_usage": artifact["chat_usage"],
                "steps": len(artifact["steps"]),
            },
            indent=2,
        )
    )
    return 0 if artifact["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
