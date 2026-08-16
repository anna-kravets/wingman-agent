"""Re-run only Supervisor composition from a saved full-agent artifact.

Use this after composition-only fixes: it preserves the expensive flight, hotel,
embedding, Pinecone and DocumentationAgent results and spends at most two text calls.
The new response, full composition steps, usage, and validation result are saved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lib import llm  # noqa: E402
from lib.agents import supervisor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid-calls", action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_paid_calls:
        parser.error("Pass --confirm-paid-calls only after explicit user approval.")

    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT.parent / ".env", override=False)
    os.environ["WINGMAN_ALLOW_LLM"] = "1"

    source_path = args.source.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    index = args.turn - 1
    if index < 0 or index >= len(source.get("turns") or []):
        raise SystemExit(f"Turn {args.turn} does not exist in {source_path}")
    saved = source["turns"][index]
    if saved.get("status") != "ok":
        raise SystemExit(f"Turn {args.turn} was not successful in {source_path}")

    refine = next(step for step in saved["steps"] if step.get("module") == "Supervisor")
    request = dict(refine["response"])
    request.update({
        "local_now": saved["local_time"],
        "_passenger_prompt": saved["prompt"],
        "assumptions": [],
    })
    history = []
    for prior in source["turns"][:index]:
        history.append({
            "prompt": prior["prompt"],
            "response": prior["response"],
            "results": prior["results"],
        })

    started = time.monotonic()
    before = {key: list(value) if isinstance(value, list) else value
              for key, value in llm.usage.items()}
    response, steps = supervisor._compose(request, saved["results"], [], history)
    usage = {
        "calls": llm.usage["calls"] - before["calls"],
        "prompt_tokens": llm.usage["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": llm.usage["completion_tokens"] - before["completion_tokens"],
        "finish_reasons": llm.usage["finish_reasons"][len(before["finish_reasons"]):],
    }
    artifact = {
        "status": "ok",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_artifact": str(source_path),
        "source_turn": args.turn,
        "prompt": saved["prompt"],
        "request": request,
        "results": saved["results"],
        "response": response,
        "steps": steps,
        "module_sequence": [step.get("module") for step in steps],
        "chat_usage": usage,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "remaining_composition_issues": supervisor._composition_issues(
            response, request, saved["results"]),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": artifact["status"],
        "output": str(output),
        "calls": usage["calls"],
        "elapsed_seconds": artifact["elapsed_seconds"],
        "remaining_composition_issues": artifact["remaining_composition_issues"],
    }, indent=2))
    return 0 if not artifact["remaining_composition_issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
