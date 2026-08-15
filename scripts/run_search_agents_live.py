"""Run explicitly confirmed live search-agent scenarios and save their traces.

Drives the real Supervisor so the date sync - the only thing connecting
FlightAgent and AccommodationAgent - is exercised rather than bypassed.
DocumentationAgent is replaced with the same no-cost fake the unit tests use,
so nothing here spends on Person C's half or needs Pinecone.

Budget: AeroDataBox is 2 units per call from a 600/month allowance, and every
chat call costs against a $13 project ceiling. `--all` runs in one process so
the route cache is shared and repeated routes cost nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, time as clock, timedelta
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.search_cases import load_search_cases  # noqa: E402
from lib import llm  # noqa: E402
from lib.agents import documentation_agent, supervisor  # noqa: E402
from lib.steps import make_step  # noqa: E402
from lib.tools import flights, hotels  # noqa: E402

CASES = load_search_cases()

DOC_PAYLOAD = {"regulation": "EU 261/2004", "entitlements": [],
               "next_actions": [], "caveats": ["Faked for search-agent validation."]}


def fake_documentation_run(request, history):
    """DocumentationAgent is Person C's; validating it here would spend on their half."""
    steps = [make_step("DocumentationAgent", phase, "faked for search validation", DOC_PAYLOAD)
             for phase in ("draft", "critique", "refine")]
    return DOC_PAYLOAD, steps


def guard_budget(used: int, ceiling: int) -> None:
    if used >= ceiling:
        raise RuntimeError(
            f"LLM call ceiling reached ({used}/{ceiling}). Raise --max-llm-calls deliberately.")


def build_request_patch(case: dict, original):
    """Wrap the Supervisor's extraction so a scenario can fix local_now and fields.

    local_now cannot be driven from the prompt: the Supervisor's stub derives it
    from datetime.now(), so 'a late-evening disruption' is not expressible as text.
    Real extraction still runs; only the named fields are overridden.
    """
    when = datetime.combine(
        date.today() + timedelta(days=case["local_now"]["offset_days"]),
        clock.fromisoformat(case["local_now"]["time"]),
    )

    def patched(prompt, history, _local_now):
        request, step = original(prompt, history, when.isoformat(timespec="seconds"))
        request.update(case["request_override"])
        request["missing"] = [f for f in supervisor.REQUIRED_FIELDS if not request.get(f)]
        return request, step

    return patched


def run_case(case: dict) -> dict:
    os.environ["WINGMAN_LIVE_DATA"] = "1" if case["live_data"] else "0"

    original_extract = supervisor._extract_request
    original_doc_run = documentation_agent.run
    supervisor._extract_request = build_request_patch(case, original_extract)
    documentation_agent.run = fake_documentation_run

    started = time.monotonic()
    artifact = {"scenario": case["case_id"], "title": case["title"],
                "probes": case["probes"], "live_data": case["live_data"],
                "started_at": datetime.now().astimezone().isoformat(),
                "model": llm.TEXT_MODEL}
    try:
        response, steps, _ = supervisor.run(case["prompt"], list(case["history"]))
        artifact.update({"status": "ok", "response": response, "steps": steps})
    except Exception as exc:  # noqa: BLE001 - the artifact must record any failure
        artifact.update({"status": "error", "error": str(exc),
                         "steps": getattr(exc, "steps", [])})
    finally:
        supervisor._extract_request = original_extract
        documentation_agent.run = original_doc_run

    artifact["elapsed_seconds"] = round(time.monotonic() - started, 3)
    artifact["observed"] = _observed(artifact)
    return artifact


def _observed(artifact: dict) -> dict:
    """What the live data actually gave us, so the report can be honest.

    Scenarios like 'thin route' depend on conditions we cannot guarantee on the
    day; recording the real counts lets the report say when one did not reproduce.
    """
    from evals.search_artifact import candidates_from_prompt

    observed = {}
    for module, key in (("FlightAgent", "flight_candidates"),
                        ("AccommodationAgent", "hotel_candidates")):
        step = next((s for s in artifact.get("steps", []) if s["module"] == module), None)
        observed[key] = len(candidates_from_prompt(step["prompt"]["user_prompt"])) if step else None
    stay = next((s for s in artifact.get("steps", []) if s["module"] == "AccommodationAgent"), None)
    observed["accommodation_dispatched"] = stay is not None
    if stay:
        options = (stay.get("response") or {}).get("options") or []
        observed["nights"] = options[0].get("nights") if options else None
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid-calls", action="store_true",
                        help="Required: makes live chat-model and AeroDataBox calls.")
    parser.add_argument("--scenario", choices=tuple(CASES))
    parser.add_argument("--all", action="store_true",
                        help="Run every scenario in one process so the route cache is shared.")
    # Four calls per scenario since the Supervisor's own two seams became real:
    # refine, FlightAgent, AccommodationAgent, compose.
    parser.add_argument("--max-llm-calls", type=int, default=80)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "live-test-output" / "search_agents_live.json")
    args = parser.parse_args()

    if not args.confirm_paid_calls:
        parser.error("Pass --confirm-paid-calls after reviewing the expected call scope.")
    if not (args.scenario or args.all):
        parser.error("Pass --scenario NAME or --all.")

    load_dotenv(REPO_ROOT / ".env", override=False)

    selected = list(CASES.values()) if args.all else [CASES[args.scenario]]
    runs = []
    stopped = None
    # Every scenario is written out even if the run stops early. A ceiling hit once
    # discarded ~40 calls' worth of paid results because the bundle was only saved
    # at the end, which is a poor way to treat a $13 budget.
    try:
        for case in selected:
            guard_budget(llm.usage["calls"], args.max_llm_calls)
            print(f"--- {case['case_id']} (llm calls so far: {llm.usage['calls']})", flush=True)
            runs.append(run_case(case))
    except RuntimeError as exc:
        stopped = str(exc)
        print(f"!!! stopping early: {exc}", flush=True)

    bundle = {"started_at": datetime.now().astimezone().isoformat(),
              "model": llm.TEXT_MODEL, "runs": runs, "chat_usage": dict(llm.usage),
              "stopped_early": stopped,
              "flight_cache_entries": len(flights._cache),
              "hotel_cache_entries": len(hotels._cache)}

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output), "scenarios": len(runs),
                      "chat_usage": bundle["chat_usage"],
                      "stopped_early": stopped,
                      "errors": [r["scenario"] for r in runs if r["status"] != "ok"]}, indent=2))
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
