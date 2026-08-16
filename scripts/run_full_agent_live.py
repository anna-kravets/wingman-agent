"""Run the approved LH318 full-agent conversation and save every review artifact.

This is intentionally separate from the search-only and DocumentationAgent runners:
it exercises the real Supervisor, every specialist, retrieval, composition, and a
follow-up using the first turn's stored results. The JSON is written after each turn
so a later failure cannot discard calls already paid for.
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
from lib.agents import documentation_agent, supervisor  # noqa: E402
from lib.tools import flights, hotels  # noqa: E402


FIRST_PROMPT = (
    "Lufthansa flight LH318 from Tel Aviv to Frankfurt was cancelled at the gate today. "
    "The airline said it was an operational issue. I am travelling with one child, my "
    "checked bags are with the airline, and the next flight they offered is tomorrow "
    "afternoon. I need a hotel tonight, meals, and the earliest reasonable replacement "
    "flight. Please also explain whether I can choose a refund and what compensation may apply."
)

FOLLOW_UP_PROMPT = "What about hotels? Can you suggest nearby options and include the estimated prices?"

# Frozen from the user's 16/8/2026 production trace. This makes the regression
# repeatable after the flights have departed and when the local AeroDataBox key is
# unavailable; models, embeddings, Pinecone and every agent remain live.
FROZEN_FLIGHTS = [
    {"flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH",
     "origin": "TLV", "destination": "FRA", "depart": "2026-08-16T16:30+03:00",
     "arrive": "2026-08-16T20:10+02:00", "status": "Expected",
     "aircraft": "Airbus A320", "terminal": "3"},
    {"flight": "LY 357", "airline": "El Al", "airline_iata": "LY",
     "origin": "TLV", "destination": "FRA", "depart": "2026-08-17T06:05+03:00",
     "arrive": "2026-08-17T09:40+02:00", "status": "Expected",
     "aircraft": "Boeing 737-900", "terminal": "3"},
    {"flight": "DE 4308", "airline": "Condor", "airline_iata": "DE",
     "origin": "TLV", "destination": "FRA", "depart": "2026-08-17T07:10+03:00",
     "arrive": "2026-08-17T10:40+02:00", "status": "Expected",
     "aircraft": "Airbus A320", "terminal": "3"},
]

FROZEN_HOTELS = [
    {"name": "TEL AVIV AIRPORT ABEL", "kind": "hostel", "distance_km": 1.9,
     "area": "Tel Aviv"},
    {"name": "Medical Hotel Shai Lev", "distance_km": 5.6, "area": "Tel Aviv",
     "wheelchair": "yes"},
    {"name": "Apropo", "distance_km": 5.7, "area": "Tel Aviv",
     "phone": "+972 3 535 2727", "website": "http://www.apropohotel.co.il",
     "wheelchair": "yes"},
    {"name": "Sadot", "distance_km": 6.3, "area": "Tel Aviv"},
    {"name": "Kfar Ha'Maccabiah", "distance_km": 7.4, "area": "Tel Aviv"},
    {"name": "Kfar Ha'Maccabiah Hotel", "distance_km": 7.4, "area": "Tel Aviv"},
    {"name": "Rarely Spacious Holiday Home", "kind": "guest_house",
     "distance_km": 8.3, "area": "Tel Aviv", "address": "3 עין יהב",
     "phone": "00972526804086"},
    {"name": "Aristocrat", "distance_km": 8.4, "area": "Tel Aviv",
     "address": "16 דמשק אליעזר", "stars": "4"},
]


def _usage_delta(before: dict, after: dict) -> dict:
    return {
        "calls": after["calls"] - before["calls"],
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "finish_reasons": after["finish_reasons"][len(before["finish_reasons"]):],
    }


def _write(output: Path, artifact: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_turn(prompt: str, history: list[dict], local_time: datetime) -> dict:
    before = {key: list(value) if isinstance(value, list) else value
              for key, value in llm.usage.items()}
    started = time.monotonic()
    turn: dict = {
        "prompt": prompt,
        "local_time": local_time.isoformat(timespec="seconds"),
        "history_turns_supplied": len(history),
    }
    try:
        response, steps, results = supervisor.run(
            prompt, history, local_time=local_time)
        turn.update({
            "status": "ok",
            "response": response,
            "steps": steps,
            "results": results,
            "module_sequence": [step.get("module") for step in steps],
        })
    except Exception as exc:  # noqa: BLE001 - a paid failure must still be saved
        turn.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "steps": getattr(exc, "steps", []),
            "results": {},
        })
    turn["elapsed_seconds"] = round(time.monotonic() - started, 3)
    after = {key: list(value) if isinstance(value, list) else value
             for key, value in llm.usage.items()}
    turn["chat_usage"] = _usage_delta(before, after)
    return turn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-paid-calls",
        action="store_true",
        help="Required: makes live text, embedding, Pinecone, flight and hotel calls.",
    )
    parser.add_argument("--max-llm-calls", type=int, default=14)
    parser.add_argument(
        "--use-frozen-search-data",
        action="store_true",
        help="Replay the exact verified flights/hotels from the reported conversation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "live-test-output" / "lh318_full_agent_after_fixes.json",
    )
    args = parser.parse_args()
    if not args.confirm_paid_calls:
        parser.error("Pass --confirm-paid-calls only after explicit user approval.")

    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT.parent / ".env", override=False)
    os.environ["WINGMAN_ALLOW_LLM"] = "1"
    os.environ["WINGMAN_LIVE_DATA"] = "1"

    required = ["LLMOD_API_KEY", "LLMOD_API_BASE", "PINECONE_API_KEY"]
    if not args.use_frozen_search_data:
        required.append("AERODATABOX_API_KEY")
    missing = [name for name in required
               if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing required configuration: " + ", ".join(missing))

    output = args.output.resolve()
    started_at = datetime.now().astimezone()
    artifact: dict = {
        "status": "running",
        "started_at": started_at.isoformat(timespec="seconds"),
        "saved_at": None,
        "text_model": llm.TEXT_MODEL,
        "embedding_model": llm.EMBEDDING_MODEL,
        "documentation_reflection_rounds": documentation_agent.MAX_REFLECTION_ROUNDS,
        "max_llm_calls": args.max_llm_calls,
        "search_data": {
            "mode": "frozen_verified_replay" if args.use_frozen_search_data else "live",
            "provenance": (
                "Exact candidates saved from the user's 2026-08-16 production conversation."
                if args.use_frozen_search_data else "Live AeroDataBox and OpenStreetMap calls."
            ),
            "flights": FROZEN_FLIGHTS if args.use_frozen_search_data else None,
            "hotels": FROZEN_HOTELS if args.use_frozen_search_data else None,
        },
        "configuration_present": {
            name: bool(os.environ.get(name)) for name in
            ("LLMOD_API_KEY", "LLMOD_API_BASE", "PINECONE_API_KEY",
             "PINECONE_INDEX_NAME", "AERODATABOX_API_KEY")
        },
        "turns": [],
    }
    _write(output, artifact)

    if args.use_frozen_search_data:
        flights.search = lambda *args, **kwargs: [dict(item) for item in FROZEN_FLIGHTS]
        hotels.search = lambda *args, **kwargs: [dict(item) for item in FROZEN_HOTELS]
        local_time = datetime.fromisoformat("2026-08-16T16:00:01")
    else:
        local_time = datetime.now()
    first = _run_turn(FIRST_PROMPT, [], local_time)
    artifact["turns"].append(first)
    artifact["saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write(output, artifact)

    if first["status"] == "ok" and llm.usage["calls"] < args.max_llm_calls:
        history = [{
            "prompt": FIRST_PROMPT,
            "response": first["response"],
            "results": first["results"],
        }]
        follow_time = (datetime.fromisoformat("2026-08-16T16:04:26")
                       if args.use_frozen_search_data else datetime.now())
        follow_up = _run_turn(FOLLOW_UP_PROMPT, history, follow_time)
        artifact["turns"].append(follow_up)

    artifact["status"] = (
        "ok" if artifact["turns"] and
        all(turn["status"] == "ok" for turn in artifact["turns"])
        else "error"
    )
    artifact["saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact["elapsed_seconds"] = round(
        sum(turn["elapsed_seconds"] for turn in artifact["turns"]), 3)
    artifact["chat_usage"] = {key: list(value) if isinstance(value, list) else value
                              for key, value in llm.usage.items()}
    _write(output, artifact)

    print(json.dumps({
        "status": artifact["status"],
        "output": str(output),
        "turns": len(artifact["turns"]),
        "elapsed_seconds": artifact["elapsed_seconds"],
        "chat_usage": artifact["chat_usage"],
        "steps_per_turn": [len(turn.get("steps", [])) for turn in artifact["turns"]],
    }, indent=2))
    return 0 if artifact["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
