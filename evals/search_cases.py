"""Load and validate the search-agent validation scenarios.

Mirrors evals/documentation_cases.py so both halves of the project describe
their evaluation cases the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

CASES_PATH = Path(__file__).with_name("search_agent_cases.json")

CHECK_NAMES = {
    "trace_shape",
    "grounding_flights",
    "grounding_hotels",
    "date_sync",
    "no_accommodation",
    "no_booking_site",
    "price_honesty",
    "no_asserted_fare",
    "meals_honesty",
    "no_unusable_status",
    "degraded_refusal",
    "deferral",
    "history_reached_agents",
}

REQUIRED_FIELDS = {
    "case_id", "title", "probes", "prompt", "history",
    "request_override", "local_now", "live_data", "checks", "expect",
}


def load_search_cases(path: Path = CASES_PATH) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported search evaluation schema version")

    cases: dict[str, dict] = {}
    for case in data.get("cases", []):
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("A case is missing case_id")
        if case_id in cases:
            raise ValueError(f"duplicate case_id: {case_id!r}")

        missing = REQUIRED_FIELDS - set(case)
        if missing:
            raise ValueError(f"{case_id} is missing: {', '.join(sorted(missing))}")
        if not case["probes"] or not case["title"]:
            raise ValueError(f"{case_id} must declare a title and what it probes")

        unknown = set(case["checks"]) - CHECK_NAMES
        if unknown:
            raise ValueError(f"{case_id} names unknown checks: {sorted(unknown)}")

        local_now = case["local_now"]
        if not isinstance(local_now.get("offset_days"), int):
            raise ValueError(f"{case_id} local_now.offset_days must be an int")
        hour, _, minute = str(local_now.get("time", "")).partition(":")
        if not (hour.isdigit() and minute.isdigit()):
            raise ValueError(f"{case_id} local_now.time must be HH:MM")

        cases[case_id] = case

    if not cases:
        raise ValueError("No search evaluation cases were found")
    return cases
