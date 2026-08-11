"""Load and validate DocumentationAgent regression and held-out cases."""

from __future__ import annotations

import json
from pathlib import Path


CASES_PATH = Path(__file__).with_name("documentation_agent_cases.json")
VALID_SPLITS = {"regression", "holdout"}
REQUIRED_REQUEST_FIELDS = {
    "airline",
    "flight_number",
    "origin",
    "destination",
    "disruption",
    "party_size",
    "local_now",
    "_passenger_prompt",
}


def load_documentation_cases(path: Path = CASES_PATH) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported documentation evaluation schema")

    cases: dict[str, dict] = {}
    for case in data.get("cases", []):
        case_id = str(case.get("case_id", "")).strip()
        if not case_id or case_id in cases:
            raise ValueError(f"Missing or duplicate documentation case_id: {case_id!r}")
        if case.get("split") not in VALID_SPLITS:
            raise ValueError(f"Invalid split for {case_id}")
        request = case.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"Missing request for {case_id}")
        missing = REQUIRED_REQUEST_FIELDS - set(request)
        if missing:
            raise ValueError(f"{case_id} request is missing: {', '.join(sorted(missing))}")
        for field in (
            "expected_namespaces",
            "must_cover_topics",
            "expected_findings",
            "forbidden_findings",
            "error_categories",
        ):
            if not isinstance(case.get(field), list) or not case[field]:
                raise ValueError(f"{case_id} must define a non-empty {field}")
        cases[case_id] = case
    if not cases:
        raise ValueError("No documentation evaluation cases were found")
    return cases
