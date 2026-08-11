"""Deterministic structural checks for a saved DocumentationAgent live artifact.

These checks deliberately do not claim to judge legal semantics. The case's expected
and forbidden findings are returned as a human-review rubric.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


SOURCE_LABEL = re.compile(r"\[S\d+\]")


def _coverage_values(prompt: str, heading: str) -> set[str]:
    match = re.search(rf"(?m)^{re.escape(heading)}:\s*(.*)$", prompt)
    if not match or match.group(1).strip() == "none":
        return set()
    return {item.strip() for item in match.group(1).split(",") if item.strip()}


def evaluate_documentation_artifact(case: dict, artifact: dict) -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    record("run_status", artifact.get("status") == "ok", str(artifact.get("status")))
    steps = artifact.get("steps") if isinstance(artifact.get("steps"), list) else []
    record("reflection_trace", len(steps) >= 3, f"steps={len(steps)}")
    if not steps:
        return _result(case, checks)

    draft_prompt = str(steps[0].get("prompt", {}).get("user_prompt", ""))
    required = _coverage_values(draft_prompt, "Required topics")
    covered = _coverage_values(draft_prompt, "Covered topics")
    expected = set(case["must_cover_topics"])
    record(
        "required_topics_present",
        expected <= required,
        f"missing={sorted(expected - required)}",
    )
    record(
        "required_topics_covered",
        expected <= covered,
        f"missing={sorted(expected - covered)}",
    )

    evidence_labels = set(SOURCE_LABEL.findall(draft_prompt))
    payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else {}
    final_sources = "\n".join(
        str(item.get("source", ""))
        for item in payload.get("entitlements", [])
        if isinstance(item, dict)
    )
    cited_labels = set(SOURCE_LABEL.findall(final_sources))
    record(
        "citation_labels_exist",
        cited_labels <= evidence_labels,
        f"unknown={sorted(cited_labels - evidence_labels)}",
    )
    record(
        "entitlements_are_cited",
        all(
            SOURCE_LABEL.search(str(item.get("source", "")))
            for item in payload.get("entitlements", [])
            if isinstance(item, dict)
        ),
        f"entitlements={len(payload.get('entitlements', []))}",
    )

    critique = (
        steps[1].get("response", {})
        if len(steps) > 1 and isinstance(steps[1].get("response"), dict)
        else {}
    )
    audit = critique.get("topic_audit") if isinstance(critique.get("topic_audit"), list) else []
    audited_topics = {
        str(item.get("topic")) for item in audit if isinstance(item, dict)
    }
    record(
        "critical_topics_audited",
        expected <= audited_topics,
        f"missing={sorted(expected - audited_topics)}",
    )
    required_audit_fields = {
        "topic",
        "status",
        "source_labels",
        "rule_branches_and_conditions",
        "missing_from_assessment",
        "reason",
    }
    malformed = [
        str(item.get("topic", "unknown"))
        for item in audit
        if isinstance(item, dict) and not required_audit_fields <= set(item)
    ]
    record("condition_audit_schema", not malformed, f"malformed={malformed}")

    return _result(case, checks)


def _result(case: dict, checks: list[dict]) -> dict:
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "automated_pass": all(check["passed"] for check in checks),
        "checks": checks,
        "manual_review": {
            "expected_findings": case["expected_findings"],
            "forbidden_findings": case["forbidden_findings"],
            "error_categories": case["error_categories"],
        },
    }


def load_artifact(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
