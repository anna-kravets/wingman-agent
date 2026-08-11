"""No-cost checks for DocumentationAgent regression and holdout fixtures."""

from evals.documentation_artifact import evaluate_documentation_artifact
from evals.documentation_cases import load_documentation_cases
from lib.rag.coverage import requirements_for
from lib.rag.routing import route_request


def test_evaluation_suite_has_regressions_and_diverse_holdouts():
    cases = load_documentation_cases()
    regressions = {case_id for case_id, case in cases.items() if case["split"] == "regression"}
    holdouts = [case for case in cases.values() if case["split"] == "holdout"]

    assert regressions == {"ly-eu-il-cancellation", "aa-us-denied-boarding"}
    assert len(holdouts) >= 6
    assert {case["request"]["disruption"] for case in holdouts} >= {
        "cancelled",
        "delayed",
        "denied_boarding",
    }
    assert len({case["request"]["airline"] for case in holdouts}) >= 5


def test_evaluation_routes_and_critical_topics_match_runtime_rules():
    for case in load_documentation_cases().values():
        route = route_request(case["request"])
        assert list(route.namespaces) == case["expected_namespaces"], case["case_id"]

        required = {
            requirement.key
            for requirement in requirements_for(
                case["request"]["disruption"], route.namespaces
            )
        }
        assert set(case["must_cover_topics"]) <= required, case["case_id"]


def test_structural_artifact_evaluator_checks_richer_reflection_schema():
    case = load_documentation_cases()["aa-us-denied-boarding"]
    topics = case["must_cover_topics"]
    coverage = ", ".join(topics)
    topic_audit = [
        {
            "topic": topic,
            "status": "addressed",
            "source_labels": ["[S1]"],
            "rule_branches_and_conditions": ["all material branches checked"],
            "missing_from_assessment": [],
            "reason": "supported",
        }
        for topic in topics
    ]
    artifact = {
        "status": "ok",
        "steps": [
            {
                "prompt": {
                    "user_prompt": (
                        f"Required topics: {coverage}\nCovered topics: {coverage}\n"
                        "[S1] Source excerpt"
                    )
                },
                "response": {},
            },
            {"response": {"topic_audit": topic_audit}},
            {"response": {}},
        ],
        "payload": {
            "entitlements": [
                {
                    "kind": "cash_compensation",
                    "summary": "Conditional compensation.",
                    "source": "[S1] supported rule",
                    "confidence": "high",
                }
            ],
            "next_actions": [],
            "caveats": [],
        },
    }

    result = evaluate_documentation_artifact(case, artifact)

    assert result["automated_pass"]
    assert result["manual_review"]["forbidden_findings"]
