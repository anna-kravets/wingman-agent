"""Run an explicitly approved live retrieval-only validation and save its evidence."""

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

from lib.rag.retrieve import retrieve_with_coverage  # noqa: E402
from lib.rag.routing import route_request  # noqa: E402


SCENARIOS = {
    "ly-eu-il-cancellation": {
        "airline": "LY",
        "flight_number": "LY324",
        "origin": "CDG",
        "destination": "TLV",
        "disruption": "cancelled",
        "party_size": 1,
        "local_now": "2026-08-10T20:00:00+03:00",
        "_passenger_prompt": (
            "My EL AL flight from Paris to Tel Aviv was cancelled at the airport. "
            "What assistance, rerouting, refund, and compensation rights apply?"
        ),
    },
    "aa-us-denied-boarding": {
        "airline": "AA",
        "flight_number": "AA100",
        "origin": "JFK",
        "destination": "LAX",
        "disruption": "denied_boarding",
        "stranded_at": "JFK",
        "party_size": 1,
        "local_now": "2026-08-11T15:00:00-04:00",
        "_passenger_prompt": (
            "American Airlines involuntarily denied me boarding at JFK because my "
            "flight to Los Angeles was oversold. What rebooking, refund, care, and "
            "cash compensation rights do I have?"
        ),
    },
    "ua-us-tarmac-delay": {
        "airline": "UA",
        "flight_number": "UA123",
        "origin": "EWR",
        "destination": "ORD",
        "disruption": "delayed",
        "stranded_at": "EWR",
        "party_size": 1,
        "local_now": "2026-08-12T14:00:00-04:00",
        "_passenger_prompt": (
            "My United flight has been sitting on the Newark tarmac for five hours. "
            "What food, water, deplaning, refund, and cash-compensation rights apply?"
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-paid-embedding-calls",
        action="store_true",
        help="Required: makes one primary and, only if needed, one fallback embedding call.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="ly-eu-il-cancellation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "live-test-output" / "provision_retrieval_live.json",
    )
    args = parser.parse_args()
    if not args.confirm_paid_embedding_calls:
        parser.error("Pass --confirm-paid-embedding-calls after reviewing the API-call scope.")

    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT.parent / ".env", override=False)
    request = dict(SCENARIOS[args.scenario])
    route = route_request(request)
    query = (
        f"{request['_passenger_prompt']}\nAirline: {request['airline']}\n"
        f"Flight: {request['flight_number']}\n"
        f"Route: {request['origin']} -> {request['destination']}\n"
        f"Disruption: {request['disruption']}"
    )

    started = time.monotonic()
    result = retrieve_with_coverage(
        query,
        namespaces=route.namespaces,
        disruption=request["disruption"],
    )
    artifact = {
        "started_at": datetime.now().astimezone().isoformat(),
        "scenario": args.scenario,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "request": request,
        "namespaces": route.namespaces,
        "embedding_batches": result.embedding_batches,
        "embedding_queries": result.embedding_queries,
        "coverage": {
            "required": result.coverage.required,
            "covered": result.coverage.covered,
            "missing": result.coverage.missing,
            "primary_recovery_attempted": result.coverage.recovery_attempted,
            "fallback_attempted": result.coverage.fallback_attempted,
        },
        "passages": [
            {
                "namespace": passage.namespace,
                "score": passage.score,
                "selection_reason": passage.selection_reason,
                "recovery_added": passage.recovery_added,
                "metadata": dict(passage.document.metadata),
                "text": passage.document.page_content,
            }
            for passage in result.passages
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_seconds": artifact["elapsed_seconds"],
                "embedding_batches": result.embedding_batches,
                "passages": len(result.passages),
                "coverage_complete": result.coverage.is_complete,
                "missing": result.coverage.missing,
            },
            indent=2,
        )
    )
    return 0 if result.coverage.is_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
