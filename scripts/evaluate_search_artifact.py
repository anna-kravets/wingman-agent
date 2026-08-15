"""Evaluate a saved search-agent live-run artifact. No API or database access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.search_artifact import evaluate  # noqa: E402
from evals.search_cases import load_search_cases  # noqa: E402

CASES = load_search_cases()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True,
                        help="A run artifact, either one scenario or an --all bundle.")
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    runs = data["runs"] if "runs" in data else [data]

    failed = 0
    for run in runs:
        case = CASES[run["scenario"]]
        results = evaluate(run, case)
        print(f"\n=== {run['scenario']} - {case['title']} ({run.get('status')}) ===")
        for result in results:
            mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip"}[result["status"]]
            print(f"  [{mark}] {result['check']}: {result['detail']}")
            failed += result["status"] == "fail"

    print(f"\n{failed} failing check(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
