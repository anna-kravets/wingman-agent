"""Evaluate a saved DocumentationAgent artifact without making API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.documentation_artifact import (  # noqa: E402
    evaluate_documentation_artifact,
    load_artifact,
)
from evals.documentation_cases import load_documentation_cases  # noqa: E402


def main() -> int:
    cases = load_documentation_cases()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=tuple(cases))
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()

    result = evaluate_documentation_artifact(
        cases[args.scenario], load_artifact(args.artifact)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["automated_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
