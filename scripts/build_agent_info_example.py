"""Build the static /api/agent_info example from one saved real-agent artifact."""

from __future__ import annotations

import argparse
import base64
import json
import textwrap
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _example_from_artifact(path: Path) -> tuple[dict, dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    turns = artifact.get("turns")
    if artifact.get("status") != "ok" or not isinstance(turns, list) or len(turns) != 1:
        raise ValueError("artifact must contain exactly one successful turn")

    turn = turns[0]
    if turn.get("status") != "ok":
        raise ValueError("captured turn did not complete successfully")
    steps = turn.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("captured turn has no execution trace")
    for step in steps:
        if set(step) != {"module", "prompt", "response"}:
            raise ValueError("captured step does not match the required API schema")
        if set(step["prompt"]) != {"system_prompt", "user_prompt"}:
            raise ValueError("captured prompt does not match the required API schema")

    example = {
        "prompt": turn["prompt"],
        "full_response": turn["response"],
        "steps": steps,
    }
    if steps[-1].get("response", {}).get("text") != example["full_response"]:
        raise ValueError("final traced response differs from the passenger-facing response")
    return artifact, example


def _module_source(artifact: dict, example: dict, source_name: str) -> str:
    raw = json.dumps([example], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    lines = "\n".join(f'    "{line}"' for line in textwrap.wrap(payload, width=100))
    turn = artifact["turns"][0]
    usage = turn.get("chat_usage") or {}
    return f'''"""Real /api/execute example returned by GET /api/agent_info.

Captured from {source_name} on {artifact.get("started_at", "an unknown date")}.
The single turn made {usage.get("calls", len(example["steps"]))} text-model calls and preserves
the exact passenger prompt, final response, and complete ordered LLM trace.
"""

import base64
import json
import zlib


_CAPTURE = (
{lines}
)

PROMPT_EXAMPLES: list[dict] = json.loads(
    zlib.decompress(base64.b64decode(_CAPTURE)).decode("utf-8")
)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "api" / "prompt_examples.py",
    )
    args = parser.parse_args()

    artifact_path = args.artifact.resolve()
    artifact, example = _example_from_artifact(artifact_path)
    output = args.output.resolve()
    output.write_text(
        _module_source(artifact, example, artifact_path.name),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "prompt_chars": len(example["prompt"]),
        "response_chars": len(example["full_response"]),
        "steps": len(example["steps"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
