"""The single choke point for every LLM call in the project.

Call through `call()` and the `steps` entry is produced for you. The spec requires
every LLM call to appear in `/api/execute`'s `steps[]`, in order — routing all of
them through one function makes that structural instead of something three people
have to remember.

Token usage is accumulated here for the same reason: it is the only place that
sees every call, so it is the only place that can answer "how much of the $13 did
that turn cost".

UNVERIFIED: the request/response shape below assumes LLMod.ai is OpenAI-compatible
(`POST {base}/chat/completions`). Nobody has had a key yet. Confirm against the real
endpoint the moment one exists — if it differs, this file is the only thing to change.
"""

import json
import os

import httpx

from lib.steps import make_step

TEXT_MODEL = "MB5R2CF-azure/gpt-5.4-mini"
EMBEDDING_MODEL = "MB5R2CF-azure/text-embedding-3-small"

TIMEOUT_SECONDS = 60

# Accumulated across the process, read by the budget check. Serverless instances are
# recycled, so this is a per-instance figure for dev — not a running project total.
usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


class LLMError(RuntimeError):
    """A call failed. `steps` carries the trace entry for it.

    The spec requires `steps[]` to describe *every* LLM call in order, and a call
    that was made and then failed is still a call. Raising bare would silently drop
    it from the trace, so the step travels with the exception and the Supervisor
    appends it (see `supervisor.dispatch`). Empty only when no call was attempted —
    a missing API key, say.
    """

    def __init__(self, message: str, steps: list[dict] | None = None):
        super().__init__(message)
        self.steps = steps or []


def _config() -> tuple[str, str]:
    key = os.environ.get("LLMOD_API_KEY")
    base = os.environ.get("LLMOD_API_BASE")
    if not key or not base:
        raise LLMError(
            "LLMod.ai is not configured — set LLMOD_API_KEY and LLMOD_API_BASE "
            "(see .env.example)."
        )
    return key, base.rstrip("/")


def call(
    module: str,
    system_prompt: str,
    user_prompt: str,
    *,
    expect_json: bool = False,
) -> tuple[object, dict]:
    """Make one LLM call. Returns (response, step).

    `response` is the parsed object when `expect_json`, otherwise the raw text.
    `step` is ready to append to the `steps[]` trace — `module` must be one of the
    names on the architecture diagram (Supervisor, FlightAgent, AccommodationAgent,
    DocumentationAgent).
    """
    key, base = _config()

    body = {
        "model": TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if expect_json:
        body["response_format"] = {"type": "json_object"}

    def failed(reason: str) -> LLMError:
        step = make_step(module, system_prompt, user_prompt, {"error": reason})
        return LLMError(f"{module}: {reason}", steps=[step])

    try:
        http_response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=TIMEOUT_SECONDS,
        )
        http_response.raise_for_status()
        data = http_response.json()
        text = data["choices"][0]["message"]["content"]
    except httpx.HTTPError as exc:
        raise failed(f"LLM request failed — {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise failed(f"unexpected response from the LLM — {exc}") from exc

    counts = data.get("usage") or {}
    usage["calls"] += 1
    usage["prompt_tokens"] += counts.get("prompt_tokens", 0)
    usage["completion_tokens"] += counts.get("completion_tokens", 0)

    if expect_json:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise failed(f"expected JSON, got {text[:200]!r}") from exc
        return parsed, make_step(module, system_prompt, user_prompt, parsed)

    return text, make_step(module, system_prompt, user_prompt, {"text": text})
