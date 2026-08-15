# Supervisor intake, composition and voice — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Supervisor check what the passenger asserts before spending money on it, consume everything the sub-agents now return, and speak as one assistant rather than as a coordinator of a crew.

**Architecture:** All nine tasks land in `lib/agents/supervisor.py` plus its callers. The refinement pass gains a clock and a `conflicts` list; the gate that already blocks on missing fields learns to block on conflicts too. `_digest` is rewritten against the payload contract Person B published on 15/8/2026, and the two compatibility fields they left open are deleted. `supervisor.run` grows a `local_time` parameter and a third return value carrying the agents' payloads, so a follow-up can be answered from results already paid for.

**Tech Stack:** Python 3.12, FastAPI, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-15-supervisor-intake-and-composition-design.md`

## Global Constraints

- **Module names are locked** and must stay verbatim everywhere, including `steps[].module`: `Supervisor`, `FlightAgent`, `AccommodationAgent`, `DocumentationAgent`.
- **`/api/execute`'s response shape is locked**: `{"status", "error", "response", "steps"}` and nothing else. New inputs are allowed; new output fields are not.
- **Every LLM call must produce exactly one entry in `steps[]`, in call order.** Never hand-build a step — call through `lib.llm.call()`.
- **$13 total LLM budget for the project.** No task here may add an LLM call to any turn. Prompt growth is the only cost this plan spends, and it is capped explicitly.
- **A bare `{"prompt": "..."}` POST must keep working unchanged.** That is what a grader sends.
- **Reversing a documented decision means updating every file that states it, in the same commit** (`CLAUDE.md` §7). `docs/PROJECT_PLAN.md` §6 is the decisions log.
- **No `Co-Authored-By:` trailers** on commits.
- Run tests with the repo venv: `.venv/Scripts/python.exe -m pytest`.
- `tests/conftest.py` is shared with Person B and Person C. Every edit to it must keep their suites green — run the whole suite, not just `tests/test_supervisor.py`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `lib/agents/supervisor.py` | All orchestration: extraction, sanity checks, the gate, the digest, composition | 1–9 |
| `api/index.py` | Parses `local_time`, unpacks the third return value, stores `results` on the turn | 1, 8 |
| `lib/llm.py` | `LLMError` gains `passenger_message` | 5 |
| `lib/agents/flight_agent.py` | Two raise sites gain `passenger_message` | 5 |
| `lib/agents/accommodation_agent.py` | One raise site gains `passenger_message`; the two deprecated fields are deleted | 5, 6 |
| `lib/agents/documentation_agent.py` | One line: `incident_time` reaches the rights prompt | 2 |
| `public/index.html` | Sends the browser's local wall clock | 1 |
| `scripts/run_search_agents_live.py` | Follows the `_extract_request` and `run` signature changes | 1, 8 |
| `tests/conftest.py` | The shared fake follows the prompt and payload changes | 3, 5 |
| `tests/test_supervisor.py` | Most new coverage | 1–8 |

---

### Task 1: The passenger's own clock

`local_now` is `datetime.now()` — UTC on Vercel. It is what the date sync compares, what any date
sanity check will compare, and what builds the AeroDataBox window, an API that expects
**airport-local** time. It stays naive: `flight_agent.py:154-156` subtracts it from a tz-stripped
departure, so an offset-aware value is a `TypeError` on every FlightAgent turn.

**Files:**
- Modify: `api/index.py` (new `_local_time`, and the `execute` body)
- Modify: `lib/agents/supervisor.py:164-212` (`_request_from`, `_extract_request`), `:307-317` (`run`)
- Modify: `public/index.html:947-953`
- Modify: `scripts/run_search_agents_live.py:65-70`
- Test: `tests/test_supervisor.py`, `tests/test_execute.py`, `tests/test_search_runner.py`

**Interfaces:**
- Produces: `supervisor.run(prompt, history, *, local_time: datetime | None = None) -> tuple[str, list[dict]]`
- Produces: `supervisor._extract_request(prompt, history, local_now: str) -> tuple[dict, dict]`
- Produces: `supervisor._request_from(parsed: dict, follow_up: bool, local_now: str) -> dict`
- Produces: `api.index._local_time(value) -> datetime | None`

- [ ] **Step 1: Write the failing tests**

In `tests/test_supervisor.py`, under a new `# --- the clock ---` section:

```python
def test_the_passengers_own_clock_reaches_the_agents():
    when = datetime.now().replace(hour=3, minute=33, second=7, microsecond=0)
    _, steps = supervisor.run(COMPLETE, [], local_time=when)

    flight = next(s for s in steps if s["module"] == "FlightAgent")
    assert "Local time now: " + when.isoformat(timespec="seconds") in flight["prompt"]["user_prompt"]


def test_without_a_clock_the_server_time_is_used():
    _, steps = supervisor.run(COMPLETE, [])

    flight = next(s for s in steps if s["module"] == "FlightAgent")
    assert f"Local time now: {date.today().isoformat()}" in flight["prompt"]["user_prompt"]
```

In `tests/test_execute.py`, under `# --- multi-turn ---`:

```python
def test_the_browsers_local_time_reaches_the_supervisor(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(when=local_time), ("ok", []))[1],
    )

    client.post("/api/execute", json={"prompt": COMPLETE, "local_time": "2026-03-04T05:06:07"})

    assert seen["when"] == datetime(2026, 3, 4, 5, 6, 7)


def test_an_offset_is_reduced_to_the_wall_clock_reading(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(when=local_time), ("ok", []))[1],
    )

    client.post("/api/execute", json={"prompt": COMPLETE, "local_time": "2026-03-04T05:06:07+03:00"})

    # Naive throughout: flight_agent subtracts local_now from a tz-stripped departure.
    assert seen["when"] == datetime(2026, 3, 4, 5, 6, 7)


def test_an_unusable_local_time_falls_back_to_the_server_clock(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(when=local_time), ("ok", []))[1],
    )

    client.post("/api/execute", json={"prompt": COMPLETE, "local_time": "yesterday evening"})

    assert seen["when"] is None
```

Add `from datetime import datetime` to `tests/test_execute.py` if it is not already imported, and
`date` to the existing `from datetime import ...` line in `tests/test_supervisor.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py tests/test_execute.py -q`
Expected: FAIL — `TypeError: run() got an unexpected keyword argument 'local_time'`, and the
`test_execute` cases fail because `_local_time` does not exist.

- [ ] **Step 3: Thread the clock through the Supervisor**

In `lib/agents/supervisor.py`, change three signatures. `_request_from`:

```python
def _request_from(parsed: dict, follow_up: bool, local_now: str) -> dict:
```

and replace the `local_now` line in its body (currently `supervisor.py:189`):

```python
    # The model has no clock, and the date sync runs on this. It is the passenger's
    # wall clock when their browser sent one, because the server's is UTC on Vercel.
    request["local_now"] = local_now
```

`_extract_request`:

```python
def _extract_request(prompt: str, history: list[dict], local_now: str) -> tuple[dict, dict]:
```

and its return:

```python
    return _request_from(parsed, bool(history), local_now), step
```

`run`:

```python
def run(prompt: str, history: list[dict], *,
        local_time: datetime | None = None) -> tuple[str, list[dict]]:
    """Returns (response_text, steps) — see `docs/PROJECT_PLAN.md` §1."""
    history = _trim(history)
    steps: list[dict] = []

    local_now = (local_time or datetime.now()).isoformat(timespec="seconds")
    request, refine_step = _extract_request(prompt, history, local_now)
```

- [ ] **Step 4: Accept `local_time` at the API**

In `api/index.py`, add the import and the parser:

```python
from datetime import datetime
```

```python
def _local_time(value) -> datetime | None:
    """The passenger's own wall clock, or None.

    Never raises: a device sending a malformed timestamp is not worth a failed turn,
    and the server clock is a working fallback. The offset is dropped rather than
    applied — every consumer of `local_now` wants the wall-clock reading, and
    `flight_agent` subtracts it from a tz-stripped departure.
    """
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return None
```

In `execute`, after `conversation_id = body.get("conversation_id")`:

```python
    local_time = _local_time(body.get("local_time"))
```

and change the call:

```python
        response_text, steps = supervisor.run(prompt, history, local_time=local_time)
```

- [ ] **Step 5: Send it from the GUI**

In `public/index.html`, immediately before the `fetch("/api/execute", ...)` call at line 947:

```js
        const now = new Date(), pad = n => String(n).padStart(2, "0");
        const localTime = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
          + `T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
```

and change the body:

```js
          body: JSON.stringify({ prompt, conversation_id: conversationId, local_time: localTime }),
```

- [ ] **Step 6: Follow the signature in the live runner and its test**

In `scripts/run_search_agents_live.py`, `build_request_patch`'s inner function takes the new
argument and ignores it in favour of the scenario's clock:

```python
    def patched(prompt, history, _local_now):
        request, step = original(prompt, history, when.isoformat(timespec="seconds"))
        request.update(case["request_override"])
        request["missing"] = [f for f in supervisor.REQUIRED_FIELDS if not request.get(f)]
        return request, step
```

In `tests/test_search_runner.py`, the three fake `original` functions take three arguments and every
`patched(...)` call passes a third. `patched` no longer overwrites `local_now` itself — it hands the
scenario clock to `original` — so the fake used by `test_local_now_is_injected_from_the_case` has to
honour that argument or the test asserts nothing:

```python
def complete_request(local_now="ignored"):
    return {"party_size": 1, "local_now": local_now, "flight_number": "LH318",
            "origin": "TLV", "destination": "FRA", "disruption": "cancelled",
            "stranded_at": "TLV"}
```

```python
    def original(prompt, history, local_now):
        return complete_request(local_now), {"module": "Supervisor"}

    patched = runner.build_request_patch(CASE, original)
    request, _ = patched("prompt", [], "ignored")
```

The other two fakes ignore the argument, which is correct — neither asserts on `local_now`.

Apply the same two changes to `test_request_override_wins` and
`test_missing_is_recomputed_so_the_gate_still_works`.

- [ ] **Step 7: Fix the pre-existing fake that now under-accepts**

`tests/test_execute.py:126` fakes `supervisor.run` with a two-parameter lambda, which
`api/index.py` now calls with a keyword:

```python
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(history=list(history)), ("ok", []))[1],
    )
```

- [ ] **Step 8: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no failures.

- [ ] **Step 9: Update the locked interface note**

In `docs/PROJECT_PLAN.md` §1, change the `supervisor.run` line in the signature block to:

```python
supervisor.run(prompt: str, history: list[dict], *, local_time: datetime | None = None) -> tuple[str, list[dict]]
```

and add below the block:

> `local_time` is the passenger's own wall clock, supplied by the GUI as an optional `local_time`
> on `/api/execute` and `None` for a bare `{"prompt": ...}` call, which falls back to the server
> clock. It is naive by contract: `flight_agent` subtracts `local_now` from a tz-stripped departure.

In `CLAUDE.md` §6, update the sentence that reads `Supervisor.run(prompt, history)` to name the new
keyword argument.

- [ ] **Step 10: Commit**

```bash
git add lib/agents/supervisor.py api/index.py public/index.html \
        scripts/run_search_agents_live.py tests/ docs/PROJECT_PLAN.md CLAUDE.md
git commit -m "Read the clock the passenger is actually standing under"
```

---

### Task 2: Record when the disruption happened

No field holds this today, so a correct date is discarded as silently as a wrong one — and
`DocumentationAgent` works out delay thresholds and claim deadlines without ever being told when the
flight was due.

**Files:**
- Modify: `lib/agents/supervisor.py:48-78` (`REFINE_SYSTEM_PROMPT`), `:164-191` (`_request_from`)
- Modify: `lib/agents/documentation_agent.py:122-137` (`_user_prompt`)
- Test: `tests/test_supervisor.py`, `tests/test_documentation_agent.py`

**Interfaces:**
- Consumes: `supervisor._request_from(parsed, follow_up, local_now)` from Task 1
- Produces: `request["incident_time"]` — ISO 8601 string or `None`

- [ ] **Step 1: Write the failing tests**

In `tests/test_supervisor.py`:

```python
def test_the_incident_time_survives_extraction():
    request = supervisor._request_from(
        {"incident_time": "2026-08-15T18:00"}, False, "2026-08-15T22:15:00")

    assert request["incident_time"] == "2026-08-15T18:00"


def test_an_absent_incident_time_is_none_not_missing():
    request = supervisor._request_from({}, False, "2026-08-15T22:15:00")

    # It is useful, not required: a passenger who cannot say when is still helped.
    assert request["incident_time"] is None
    assert "incident_time" not in request["missing"]
```

In `tests/test_documentation_agent.py`:

```python
def test_the_incident_time_reaches_the_rights_prompt():
    prompt = documentation_agent._user_prompt({"incident_time": "2026-08-15T18:00"}, [])

    # Delay thresholds and claim deadlines both run off when the flight was due.
    assert "2026-08-15T18:00" in prompt
```

Check the existing imports at the top of `tests/test_documentation_agent.py`; add
`from lib.agents import documentation_agent` if it is not already there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py -k incident tests/test_documentation_agent.py -k incident -q`
Expected: FAIL — `KeyError: 'incident_time'`, and the string is absent from the rights prompt.

- [ ] **Step 3: Extract it**

In `lib/agents/supervisor.py`, add `"incident_time"` to the tuple `_request_from` iterates over:

```python
    request = {
        field: _text(parsed.get(field))
        for field in (
            "airline", "flight_number", "origin", "destination", "stranded_at",
            "arrive_by", "incident_time",
        )
    }
```

In `REFINE_SYSTEM_PROMPT`, add `"incident_time"` to the JSON shape line:

```
{"airline", "flight_number", "origin", "destination",
 "disruption": "delayed" | "cancelled" | "denied_boarding" | null,
 "stranded_at", "party_size", "arrive_by", "incident_time",
 "needs": ["flight", "stay", "rights"],
 "missing": [field names]}
```

and add this paragraph after the `"stranded_at"` paragraph:

```
"incident_time" is when the disrupted flight was scheduled to leave, in ISO 8601. Give it whenever
the passenger says or implies it, resolving relative words against the current date and time you are
given. Leave it null if they did not say - it is useful, not required.
```

- [ ] **Step 4: Pass it to DocumentationAgent**

In `lib/agents/documentation_agent.py`, add one line to `_user_prompt`'s `lines` list, directly
after the `What happened:` line:

```python
        f"Flight was due to leave: {request.get('incident_time') or 'not stated'}",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add lib/agents/supervisor.py lib/agents/documentation_agent.py tests/
git commit -m "Keep the one date the whole plan depends on"
```

---

### Task 3: Detect what cannot be true

Two sources, merged. The model sees the contradictions only it can see — "today, 11th of November"
read against a date that is not today. Python re-checks the arithmetic, because that is the part a
model gets quietly wrong, and reuses `flights.route_problem`, which already catches
`origin == destination` and unknown IATA codes in passenger-ready wording. This task produces the
data only; Task 4 acts on it.

**Files:**
- Modify: `lib/agents/supervisor.py` (imports, new constants, new `_when`/`_conflicts`, `_request_from`, `REFINE_SYSTEM_PROMPT`)
- Modify: `tests/conftest.py:89-99` (`_refine_response`)
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `request["incident_time"]` from Task 2
- Produces: `supervisor._conflicts(request: dict) -> list[dict]`, entries `{"field", "stated", "reason"}`
- Produces: `request["conflicts"]` — deterministic entries first, model-reported entries appended for fields the deterministic pass did not already claim
- Produces: `supervisor.INCIDENT_FUTURE_LIMIT_DAYS = 30`, `supervisor.INCIDENT_PAST_LIMIT_DAYS = 14`

- [ ] **Step 1: Write the failing tests**

In `tests/test_supervisor.py`, under a new `# --- sanity checks ---` section:

```python
def request_stating(**fields) -> dict:
    base = {"local_now": "2026-08-15T22:15:00", "origin": "TLV", "destination": "FRA",
            "stranded_at": "TLV", "incident_time": None, "arrive_by": None}
    return {**base, **fields}


def fields_of(conflicts):
    return [c["field"] for c in conflicts]


def test_a_date_months_away_is_a_conflict():
    # The motivating case: "today, 11th of November" read on 15 August.
    found = supervisor._conflicts(request_stating(incident_time="2026-11-11T09:00"))

    assert fields_of(found) == ["incident_time"]
    assert found[0]["stated"] == "2026-11-11T09:00"


def test_a_flight_cancelled_a_fortnight_ahead_is_not_a_conflict():
    # Airlines cancel in advance. Flagging that would bounce a real passenger.
    assert supervisor._conflicts(request_stating(incident_time="2026-08-29T09:00")) == []


def test_a_disruption_from_last_month_is_a_conflict():
    # Still a valid claim, but "a bed tonight" for it is nonsense.
    found = supervisor._conflicts(request_stating(incident_time="2026-07-04T09:00"))

    assert fields_of(found) == ["incident_time"]


def test_an_unparseable_incident_time_is_left_alone():
    # The model's job, not arithmetic's.
    assert supervisor._conflicts(request_stating(incident_time="last Tuesday")) == []


def test_a_route_that_cannot_exist_is_a_conflict():
    found = supervisor._conflicts(request_stating(origin="TLV", destination="TLV"))

    assert fields_of(found) == ["route"]
    assert "no flight to look for" in found[0]["reason"]


def test_a_half_stated_route_is_left_to_the_missing_fields_gate():
    # Otherwise the passenger is asked the same thing twice, differently worded.
    assert supervisor._conflicts(request_stating(origin="TLV", destination=None)) == []


def test_an_unknown_airport_is_a_conflict():
    found = supervisor._conflicts(request_stating(stranded_at="ZZZ"))

    assert fields_of(found) == ["stranded_at"]


def test_a_deadline_already_past_is_a_conflict():
    found = supervisor._conflicts(request_stating(arrive_by="2026-08-14T09:00"))

    assert fields_of(found) == ["arrive_by"]


def test_a_clean_request_has_no_conflicts():
    assert supervisor._conflicts(request_stating()) == []


def test_the_models_conflicts_are_kept_alongside_the_checked_ones():
    parsed = {"origin": "TLV", "destination": "TLV", "stranded_at": "TLV",
              "conflicts": [{"field": "airline", "stated": "Lufthansa",
                             "reason": "the flight number LY357 is El Al"}]}
    request = supervisor._request_from(parsed, False, "2026-08-15T22:15:00")

    assert fields_of(request["conflicts"]) == ["route", "airline"]


def test_the_checked_conflict_wins_when_both_name_the_same_field():
    parsed = {"origin": "TLV", "destination": "TLV", "stranded_at": "TLV",
              "conflicts": [{"field": "route", "stated": "TLV to TLV",
                             "reason": "made up by the model"}]}
    request = supervisor._request_from(parsed, False, "2026-08-15T22:15:00")

    assert fields_of(request["conflicts"]) == ["route"]
    assert "made up by the model" not in request["conflicts"][0]["reason"]


def test_junk_in_the_models_conflicts_is_dropped():
    parsed = {"origin": "TLV", "destination": "FRA", "stranded_at": "TLV",
              "conflicts": ["not an object", {"field": "airline"}, {"reason": ""}]}
    request = supervisor._request_from(parsed, False, "2026-08-15T22:15:00")

    assert request["conflicts"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py -q`
Expected: FAIL — `AttributeError: module 'lib.agents.supervisor' has no attribute '_conflicts'`.

- [ ] **Step 3: Add the deterministic checks**

In `lib/agents/supervisor.py`, extend the imports:

```python
from lib.tools import airports, flights
```

Add the constants near `REQUIRED_FIELDS`:

```python
# Generous on purpose: airlines cancel a fortnight ahead, and flagging that would bounce
# a real passenger. It still catches "today, 11th of November" read in August.
INCIDENT_FUTURE_LIMIT_DAYS = 30
# Not about whether the claim is valid — it is, for months — but about whether a plan
# that finds a bed tonight makes any sense for it.
INCIDENT_PAST_LIMIT_DAYS = 14
```

Add the two functions after `_party_size`:

```python
def _when(value) -> datetime | None:
    """A stated time as a naive datetime, or None if it is absent or unparseable."""
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _conflicts(request: dict) -> list[dict]:
    """What the passenger stated that cannot be true, from arithmetic alone.

    The model reports the contradictions only it can see — a relative word next to a
    date that is not that day. This reports the ones it gets quietly wrong. Both lists
    are merged in `_request_from`, with these winning on any shared field.
    """
    now = _when(request.get("local_now"))
    found: list[dict] = []

    incident = _when(request.get("incident_time"))
    if now and incident:
        days = (incident.date() - now.date()).days
        if days > INCIDENT_FUTURE_LIMIT_DAYS:
            found.append({"field": "incident_time", "stated": request["incident_time"],
                          "reason": f"that is {days} days from now"})
        elif -days > INCIDENT_PAST_LIMIT_DAYS:
            found.append({"field": "incident_time", "stated": request["incident_time"],
                          "reason": f"that was {-days} days ago"})

    origin, destination = request.get("origin"), request.get("destination")
    # Only when both are present: a half-stated route is already a missing field, and
    # asking about it twice in two different wordings helps nobody.
    if origin and destination:
        problem = flights.route_problem(origin, destination)
        if problem:
            found.append({"field": "route", "stated": f"{origin} to {destination}",
                          "reason": problem})

    stranded = request.get("stranded_at")
    if stranded and not airports.lookup(stranded):
        found.append({"field": "stranded_at", "stated": stranded,
                      "reason": "that is not an airport code I can look up"})

    arrive_by = _when(request.get("arrive_by"))
    if now and arrive_by and arrive_by < now:
        found.append({"field": "arrive_by", "stated": request["arrive_by"],
                      "reason": "that deadline has already passed"})

    return found
```

- [ ] **Step 4: Merge the model's list in**

In `_request_from`, immediately before the `request["missing"] = ...` line:

```python
    reported = parsed.get("conflicts")
    from_model = [
        {"field": _text(c.get("field")) or "", "stated": _text(c.get("stated")) or "",
         "reason": _text(c.get("reason")) or ""}
        for c in (reported if isinstance(reported, list) else [])
        if isinstance(c, dict) and _text(c.get("field")) and _text(c.get("reason"))
    ]
    checked = _conflicts(request)
    claimed = {c["field"] for c in checked}
    request["conflicts"] = checked + [c for c in from_model if c["field"] not in claimed]
```

- [ ] **Step 5: Ask the model for them**

In `REFINE_SYSTEM_PROMPT`, add `"conflicts"` to the JSON shape line:

```
 "needs": ["flight", "stay", "rights"],
 "conflicts": [{"field", "stated", "reason"}],
 "missing": [field names]}
```

and add this paragraph after the `"incident_time"` paragraph from Task 2:

```
"conflicts" is for anything the passenger stated that cannot be true. You are given the current
local date and time below. If they write a relative word - "today", "this morning", "tonight" -
next to a date that is not that day, that is a conflict on "incident_time". So is an airline that
does not match the flight number they gave, or an airport that contradicts the route they described.
Give {"field": which one, "stated": what they wrote, "reason": why it cannot be right}. Report only
what you are sure of, and leave it empty otherwise; date arithmetic is re-checked separately.
```

- [ ] **Step 6: Update the shared fake**

In `tests/conftest.py`, add two keys to the dict `_refine_response` returns:

```python
        "arrive_by": None,
        "incident_time": None,
        "conflicts": [],
        "needs": needs,
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add lib/agents/supervisor.py tests/
git commit -m "Notice when the passenger's own account cannot be true"
```

---

### Task 4: Ask about the ones that matter, assume the rest

A wrong incident date corrupts the flight window and the entitlement clock alike, so it is worth a
round trip. An airline name that disagrees with a flight number is not — bouncing a stressed
passenger over that is the friction this product exists to remove.

**Files:**
- Modify: `lib/agents/supervisor.py` (new constants, `_sentence`, `_conflict_question`, `_request_from`, `run`'s gate, `_compose_prompt`, `COMPOSE_SYSTEM_PROMPT`)
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `request["conflicts"]` from Task 3
- Produces: `supervisor.BLOCKING_CONFLICTS = {"incident_time", "route", "stranded_at"}`
- Produces: `supervisor.CLEARED_ON_CONFLICT = {"arrive_by"}`
- Produces: `supervisor._sentence(text: str) -> str` — capitalises the first character, used again in Task 5
- Produces: `request["assumptions"]` — `list[str]`, one per non-blocking conflict

- [ ] **Step 1: Write the failing tests**

In `tests/test_supervisor.py`, under the `# --- sanity checks ---` section:

```python
IMPOSSIBLE_ROUTE = "LH318 TLV -> TLV was cancelled at the gate"


def test_a_blocking_conflict_asks_instead_of_dispatching():
    text, steps = supervisor.run(IMPOSSIBLE_ROUTE, [])

    assert "no flight to look for" in text
    # Same economics as the missing-fields gate: one call, not seven.
    assert modules_of(steps) == ["Supervisor"]


def test_the_conflict_question_reads_as_a_sentence():
    text, _ = supervisor.run(IMPOSSIBLE_ROUTE, [])

    # route_problem's reasons start lower-case because they are interpolated after
    # "FlightAgent: " today.
    assert "The origin and destination are both TLV" in text


def test_a_blocking_conflict_with_nothing_to_dispatch_does_not_interrogate():
    history = [{"prompt": IMPOSSIBLE_ROUTE, "response": "The origin and destination are both TLV"}]
    text, steps = supervisor.run("never mind, thanks", history)

    assert modules_of(steps) == ["Supervisor", "Supervisor"]
    assert "Before I can help" not in text


def test_a_soft_conflict_proceeds_and_states_the_assumption():
    request = supervisor._request_from(
        {"origin": "TLV", "destination": "FRA", "stranded_at": "TLV",
         "arrive_by": "2026-08-14T09:00"},
        False, "2026-08-15T22:15:00")

    assert request["assumptions"] == [
        "They said 2026-08-14T09:00, but that deadline has already passed. Working without it."
    ]


def test_a_field_that_cannot_be_true_never_reaches_an_agent():
    request = supervisor._request_from(
        {"origin": "TLV", "destination": "FRA", "stranded_at": "TLV",
         "arrive_by": "2026-08-14T09:00"},
        False, "2026-08-15T22:15:00")

    # An impossible deadline handed to FlightAgent is worse than no deadline.
    assert request["arrive_by"] is None


def test_a_blocking_conflict_produces_no_assumption():
    request = supervisor._request_from(
        {"origin": "TLV", "destination": "TLV", "stranded_at": "TLV"},
        False, "2026-08-15T22:15:00")

    # It is being asked about, not assumed around.
    assert request["assumptions"] == []


def test_assumptions_reach_the_composing_call():
    request = {"assumptions": ["They said X, but Y."], "local_now": "2026-08-15T22:15:00"}
    prompt = supervisor._compose_prompt(request, "digest", [])

    assert "They said X, but Y." in prompt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py -q`
Expected: FAIL — the impossible route dispatches all four modules, and `KeyError: 'assumptions'`.

- [ ] **Step 3: Add the severity split and the question wording**

In `lib/agents/supervisor.py`, add after `QUESTIONS`:

```python
# A wrong date corrupts the flight window and the entitlement clock alike, so it is worth
# a round trip. An airline name that disagrees with a flight number is not.
BLOCKING_CONFLICTS = {"incident_time", "route", "stranded_at"}

# Not worth asking about, but a field that cannot be true must not reach an agent's prompt
# either. Anything listed here is cleared, and the assumption is stated in the plan.
CLEARED_ON_CONFLICT = {"arrive_by"}

CONFLICT_QUESTIONS = {
    "incident_time": "You said the flight was on {stated}, but right now it is {now}. Which is right?",
    "stranded_at": "I could not find an airport with the code {stated}. Which airport are you at?",
}
```

and the two helpers, after `_conflicts`:

```python
def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _conflict_question(conflict: dict, now: str) -> str:
    template = CONFLICT_QUESTIONS.get(conflict["field"])
    if template:
        return template.format(stated=conflict["stated"], now=now)
    if conflict["field"] == "route":
        # route_problem already writes a passenger-ready sentence. It starts lower-case
        # because today it is interpolated after "FlightAgent: ".
        return _sentence(conflict["reason"])
    return f"You said {conflict['stated']} — {conflict['reason']}. Which is right?"
```

- [ ] **Step 4: Record the assumptions**

In `_request_from`, immediately after the `request["conflicts"] = ...` line from Task 3:

```python
    request["assumptions"] = []
    for conflict in request["conflicts"]:
        if conflict["field"] in BLOCKING_CONFLICTS:
            continue
        note = f"They said {conflict['stated']}, but {conflict['reason']}."
        if conflict["field"] in CLEARED_ON_CONFLICT:
            request[conflict["field"]] = None
            note += " Working without it."
        request["assumptions"].append(note)
```

- [ ] **Step 5: Widen the gate**

In `run`, replace the gate block (currently `supervisor.py:321-329`):

```python
    blocking = [c for c in request["conflicts"] if c["field"] in BLOCKING_CONFLICTS]

    # The gate: never dispatch against a request that is missing what the crew needs, or
    # that states something that cannot be true. Nothing to dispatch means nothing is
    # blocked, so a follow-up answered from the conversation is never interrogated for
    # details it already gave.
    if (request["missing"] or blocking) and needs:
        asked = list(dict.fromkeys(
            [_conflict_question(c, request["local_now"]) for c in blocking]
            + [QUESTIONS[f] for f in request["missing"]]
        ))
        opener = ("Before I can help, I need to check a couple of things:" if blocking
                  else "Before I can help, I need a couple of details:")
        return opener + "\n" + "\n".join(f"  - {q}" for q in asked), steps
```

- [ ] **Step 6: Carry the assumptions into composition**

In `_compose_prompt`, after the `block = _history_block(history)` stanza:

```python
    if request.get("assumptions"):
        lines += ["", "Assumptions you must state in the plan:"]
        lines += [f"  - {a}" for a in request["assumptions"]]
```

and add one line to `COMPOSE_SYSTEM_PROMPT`, before the closing paragraph:

```
State any assumption you are given, in your own words, so the passenger can correct it.
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. `test_underspecified_message_asks_instead_of_dispatching` still passes because the
missing-fields opener is unchanged.

- [ ] **Step 8: Record the request shape**

In `docs/PROJECT_PLAN.md` §1, replace the `request` block with:

```python
{"airline", "flight_number", "origin", "destination",
 "disruption": "delayed" | "cancelled" | "denied_boarding",
 "stranded_at", "party_size", "arrive_by": iso8601 | None,
 "incident_time": iso8601 | None,
 "conflicts": [{"field", "stated", "reason"}],
 "assumptions": [str],
 "needs": ["flight", "stay", "rights"], "local_now": iso8601, "missing": [field names]}
```

and add below it:

> `conflicts` is what the passenger stated that cannot be true — merged from the refinement call and
> a deterministic re-check. Fields in `supervisor.BLOCKING_CONFLICTS` stop the dispatch and are asked
> about; the rest become `assumptions`, which the plan states out loud.

- [ ] **Step 9: Commit**

```bash
git add lib/agents/supervisor.py tests/ docs/PROJECT_PLAN.md
git commit -m "Ask about a date that cannot be right before spending on it"
```

---

### Task 5: One assistant, not a crew

`COMPOSE_SYSTEM_PROMPT` says "the results the crew returned", "a crew result", "part of the crew
failed", and the compose prompt is headed "What the crew came back with:". The model echoes what we
wrote. It is also never told who it is. Separately, `dispatch` interpolates the raw exception into
passenger text, so `RuntimeError("Pinecone unreachable")` is something a stranded passenger reads —
while Person B's carefully written refusal text is the one thing worth keeping.

Internal comments and docstrings that say "crew" are left alone. They are not prompts, and churning
them is not this task.

**Files:**
- Modify: `lib/llm.py:37-49` (`LLMError.__init__`)
- Modify: `lib/agents/flight_agent.py:212`, `:224`
- Modify: `lib/agents/accommodation_agent.py:200`
- Modify: `lib/agents/supervisor.py` (`COMPOSE_SYSTEM_PROMPT`, new `FAILURE_MESSAGES`, `dispatch`, `_digest`'s failure lines, `_compose_prompt`)
- Modify: `tests/conftest.py:102-112`
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `supervisor._sentence` from Task 4
- Produces: `llm.LLMError(message, steps=None, passenger_message=None)` with `.passenger_message`
- Produces: `supervisor.FAILURE_MESSAGES` keyed by `"flight" | "stay" | "rights"`
- Produces: `dispatch(key, fn, *args)` — the `label` parameter is gone; `failures` is now a list of finished passenger sentences

- [ ] **Step 1: Write the failing tests**

Replace `test_one_agent_failing_does_not_lose_the_rest_of_the_plan` in `tests/test_supervisor.py`
with:

```python
def test_one_agent_failing_does_not_lose_the_rest_of_the_plan(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("Pinecone unreachable")

    monkeypatch.setattr(documentation_agent, "run", explode)
    text, steps = supervisor.run(COMPLETE, [])

    assert "Onward flight" in text or "ONWARD FLIGHT" in text
    assert supervisor.FAILURE_MESSAGES["rights"] in text
    # The cause is for us, not for a passenger standing at a gate.
    assert "Pinecone" not in text
    assert "DocumentationAgent" not in text
```

and add:

```python
def test_an_agents_own_refusal_wording_survives(monkeypatch):
    from lib.agents import flight_agent
    from lib.llm import LLMError

    def refuse(*args, **kwargs):
        raise LLMError("FlightAgent: internal wording",
                       steps=[], passenger_message=flight_agent.NO_LIVE_DATA)

    monkeypatch.setattr(flight_agent, "run", refuse)
    text, _ = supervisor.run(COMPLETE, [])

    assert "check your airline's app or the departures board" in text
    assert "internal wording" not in text
    assert "FlightAgent" not in text


def test_an_internal_failure_falls_back_to_a_written_sentence(monkeypatch):
    from lib.agents import flight_agent
    from lib.llm import LLMError

    def refuse(*args, **kwargs):
        raise LLMError("FlightAgent: no option named a flight that was actually offered", steps=[])

    monkeypatch.setattr(flight_agent, "run", refuse)
    text, _ = supervisor.run(COMPLETE, [])

    assert supervisor.FAILURE_MESSAGES["flight"] in text
    assert "no option named a flight" not in text


def test_the_composing_prompt_never_names_the_inside_of_the_system():
    _, steps = supervisor.run(COMPLETE, [])
    compose = [s for s in steps if s["module"] == "Supervisor"][-1]["prompt"]

    banned = ("crew", "FlightAgent", "AccommodationAgent", "DocumentationAgent", "agent")
    haystack = (compose["system_prompt"] + compose["user_prompt"]).lower()
    for word in banned:
        assert word.lower() not in haystack, f"{word!r} leaks into the composing call"


def test_the_composing_prompt_says_who_it_is():
    _, steps = supervisor.run(COMPLETE, [])
    compose = [s for s in steps if s["module"] == "Supervisor"][-1]["prompt"]

    assert "Wingman" in compose["system_prompt"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'FAILURE_MESSAGES'`, and the banned-word
assertion trips on "crew".

- [ ] **Step 3: Let an exception carry passenger-facing text**

In `lib/llm.py`, replace `LLMError.__init__`:

```python
    def __init__(self, message: str, steps: list[dict] | None = None,
                 passenger_message: str | None = None):
        super().__init__(message)
        self.steps = steps or []
        # Set only where the wording was written for the passenger. The same exception
        # also carries internal text ("no option named a flight that was actually
        # offered"), and nothing else tells the two apart.
        self.passenger_message = passenger_message
```

In `lib/agents/flight_agent.py`, the two refusals:

```python
    if problem:
        raise llm.LLMError(f"{MODULE}: {problem}", steps=[], passenger_message=problem)
```

```python
        raise llm.LLMError(f"{MODULE}: {NO_LIVE_DATA}", steps=[], passenger_message=NO_LIVE_DATA)
```

In `lib/agents/accommodation_agent.py`:

```python
        raise llm.LLMError(f"{MODULE}: {NO_LIVE_DATA}", steps=[], passenger_message=NO_LIVE_DATA)
```

- [ ] **Step 4: Stop printing exceptions**

In `lib/agents/supervisor.py`, add the import and the constant:

```python
import logging
```

```python
logger = logging.getLogger(__name__)

# What the passenger reads when an agent could not finish and wrote nothing better.
FAILURE_MESSAGES = {
    "flight": "I could not get onward flight options just now.",
    "stay": "I could not find somewhere to stay just now.",
    "rights": "I could not work out what you are owed just now.",
}
```

Replace `dispatch` inside `run`:

```python
    # One agent failing must not cost the passenger the rest of the plan.
    def dispatch(key: str, fn, *args):
        try:
            payload, agent_steps = fn(*args, history)
            results[key] = payload
            steps.extend(agent_steps)
        except Exception as exc:
            # A call that was made and then failed is still a call, and the spec wants
            # every one of them in `steps`. lib.llm.LLMError carries the failed call's
            # step; an agent that got partway through several calls attaches the ones
            # that already succeeded (see documentation_agent).
            steps.extend(getattr(exc, "steps", []))
            # The cause is ours, not the passenger's. `steps` keeps it for the trace and
            # the log keeps it for the failures that carry no step at all.
            logger.exception("%s could not be completed", key)
            reason = getattr(exc, "passenger_message", None) or FAILURE_MESSAGES[key]
            failures.append(_sentence(reason))
```

and drop the first argument from the three call sites:

```python
    if "flight" in needs:
        dispatch("flight", flight_agent.run, request)
```
```python
            dispatch("stay", accommodation_agent.run, request, stay_window)
```
```python
    if "rights" in needs:
        dispatch("rights", documentation_agent.run, request)
```

In `_digest`, replace the failure loop:

```python
    for failure in failures:
        lines.append(failure)
```

- [ ] **Step 5: Rewrite the voice**

Replace `COMPOSE_SYSTEM_PROMPT` entirely:

```python
COMPOSE_SYSTEM_PROMPT = """You are Wingman, writing directly to one passenger whose flight has just been disrupted.

You are a single assistant. Never mention how you work: no tools, no searches, no internal steps, and
never any suggestion that the work was divided up or handed to anyone. The passenger is talking to
you, not to a system.

Speak to one tired person, not to a user. Short sentences, active voice, no jargon and no
regulation-speak they would have to decode. Lead with the flight, then where they sleep, then what
they are owed and what to do about it - that is the order they need it in.

State amounts and entitlements only where the findings support them, and say which document each came
from. Where something could not be finished, say plainly what is missing rather than papering over it.

State any assumption you are given, in your own words, so the passenger can correct it.

End by inviting a follow-up: the passenger can compare options or ask about the terms of any one of
them.
"""
```

In `_compose_prompt`, change the last line:

```python
    lines += ["", "Findings:", digest]
```

- [ ] **Step 6: Fix the two existing tests that assert on the old failure wording**

`dispatch` no longer takes a label, so `"Could not complete: onward flights"` no longer exists. In
`tests/test_supervisor.py`:

```python
def test_a_half_labelled_option_still_produces_a_plan(monkeypatch):
    ...
    assert "Onward flight" in text or "ONWARD FLIGHT" in text
    assert supervisor.FAILURE_MESSAGES["flight"] not in text
```

```python
def test_a_failing_flight_search_skips_the_stay_rather_than_guessing_nights(monkeypatch):
    ...
    assert "AccommodationAgent" not in modules_of(steps)
    assert supervisor.FAILURE_MESSAGES["flight"] in text
    assert "DocumentationAgent" in modules_of(steps)  # rights are independent
```

- [ ] **Step 7: Update the shared fake's marker**

In `tests/conftest.py`, three occurrences of the old header:

```python
def _compose_response(user_prompt: str) -> str:
    """The fake composer hands the findings straight back. The real model rewrites them
    as prose for one tired person; the tests assert on the facts, not the prose."""
    return user_prompt.rpartition("Findings:")[2].strip()


def _supervisor_response(user_prompt: str):
    """Both Supervisor calls arrive here: the composing one is the one carrying results."""
    if "Findings:" in user_prompt:
        return _compose_response(user_prompt)
    return _refine_response(user_prompt)
```

And in `tests/test_supervisor.py`, `test_a_failed_composing_call_still_hands_over_the_plan` matches
on the same string:

```python
        if "Findings:" in user_prompt:
```

- [ ] **Step 8: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add lib/llm.py lib/agents/ tests/
git commit -m "Speak as one assistant, and stop handing over stack traces"
```

---

### Task 6: Read everything the agents now return

`_digest` forwards the recommended option only, and of that a handful of fields — so the composing
call is never shown a second option to compare, though comparison is what `agent_info.description`
promises. It also reads `meals_included`, which is `False` when the truth is `meals: "unknown"`: the
plan asserts "no meals" as a fact today. This is the contract in `PROJECT_PLAN.md:97-119`, and
finishing it lets Person B delete the two compatibility fields they are holding open.

**Files:**
- Modify: `lib/agents/supervisor.py` (new `MAX_DIGEST_OPTIONS`, `MEALS_TEXT`, `_options`, `_flight_line`, `_stay_line`; rewrite `_digest`)
- Modify: `lib/agents/accommodation_agent.py:104-109` (delete `_area_text`), `:112-139` (delete two keys)
- Modify: `tests/test_accommodation_agent.py:80-86`, `:129`, `:137`
- Modify: `tests/test_search_artifact.py:57`
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Produces: `supervisor.MAX_DIGEST_OPTIONS = 3`
- Produces: `supervisor._options(payload: dict | None) -> list[dict]` — recommended first, capped; reused by Task 8
- Produces: `supervisor._flight_line(option: dict) -> str`, `supervisor._stay_line(option: dict) -> str`

- [ ] **Step 1: Write the failing tests**

First give the shared fake a second option to work with. In `tests/conftest.py`, add to
`fake_search_data`'s flight list a second candidate, and to `_flight_response` a second option:

```python
    monkeypatch.setattr(flights, "search", lambda *a, **k: [{
        "flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH",
        "origin": "TLV", "destination": "FRA",
        "depart": depart.isoformat(),
        "arrive": (depart + timedelta(hours=3, minutes=25)).isoformat(),
        "status": "Expected", "aircraft": "Airbus A320", "terminal": "3",
    }, {
        "flight": "LY 357", "airline": "El Al", "airline_iata": "LY",
        "origin": "TLV", "destination": "FRA",
        "depart": (depart - timedelta(hours=3)).isoformat(),
        "arrive": (depart + timedelta(minutes=25)).isoformat(),
        "status": "Expected", "aircraft": "Boeing 737-900", "terminal": "3",
    }])
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [{
        "name": "Airport Plaza", "distance_km": 2.4, "stars": "4",
        "breakfast": None, "area": "Lod", "phone": "+972 3 000 0000",
        "address": "12 HaNasi", "wheelchair": "yes",
    }])
```

```python
def _flight_response(user_prompt: str) -> dict:
    """Only the model's half: the agent fills the facts from the candidate.

    The flight numbers must match fake_search_data's candidates, which is why this
    fake got shorter rather than longer when the payload grew.
    """
    return {
        "options": [
            {"id": "F1", "flight_number": "LH 687",
             "rebooking": "Rebooking terms come from your Contract of Carriage.",
             "notes": "Earliest nonstop."},
            {"id": "F2", "flight_number": "LY 357",
             "rebooking": "A different airline from the one that cancelled.",
             "notes": "Leaves earlier, lands earlier."},
        ],
        "recommended_id": "F1",
        "caveats": [],
    }
```

Then, in `tests/test_supervisor.py`, replace the `# --- the partial-failure policy ---` section's
`"Onward flight"` assertions' expectations by adding a new `# --- the digest ---` section:

```python
def digest_of(text_and_steps):
    return text_and_steps[0]


def test_every_option_reaches_the_plan_not_just_the_recommended_one():
    text, _ = supervisor.run(COMPLETE, [])

    # agent_info promises the passenger can compare. The composing call has to see
    # more than one option for that sentence to be true.
    assert "LH 687" in text
    assert "LY 357" in text


def test_the_recommended_option_comes_first():
    text, _ = supervisor.run(COMPLETE, [])

    assert text.index("LH 687") < text.index("LY 357")


def test_the_new_flight_facts_reach_the_plan():
    text, _ = supervisor.run(COMPLETE, [])

    assert "terminal 3" in text
    assert "Airbus A320" in text
    assert "Expected" in text


def test_the_hotels_phone_number_reaches_the_plan():
    text, _ = supervisor.run(COMPLETE, [])

    # The one job neither agent can do: confirming a room and a rate.
    assert "+972 3 000 0000" in text


def test_the_distance_reaches_the_plan_from_distance_km():
    text, _ = supervisor.run(COMPLETE, [])

    assert "2.4 km from the terminal" in text


def test_unknown_meals_are_not_asserted_as_absent():
    text, _ = supervisor.run(COMPLETE, [])

    # `meals_included: false` was a lie dressed as data - we almost never know.
    assert "meals not confirmed" in text
    assert "no meals" not in text


def test_the_deprecated_fields_are_read_nowhere():
    import inspect

    source = inspect.getsource(supervisor)
    assert "meals_included" not in source
    assert '"area"' not in source and "'area'" not in source


def test_the_options_shown_are_capped():
    payload = {"options": [{"id": f"F{n}"} for n in range(9)], "recommended_id": "F4"}

    shown = supervisor._options(payload)

    assert len(shown) == supervisor.MAX_DIGEST_OPTIONS
    assert shown[0]["id"] == "F4"


def test_no_payload_yields_no_options():
    assert supervisor._options(None) == []
    assert supervisor._options({"options": []}) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py -q`
Expected: FAIL — `AttributeError: ... has no attribute '_options'`, "LY 357" absent, "no meals"
present.

- [ ] **Step 3: Add the option helpers**

In `lib/agents/supervisor.py`, add near `HISTORY_TURNS`:

```python
# Three each is what the agents already return at most, and the digest is re-sent to the
# composing call on every turn — this is the cap that keeps that honest.
MAX_DIGEST_OPTIONS = 3

MEALS_TEXT = {
    "included": "meals included",
    "not_included": "no meals",
    "unknown": "meals not confirmed",
}
```

and the three functions, replacing nothing (add them above `_digest`):

```python
def _options(payload: dict | None) -> list[dict]:
    """Every option a payload holds, recommended first, capped."""
    if not payload:
        return []
    options = [o for o in (payload.get("options") or []) if isinstance(o, dict)]
    wanted = payload.get("recommended_id")
    options.sort(key=lambda o: o.get("id") != wanted)
    return options[:MAX_DIGEST_OPTIONS]


def _flight_line(option: dict) -> str:
    """One flight, built only from the fields that actually came back.

    Nothing is indexed with []: the agent's validation guarantees an id and a real
    candidate behind it, not that every optional field survived.
    """
    parts = [
        " ".join(p for p in (option.get("flight_number"), option.get("airline")) if p),
        " to ".join(p for p in (option.get("origin"), option.get("destination")) if p),
        f"departs {option['depart']}" if option.get("depart") else "",
        f"terminal {option['terminal']}" if option.get("terminal") else "",
        f"arrives {option['arrive']}" if option.get("arrive") else "",
        f"{option['duration_minutes']} minutes in the air" if option.get("duration_minutes") else "",
        "arrives the day after it leaves" if option.get("arrives_next_day") else "",
        option.get("aircraft") or "",
        f"status {option['status']}" if option.get("status") else "",
    ]
    return ", ".join(p for p in parts if p) + "."


def _stay_line(option: dict) -> str:
    """One stay. `distance_km` and `meals` replace the deprecated `area`/`meals_included`."""
    distance = option.get("distance_km")
    parts = [
        option.get("name") or "",
        f"a {option['kind']}" if option.get("kind") else "",
        f"{distance} km from the terminal" if distance is not None else "",
        option.get("city") or "",
        option.get("address") or "",
        f"phone {option['phone']}" if option.get("phone") else "",
        option.get("website") or "",
        f"{option['stars']} stars" if option.get("stars") else "",
        f"step-free access: {option['wheelchair']}" if option.get("wheelchair") else "",
        " to ".join(p for p in (option.get("check_in"), option.get("check_out")) if p),
        f"{option['nights']} night(s)" if option.get("nights") else "",
        MEALS_TEXT.get(option.get("meals"), MEALS_TEXT["unknown"]),
        option.get("price_estimate") or "",
    ]
    return ", ".join(p for p in parts if p) + "."
```

- [ ] **Step 4: Rewrite the digest**

Replace `_digest` and delete the now-unused `_line` helper:

```python
def _digest(results: dict, failures: list[str]) -> str:
    """What was found, as flat lines.

    Both the composing call's input and — if that call is the one that fails — the plan
    the passenger gets instead, so the headings stay plain English. Nothing is indexed
    with []: a model that left a field out must not cost the passenger a plan that six
    calls already paid for.
    """
    lines: list[str] = []

    onward = _options(results.get("flight"))
    if onward:
        lines.append("ONWARD FLIGHT")
        lines.append(f"  Recommended: {_flight_line(onward[0])}")
        for extra in ("rebooking", "notes"):
            if onward[0].get(extra):
                lines.append(f"    {onward[0][extra]}")
        if len(onward) > 1:
            lines.append("  Also available:")
            lines += [f"    {_flight_line(o)}" for o in onward[1:]]
        lines.append("")

    stays = _options(results.get("stay"))
    if stays:
        lines.append("SOMEWHERE TO SLEEP")
        lines.append(f"  Recommended: {_stay_line(stays[0])}")
        if stays[0].get("notes"):
            lines.append(f"    {stays[0]['notes']}")
        if len(stays) > 1:
            lines.append("  Also available:")
            lines += [f"    {_stay_line(o)}" for o in stays[1:]]
        lines.append("")

    rights = results.get("rights")
    if rights:
        lines.append(f"WHAT YOU ARE OWED (under {rights.get('regulation')})")
        for item in rights.get("entitlements") or []:
            source, confidence = item.get("source"), item.get("confidence")
            lines.append(
                f"  - {item.get('kind')}: {item.get('summary')}"
                + (f" [{source}]" if source else "")
                + (f" ({confidence} confidence)" if confidence else "")
            )
        for action in rights.get("next_actions") or []:
            lines.append(f"  Next: {action}")
        # These carry no NOTE:/ASK:/CONFIRM: prefix and are not that protocol — they are
        # gaps in the evidence the reflection loop found, and belong in their own block.
        for caveat in rights.get("caveats") or []:
            lines.append(f"  Not established from the sources: {caveat}")
        lines.append("")

    if failures:
        lines.append("COULD NOT COMPLETE")
        lines += [f"  - {failure}" for failure in failures]
        lines.append("")

    lines.append("Ask me to compare any of these, or about the terms of one in particular.")
    return "\n".join(lines)
```

- [ ] **Step 5: Delete the compatibility fields Person B left open**

In `lib/agents/accommodation_agent.py`, delete `_area_text` entirely, and delete two keys from
`_enrich`'s dict — the `"area": _area_text(candidate),` line with its two comment lines, and the
`"meals_included": meals == "included",` line with its two comment lines. `"city"` stays.

- [ ] **Step 6: Drop the tests that guarded them**

In `tests/test_accommodation_agent.py`: delete `test_the_deprecated_area_still_carries_the_distance`
whole (lines 80-86 including the blank lines around it), delete the line
`assert option["meals_included"] is False      # deprecated, for _digest (P7)` and the comment above
it, and delete `assert option["meals_included"] is (expected == "included")`.

In `tests/test_search_artifact.py`, remove `"area": "2.4 km from the terminal, Lod",` from
`GOOD_HOTEL`.

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`

The headings changed from `Onward flight` / `Somewhere to sleep` to `ONWARD FLIGHT` /
`SOMEWHERE TO SLEEP`, so four existing assertions in `tests/test_supervisor.py` need the new casing.
Change each of these lines:

```python
# in test_complete_message_dispatches_the_crew
    assert "ONWARD FLIGHT" in text

# in test_one_agent_failing_does_not_lose_the_rest_of_the_plan
    assert "ONWARD FLIGHT" in text
    assert "SOMEWHERE TO SLEEP" in text

# in test_a_failed_composing_call_still_hands_over_the_plan
    assert "ONWARD FLIGHT" in text
    assert "SOMEWHERE TO SLEEP" in text

# in test_a_half_labelled_option_still_produces_a_plan
    assert "ONWARD FLIGHT" in text
```

`test_a_half_labelled_option_still_produces_a_plan` passes a bare option with only `id` and `depart`,
which `_flight_line` must survive — that is the point of building every part with `.get()`.

Expected after those edits: PASS.

- [ ] **Step 8: Close the deprecation bookkeeping**

In `docs/PROJECT_PLAN.md` §1, delete the final bullet ("**Two deprecated fields remain...**") and
remove `area` and `meals_included` from the `AccommodationAgent` payload block if present.

In `docs/search-agents-capabilities.md:93`, replace the sentence about the two deprecated fields
carrying the old shape with a note that they were deleted on 15/8/2026 once `_digest` migrated.

In `docs/superpowers/specs/2026-08-15-search-agent-payload-refinement-design.md` §7, mark the
bookkeeping done.

- [ ] **Step 9: Commit**

```bash
git add lib/agents/ tests/ docs/
git commit -m "Show the passenger everything that was found, not one line of it"
```

---

### Task 7: Route the caveats instead of printing them

Both search agents state that `caveats` are "for the assistant coordinating this plan, not the
passenger". The load-bearing ones are generated in code precisely so they are reliable enough to
branch on (payload spec P3/P4). `ASK:` in particular is a question for the passenger arriving after
dispatch — a channel the gate does not have.

**Files:**
- Modify: `lib/agents/supervisor.py` (new `CAVEAT_BLOCKS`, `_split_caveats`; `_digest`; `COMPOSE_SYSTEM_PROMPT`)
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `_digest(results, failures)` from Task 6
- Produces: `supervisor.CAVEAT_BLOCKS` — ordered `((prefix, heading), ...)`
- Produces: `supervisor._split_caveats(results: dict) -> dict[str, list[str]]` keyed by prefix

- [ ] **Step 1: Write the failing tests**

In `tests/test_supervisor.py`, under a new `# --- caveats ---` section:

```python
CAVEATED = {
    "flight": {"options": [{"id": "F1"}], "recommended_id": "F1", "caveats": [
        "CONFIRM: LH 687 leaves in about 40 minutes - check they can reach the gate.",
        "NOTE: every option is on a different airline.",
    ]},
    "stay": {"options": [{"id": "H1"}], "recommended_id": "H1", "caveats": [
        "ASK: none of these has a phone number listed.",
        "every price here is an estimate.",
    ]},
}


def test_caveats_are_routed_to_their_own_blocks():
    text = supervisor._digest(CAVEATED, [])

    assert "BEFORE YOU ACT ON THIS" in text
    assert "THINGS I NEED FROM YOU" in text
    assert "WORTH KNOWING" in text


def test_the_prefix_is_stripped_before_the_passenger_sees_it():
    text = supervisor._digest(CAVEATED, [])

    assert "CONFIRM:" not in text
    assert "ASK:" not in text
    assert "NOTE:" not in text
    assert "LH 687 leaves in about 40 minutes" in text


def test_an_unprefixed_caveat_is_treated_as_a_note():
    routed = supervisor._split_caveats(CAVEATED)

    assert "every price here is an estimate." in routed["NOTE:"]


def test_the_rights_caveats_are_not_routed_as_the_search_protocol():
    results = {"rights": {"regulation": "EU 261/2004", "entitlements": [],
                          "next_actions": [], "caveats": ["No evidence on meals was retrieved."]}}
    text = supervisor._digest(results, [])

    # Same field name, different meaning: evidence gaps, not the NOTE:/ASK:/CONFIRM: protocol.
    assert "Not established from the sources: No evidence on meals was retrieved." in text
    assert "WORTH KNOWING" not in text


def test_no_caveats_means_no_empty_headings():
    text = supervisor._digest({"flight": {"options": [{"id": "F1"}], "recommended_id": "F1"}}, [])

    assert "BEFORE YOU ACT ON THIS" not in text
    assert "WORTH KNOWING" not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py -k caveat -q`
Expected: FAIL — `AttributeError: ... has no attribute '_split_caveats'`.

- [ ] **Step 3: Add the router**

In `lib/agents/supervisor.py`, add near `MEALS_TEXT`:

```python
# The search agents' caveat protocol (payload spec P4). Ordered: what stops the passenger
# acting comes before what is asked of them, which comes before what is merely worth
# knowing. They are written for whoever is coordinating the plan, so they are translated
# here rather than printed.
CAVEAT_BLOCKS = (
    ("CONFIRM:", "BEFORE YOU ACT ON THIS"),
    ("ASK:", "THINGS I NEED FROM YOU"),
    ("NOTE:", "WORTH KNOWING"),
)
```

and the function, above `_digest`:

```python
def _split_caveats(results: dict) -> dict[str, list[str]]:
    """Search-agent caveats, routed by prefix and stripped of it.

    An unprefixed one is a note: the model may add its own and the prefix is only asked
    for, while the ones that matter are generated in code and always carry it.
    """
    routed = {prefix: [] for prefix, _ in CAVEAT_BLOCKS}
    for key in ("flight", "stay"):
        for caveat in (results.get(key) or {}).get("caveats") or []:
            text = str(caveat).strip()
            prefix = next(
                (p for p, _ in CAVEAT_BLOCKS if text.upper().startswith(p)), None
            )
            body = text[len(prefix):].strip() if prefix else text
            if body:
                routed[prefix or "NOTE:"].append(body)
    return routed
```

- [ ] **Step 4: Print the blocks**

In `_digest`, immediately before the `if failures:` block:

```python
    routed = _split_caveats(results)
    for prefix, heading in CAVEAT_BLOCKS:
        if routed[prefix]:
            lines.append(heading)
            lines += [f"  - {item}" for item in routed[prefix]]
            lines.append("")
```

- [ ] **Step 5: Tell the composing call what to do with them**

In `COMPOSE_SYSTEM_PROMPT`, add after the "State amounts and entitlements..." paragraph:

```
Anything under "Before you act on this" goes ahead of your recommendation, not after it. If there is
a "Things I need from you" block, close on it as a direct question instead of the general invitation.
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/agents/supervisor.py tests/
git commit -m "Turn the crew's shorthand into something a passenger can act on"
```

---

### Task 8: Keep the options that were already paid for

A follow-up asking "what else was there?" currently re-dispatches FlightAgent — another LLM call and
2 AeroDataBox units for data that was already fetched and thrown away. The history column is
`jsonb`, so storing the payloads on the turn needs no schema change, and
`conversation.list_conversations` reads only `prompt` and `response` and ignores anything else.

**Files:**
- Modify: `lib/agents/supervisor.py` (`run`'s return, `_prior_results_block`, `_refine_prompt`, `_compose_prompt`, `REFINE_SYSTEM_PROMPT`)
- Modify: `api/index.py` (unpack three values, store `results`)
- Modify: `scripts/run_search_agents_live.py:89`
- Test: `tests/test_supervisor.py`, `tests/test_execute.py`

**Interfaces:**
- Consumes: `supervisor._options` from Task 6
- Produces: `supervisor.run(prompt, history, *, local_time=None) -> tuple[str, list[dict], dict]` — the third value is `{"flight": payload, "stay": payload, "rights": payload}`, successful keys only
- Produces: `supervisor._prior_results_block(history: list[dict]) -> list[str]`
- Produces: a stored turn is `{"prompt": str, "response": str, "results": dict}`

- [ ] **Step 1: Write the failing tests**

In `tests/test_supervisor.py`, add a `# --- results across turns ---` section:

```python
def test_the_results_come_back_with_the_plan():
    _, _, results = supervisor.run(COMPLETE, [])

    assert set(results) == {"flight", "stay", "rights"}
    assert results["flight"]["recommended_id"] == "F1"


def test_a_gated_turn_returns_no_results():
    _, _, results = supervisor.run("my flight got cancelled help", [])

    assert results == {}


def test_a_failed_agent_leaves_its_key_out(monkeypatch):
    monkeypatch.setattr(documentation_agent, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _, _, results = supervisor.run(COMPLETE, [])

    assert "rights" not in results
    assert "flight" in results


def test_earlier_options_reach_the_refinement_call():
    history = [{"prompt": COMPLETE, "response": "a plan",
                "results": {"flight": {"options": [{"id": "F1", "flight_number": "LH 687",
                                                    "depart": "2026-08-16T09:40"}],
                                       "recommended_id": "F1"}}}]
    _, steps, _ = supervisor.run("what else was there?", history)

    assert "LH 687" in steps[0]["prompt"]["user_prompt"]


def test_a_question_answered_from_earlier_options_dispatches_nobody():
    history = [{"prompt": COMPLETE, "response": "a plan",
                "results": {"flight": {"options": [{"id": "F1", "flight_number": "LH 687"}],
                                       "recommended_id": "F1"}}}]
    _, steps, _ = supervisor.run("never mind, thanks", history)

    # Re-running a search for something already found costs time and money.
    assert modules_of(steps) == ["Supervisor", "Supervisor"]


def test_a_turn_without_results_is_not_a_crash():
    history = [{"prompt": "hello", "response": "hi"}]

    assert supervisor._prior_results_block(history) == []


def test_only_the_most_recent_results_are_shown():
    history = [
        {"prompt": "a", "response": "a", "results": {"flight": {
            "options": [{"id": "F1", "flight_number": "OLD 111"}], "recommended_id": "F1"}}},
        {"prompt": "b", "response": "b", "results": {"flight": {
            "options": [{"id": "F1", "flight_number": "NEW 222"}], "recommended_id": "F1"}}},
    ]
    block = "\n".join(supervisor._prior_results_block(history))

    # History is re-sent on every call of every turn.
    assert "NEW 222" in block
    assert "OLD 111" not in block
```

Every other `supervisor.run(...)` call in `tests/test_supervisor.py` now unpacks three values. The
edit is mechanical and exhaustive — find every line matching `= supervisor.run(` and add one `_`:

| was | becomes |
|---|---|
| `text, steps = supervisor.run(...)` | `text, steps, _ = supervisor.run(...)` |
| `text, _ = supervisor.run(...)` | `text, _, _ = supervisor.run(...)` |
| `_, steps = supervisor.run(...)` | `_, steps, _ = supervisor.run(...)` |

Find them with `grep -n "= supervisor.run(" tests/test_supervisor.py`. There must be no remaining
two-value unpack when you are done.

In `tests/test_execute.py`:

```python
def test_the_agents_results_are_stored_on_the_turn(monkeypatch):
    store = {"c1": []}

    monkeypatch.setattr(conversation, "load_history",
                        lambda owner_id, conversation_id: list(store.get(conversation_id, [])))
    monkeypatch.setattr(conversation, "save_history",
                        lambda owner_id, conversation_id, history, title:
                            store.__setitem__(conversation_id, history))
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: ("ok", [], {"flight": {"options": []}}),
    )

    client.post("/api/execute", json={"prompt": COMPLETE, "conversation_id": "c1"})

    assert store["c1"][-1]["results"] == {"flight": {"options": []}}
```

and the two existing fakes of `supervisor.run` in that file return three values.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py tests/test_execute.py -q`
Expected: FAIL — `ValueError: not enough values to unpack (expected 3, got 2)`.

- [ ] **Step 3: Return the results**

In `lib/agents/supervisor.py`, change `run`'s signature and both return points:

```python
def run(prompt: str, history: list[dict], *,
        local_time: datetime | None = None) -> tuple[str, list[dict], dict]:
    """Returns (response_text, steps, results) — see `docs/PROJECT_PLAN.md` §1.

    `results` is what each agent came back with, so a follow-up can be answered from
    options already paid for instead of re-dispatching for them.
    """
```

The gate's early return:

```python
        return opener + "\n" + "\n".join(f"  - {q}" for q in asked), steps, {}
```

The final return:

```python
    return text, steps, results
```

- [ ] **Step 4: Render the prior options**

Add above `_refine_prompt`:

```python
def _prior_results_block(history: list[dict]) -> list[str]:
    """The options already on the table, identities only.

    The most recent turn that produced any, and nothing more: history is re-sent on
    every call of every turn, so whole payloads would cost O(n^2) tokens against a $13
    project budget (`docs/PROJECT_PLAN.md` §7).
    """
    prior = next((t.get("results") for t in reversed(history) if t.get("results")), None)
    if not prior:
        return []

    lines = ["Options already on the table from earlier in this conversation:"]
    onward = _options(prior.get("flight"))
    if onward:
        lines.append("  flights: " + "; ".join(
            " ".join(p for p in (o.get("id"), o.get("flight_number"),
                                 f"departs {o['depart']}" if o.get("depart") else "") if p)
            for o in onward))
    stays = _options(prior.get("stay"))
    if stays:
        lines.append("  stays: " + "; ".join(
            " ".join(p for p in (o.get("id"), o.get("name"),
                                 f"{o['distance_km']} km" if o.get("distance_km") is not None else "")
                     if p)
            for o in stays))
    return lines if len(lines) > 1 else []
```

In `_refine_prompt`, insert the block before the passenger's message:

```python
def _refine_prompt(prompt: str, history: list[dict]) -> str:
    lines = _history_block(history)
    lines += _prior_results_block(history)
    if lines:
        lines.append("")
    lines.append(f"Passenger's message: {prompt.strip()}")
    return "\n".join(lines)
```

In `_compose_prompt`, after the `block = _history_block(history)` stanza:

```python
    prior = _prior_results_block(history)
    if prior:
        lines += [""] + prior
```

- [ ] **Step 5: Teach the refinement call to use them**

In `REFINE_SYSTEM_PROMPT`, replace the sentence
`Return an empty "needs" when the question can be answered from what is already on the table.` with:

```
Return an empty "needs" when the question can be answered from what is already on the table -
including any options listed under "Options already on the table", which were already found and
paid for. Re-running a search for something already found costs the passenger time and costs the
project money.
```

- [ ] **Step 6: Store the results on the turn**

In `api/index.py`:

```python
        response_text, steps, results = supervisor.run(prompt, history, local_time=local_time)
```

```python
    if conversation_id:
        turn = {"prompt": prompt, "response": response_text, "results": results}
```

- [ ] **Step 7: Follow the return shape in the live runner**

In `scripts/run_search_agents_live.py`:

```python
        response, steps, _ = supervisor.run(case["prompt"], list(case["history"]))
```

- [ ] **Step 8: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 9: Record the shapes**

In `docs/PROJECT_PLAN.md` §1, update the `supervisor.run` signature line to
`-> tuple[str, list[dict], dict]`, and add after the `history` paragraph:

> A stored turn is `{"prompt", "response", "results"}`. `results` is the third value
> `supervisor.run` returns — each agent's payload, successful keys only — so a follow-up can be
> answered from options already paid for. `conversations.history` is `jsonb` and
> `conversation.list_conversations` reads only `prompt` and `response`, so this needed no schema
> change.

- [ ] **Step 10: Commit**

```bash
git add lib/agents/supervisor.py api/index.py scripts/run_search_agents_live.py \
        tests/ docs/PROJECT_PLAN.md
git commit -m "Stop paying twice for options already found"
```

---

### Task 9: Log the decisions and verify the whole thing

**Files:**
- Modify: `docs/PROJECT_PLAN.md` §6 (decisions log)
- Modify: `api/index.py` (`AGENT_INFO["description"]`)
- Test: the full suite

- [ ] **Step 1: Add the decisions**

Append these rows to the `docs/PROJECT_PLAN.md` §6 table, matching the existing `| date | decision |
rationale |` format:

| Date | Decision | Rationale |
|---|---|---|
| 15/8/2026 | `local_now` is the **passenger's** wall clock, sent by the GUI as an optional `local_time` on `/api/execute`, and stays **naive** | The server clock is UTC on Vercel — wrong for the date sync, and wrong for AeroDataBox, which expects airport-local time and is now cached by date. Naive because `flight_agent` subtracts `local_now` from a tz-stripped departure; an aware value is a `TypeError` on every turn. |
| 15/8/2026 | Intake sanity checks are **hybrid**: the refinement call reports semantic conflicts, Python re-checks the arithmetic and reuses `flights.route_problem` | No extra LLM call. Only the model can see "today, 11th of November" as a contradiction; only code gets the date arithmetic reliably right. `route_problem` already caught two of the checks — it just ran after the gate, where it raised instead of asking. |
| 15/8/2026 | Conflicts in `BLOCKING_CONFLICTS` stop the dispatch and are asked about; the rest become stated `assumptions`, and impossible fields are cleared | A wrong incident date corrupts the flight window and the entitlement clock alike, so it is worth a round trip. Bouncing a stressed passenger over a cosmetic mismatch is the friction this exists to remove. |
| 15/8/2026 | `supervisor.run` returns `(text, steps, results)` and the turn stores `results` | A follow-up "what else was there?" cost a re-dispatch — an LLM call plus 2 AeroDataBox units — for data already fetched and discarded. `history` is `jsonb`, so no schema change. |
| 15/8/2026 | `LLMError` gains `passenger_message`; the Supervisor never prints an exception | `RuntimeError("Pinecone unreachable")` was reaching a stranded passenger, while the agents' own carefully written refusals were worth keeping. The same exception type carries both and nothing else told them apart. |
| 15/8/2026 | Cross-checking between agents is **out of scope** | Reconciling hotel distance against departure time, or entitlement claims against found flights, is the largest surface in the design and none of the three problems needed it. 8 days to the deadline. |

- [ ] **Step 2: Update what the agent says it does**

In `api/index.py`, in `AGENT_INFO["description"]`, replace this fragment of the "What it CAN do"
paragraph:

```python
        "What it CAN do: interpret an underspecified account of a disruption and ask what's "
        "missing; propose onward flights; propose accommodation matched to those dates; explain "
```

with:

```python
        "What it CAN do: interpret an underspecified account of a disruption, ask what's "
        "missing, and check what it was told against the clock and the map before acting on it "
        "— a date that cannot be today, a route that cannot exist; propose onward flights; "
        "propose accommodation matched to those dates; explain "
```

- [ ] **Step 3: Verify the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, zero failures. Record the count.

- [ ] **Step 4: Verify no internal vocabulary can reach a prompt**

Run: `.venv/Scripts/python.exe -m pytest tests/test_supervisor.py -k "composing_prompt" -v`
Expected: PASS — both `test_the_composing_prompt_never_names_the_inside_of_the_system` and
`test_the_composing_prompt_says_who_it_is`.

- [ ] **Step 5: Verify the app still starts and a bare prompt still works**

Run: `.venv/Scripts/python.exe -c "from api.index import app; print('imported ok')"`
Expected: `imported ok`, no traceback.

Then start the server and POST the grader's shape — no `conversation_id`, no `local_time`. Set
`WINGMAN_ALLOW_LLM=0` for this check: it is an ad-hoc script running outside the pytest suite, so it
is exactly the case the budget guard exists for — `load_dotenv()` in `api/index.py` will re-arm
`LLMOD_API_KEY`/`LLMOD_API_BASE` from the real `.env` on import, and this check has no business
spending against them.

```bash
WINGMAN_ALLOW_LLM=0 .venv/Scripts/python.exe -m uvicorn api.index:app --port 8000 &
curl -s -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt":"LH318 TLV -> FRA was cancelled at the gate"}' | head -c 400
```

Expected: a JSON object with exactly `status`, `error`, `response`, `steps`. With calls disabled this
returns `status: "error"` with the disabled-calls message — that is the correct outcome and proves
the shape without spending anything. Stop the server afterwards.

- [ ] **Step 6: Commit**

```bash
git add docs/PROJECT_PLAN.md api/index.py
git commit -m "Log the intake and composition decisions"
```

---

## Notes for the executor

**Three files here belong to other people.** All three changes are additive or already agreed, but
they belong in a handover conversation, not just a PR:

- `lib/llm.py` and the three raise sites in `flight_agent.py` / `accommodation_agent.py` (Task 5) — Person B
- deleting `area` and `meals_included` from `accommodation_agent._enrich` (Task 6) — Person B, already scheduled in their spec §7
- one line in `documentation_agent._user_prompt` (Task 2) — Person C

**`tests/conftest.py` is shared.** Tasks 3, 5 and 6 all edit it. After each, run the whole suite —
not `tests/test_supervisor.py` alone — because Person B's and Person C's tests run against the same
fakes.

**No task adds an LLM call.** If an approach seems to need one, it is the wrong approach; the budget
is $13 for the project and this plan spends only prompt tokens, capped by `MAX_DIGEST_OPTIONS` and
by showing one turn of prior results.
