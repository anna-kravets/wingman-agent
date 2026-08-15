# Supervisor intake, composition and voice — design

**Owner:** Person A · **Date:** 15/8/2026 · **Deadline:** 23/8/2026 — 8 days.

Three problems, one file. The Supervisor accepts whatever the passenger asserts without checking it,
throws away most of what the sub-agents return, and tells the passenger about "the crew".

**Consumes** `docs/superpowers/specs/2026-08-15-search-agent-payload-refinement-design.md`, which
names this work as its dependency: *"Person A is rewriting `supervisor._digest` to consume every
field the sub-agents return, and to act on `caveats`."* Two deprecated compatibility fields (`area`,
`meals_included`) are deleted here, closing that spec's §7 bookkeeping.

---

## 1. Why now

**Intake.** The refinement pass never sees a clock. `local_now` is stamped in Python *after* the call
(`supervisor.py:189`), so the model cannot tell that "today, 11th of November" is not today. Nothing
else checks it either: `missing` fields are the only reason the gate ever asks a question, so a
stated-but-wrong fact goes straight to six paid LLM calls and a plan built on it. There is no field
holding when the disruption happened at all, so even a correct date is discarded.

**Composition.** `_digest` forwards the recommended option only, and of that a handful of fields.
Every alternative, every `rebooking` note, every price hedge and the whole of `caveats` die there.
The product promises comparison — `agent_info.description` says the passenger can "compare options"
— and the composing call is never shown a second option to compare. Worse, `_digest` reads
`meals_included`, which is `False` when the truth is `meals: "unknown"`: the plan asserts "no meals"
as a fact today.

**Voice.** `COMPOSE_SYSTEM_PROMPT` says "the results the crew returned", "a crew result", "part of
the crew failed" (`supervisor.py:80,86,87`), and the compose prompt is headed "What the crew came
back with:" (`:284`). The model echoes what we wrote. It is also never told who it is. Separately,
`dispatch` interpolates the raw exception into passenger text (`:346`), so
`RuntimeError("Pinecone unreachable")` is something a stranded passenger reads.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| S1 | `local_now` is the **passenger's** wall clock, supplied by the GUI as an optional `local_time` on `/api/execute`, falling back to the server clock | The server clock is UTC on Vercel. It is what the date sync compares, what any date sanity check compares, and what builds the AeroDataBox window — an API that expects **airport-local** time. `flights.py:140` now caches on `after.date()`, so a UTC clock queries the wrong window *and* poisons the cache key under the wrong date. |
| S2 | `local_now` stays a **naive** ISO string | `flight_agent.py:154-156` computes `datetime.fromisoformat(option["depart"]).replace(tzinfo=None) - datetime.fromisoformat(request["local_now"])`. An offset-aware `local_now` makes that `naive - aware` → `TypeError` on every FlightAgent turn. Carrying the offset buys nothing: every consumer wants wall-clock local time. |
| S3 | Sanity checking is **hybrid** — the model reports semantic conflicts in the same refinement call, Python re-checks anything that is arithmetic | Zero extra LLM calls. The model is the only thing that can see "today, 11th of November" as a self-contradiction; it is also the thing that gets date arithmetic quietly wrong, which is exactly the case being fixed. Each side does what it is reliable at. |
| S4 | **Hard conflicts block dispatch, soft ones proceed with a stated assumption.** Severity is a fixed field list in Python. `BLOCKING_CONFLICTS = {"route", "stranded_at"}` only — **reversed same day:** `incident_time` was in this set too, but nothing downstream needs it exact (the flight window comes from `local_now`, not `incident_time`), so a disruption reported weeks late — the product's own core case — was being asked a question with no right answer, because the date was right. It is now soft: stated as an assumption and cleared, same as `arrive_by` | A route that cannot exist or an airport nobody can look up leaves the crew nothing to search against, so asking is cheaper than a confident wrong plan. Bouncing a stressed passenger over a cosmetic mismatch — or a valid, merely late, disruption — is the friction this product exists to remove. |
| S5 | The gate **reuses `flights.route_problem`** rather than reimplementing route checks | It already catches `origin == destination` and unknown IATA codes, in passenger-ready wording, tested. It just runs too late — inside `FlightAgent.run`, after the gate, where it raises and the passenger gets a dead end instead of a question. |
| S6 | `_digest` consumes **every** field in the new payloads and the two deprecated fields are deleted | The contract in `PROJECT_PLAN.md:97-119`. `phone` in particular hands the passenger the one job neither agent can do — confirming a room and a rate. |
| S7 | `caveats` are **routed by prefix, never printed** | Both agents state "`caveats` is for the assistant coordinating this plan, not the passenger". `NOTE:` folds into the prose, `CONFIRM:` surfaces before the recommendation, `ASK:` becomes the question the plan closes on. |
| S8 | `DocumentationAgent.caveats` are handled **separately** from search-agent caveats | Same field name, different meaning: evidence gaps, unprefixed. Prefix-parsing them would file them all under `NOTE:`. |
| S9 | `supervisor.run` returns **`(text, steps, results)`** and `api/index.py` stores `results` on the turn | The history column is `jsonb`, so no schema change. A follow-up "what else was there?" then costs one refinement call instead of a re-dispatch plus 2 AeroDataBox units. |
| S10 | Only the **most recent** turn's results reach a prompt, trimmed to option identities | History is re-sent on every call of every turn. Full payloads would be O(n²) tokens against $13. |
| S11 | `LLMError` gains an optional `passenger_message`; `dispatch` prefers it and falls back to a fixed sentence per need | The same exception type carries both Person B's careful refusal text (`NO_LIVE_DATA`, `route_problem`) and internal text like "no option named a flight that was actually offered", and every message is `MODULE: `-prefixed. Nothing else distinguishes them. Stripping the prefix and passing the rest through would ship the internal ones verbatim. |
| S12 | Cross-checking between agents is **out of scope** | Reconciling hotel distance against departure time, or entitlement claims against found flights, is the largest new surface here and none of it is required by the three problems above. 8 days left. |

---

## 3. The clock (S1, S2)

`POST /api/execute` accepts an optional `local_time` alongside the existing optional
`conversation_id`. A bare `{"prompt": "..."}` — what a grader sends — behaves exactly as today.

```python
# api/index.py
def _local_time(value) -> datetime | None:
    """The passenger's wall clock, or None. Never raises: a bad value is not worth a failed turn."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)   # S2: keep the wall-clock reading, drop the offset
```

Passed through as `supervisor.run(prompt, history, local_time=...)` — keyword-only, so the locked
positional call shape still works. `_request_from` uses it where it currently calls `datetime.now()`.

The GUI sends its own local wall time, no offset:

```js
const d = new Date(), p = n => String(n).padStart(2, '0');
const local_time = `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}`
                 + `T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
```

---

## 4. Intake and sanity checks (S3, S4, S5)

### What the refinement call sees and returns

`_refine_prompt` gains one line, in the **user** prompt because it changes every call:

```
Current local date and time: 2026-08-15T22:15:00
```

The extraction JSON gains two keys:

- **`incident_time`** — ISO 8601 or null. The scheduled departure of the disrupted flight. New: no
  field held this before, so it was dropped whether right or wrong. It also gives
  `DocumentationAgent` the entitlement clock it currently lacks.
- **`conflicts`** — `[{"field", "stated", "reason"}]`. What the passenger asserted that contradicts
  either the current date/time or something else they said. The system prompt names the case
  directly: a relative word ("today", "this morning") next to an absolute date that is not that day.

### What Python re-checks

`_conflicts(request)` runs deterministically and merges with the model's list, deduped by field:

| Field | Trigger |
|---|---|
| `incident_time` | parses, and is more than `INCIDENT_FUTURE_LIMIT_DAYS = 30` ahead of `local_now` |
| `incident_time` | parses, and is more than `INCIDENT_PAST_LIMIT_DAYS = 14` behind `local_now` |
| `route` | `flights.route_problem(origin, destination)` returns a reason (S5) |
| `stranded_at` | set, and `airports.lookup` does not know it |
| `arrive_by` | parses, and is earlier than `local_now` |

The future limit is deliberately generous: a flight cancelled a fortnight in advance is a real case
and must not be flagged. It still catches the motivating example — "today, 11th of November" read on
15 August is 88 days out. The past limit is about usefulness, not validity: a claim is still good
months later, but "a bed tonight" for a three-week-old disruption is nonsense.

### Blocking vs soft (S4)

```python
BLOCKING_CONFLICTS = {"route", "stranded_at"}
```

**Reversed same day.** `incident_time` was in this set at first, on the theory that a wrong incident
date corrupts the flight window and the entitlement clock. It does not: the flight window is built
from `local_now` (`_stay_window`), never from `incident_time`, whose only consumer anywhere is one
informational line in `documentation_agent.py:129`. Blocking on it meant a disruption more than
`INCIDENT_PAST_LIMIT_DAYS` old — the product's own core "compensation discovered weeks later" case —
was asked *"You said the flight was on X, but right now it is Y. Which is right?"*, a question with
no right answer, because the date **is** right. Re-extraction flagged it again on the next turn, so
there was no way through. `incident_time` now behaves like any other soft conflict.

Blocking conflicts join `missing` at the existing gate (`supervisor.py:324`) — same one-call-not-seven
economics, and the same exemption when `needs` is empty, so a follow-up is never interrogated for
details it already gave.

Question text is slot-filled, no second LLM call:

```python
CONFLICT_QUESTIONS = {
    "route": "{reason}",  # route_problem already writes a passenger-ready sentence,
                          # sentence-cased on use: its reasons start lower-case by design,
                          # because today they are interpolated after "FlightAgent: ".
    "stranded_at": "I could not find an airport with the code {stated}. Which airport are you at?",
}
# fallback for anything the model reports on another field:
# "You said {stated} — {reason}. Which is right?"
```

Soft conflicts do two things: record a sentence in `request["assumptions"]`, which the compose prompt
renders so the plan states it out loud, and **null the field** where keeping it would poison a
downstream prompt — an `arrive_by` in the past becomes "as soon as possible" rather than an
impossible deadline handed to FlightAgent, and an `incident_time` that reads as suspiciously far away
is dropped rather than handed to DocumentationAgent unexplained.

---

## 5. Composition (S6, S7, S8)

`_digest` is rewritten. It is both the composing call's input and, when that call fails, the plan the
passenger reads — so its headings stay plain English and nothing is indexed with `[]`.

```
ONWARD FLIGHT
  Recommended: LH 687 Lufthansa, TLV to FRA. Departs 2026-08-16T09:40 from terminal 3,
    arrives 13:05 (3h 25m). Airbus A320. Status: Expected.
    Rebooking: <rebooking>
    <notes>
  Also available:
    LY 357 El Al, departs 06:05, arrives 09:40.
    ...

SOMEWHERE TO SLEEP
  Recommended: Airport Plaza, 2.4 km from the terminal, Lod. 12 HaNasi.
    Phone +972 3 000 0000. 4 stars. Step-free access: yes.
    2026-08-15 to 2026-08-16, 1 night. Meals: not confirmed.
    Roughly EUR 110-140 for the night (estimate - not a quoted price).
    <notes>
  Also available: ...

WHAT YOU ARE OWED (EU 261/2004)
  - hotel: <summary> [EU 261 Art. 9(1)(b)] (high confidence)
  Next: <action>
  Not established from the sources:
  - <DocumentationAgent caveat>            <- S8, unprefixed, its own block

BEFORE YOU ACT ON THIS
  - <CONFIRM: caveats, prefix stripped>

THINGS I NEED FROM YOU
  - <ASK: caveats, prefix stripped>

WORTH KNOWING
  - <NOTE: caveats, prefix stripped>

COULD NOT COMPLETE
  - <passenger message per failed need>    <- S11
```

`MAX_DIGEST_OPTIONS = 3` per agent. Field mapping, closing the payload spec's §7:

| was | becomes |
|---|---|
| `stay.area` *(deprecated)* | `distance_km` + `city` |
| `stay.meals_included` *(deprecated)* | `meals`: `included` / `not_included` / `unknown` → "not confirmed" |
| `flight.fare_conditions` (never read) | `rebooking` |
| — | `terminal`, `aircraft`, `status`, `duration_minutes`, `arrives_next_day`, `airline_iata` |
| — | `phone`, `address`, `website`, `kind`, `stars`, `wheelchair` |

`kind` is present only when the property is *not* an ordinary hotel, so it is rendered only when
present — "a hostel, so expect shared facilities" is a surprise worth removing.

`_split_caveats(caveats)` routes on the prefix; an unprefixed string from a search agent falls into
`NOTE:`. Both deprecated reads are deleted in the same commit, and Person B's two compatibility
fields can then be removed from `accommodation_agent._enrich`.

---

## 6. Results across turns (S9, S10)

```python
supervisor.run(prompt, history, *, local_time=None) -> tuple[str, list[dict], dict]
# results = {"flight": payload, "stay": payload, "rights": payload}  — successful keys only
```

`api/index.py` stores `{"prompt", "response", "results"}` on the turn. No schema change:
`conversations.history` is `jsonb`, and `conversation.list_conversations` reads only `prompt` and
`response`, ignoring unknown keys.

A new `_prior_results_block(history)` renders the **last** turn that carries results, identities only:

```
Options already on the table from earlier in this conversation:
  flights: F1 LH 687 departs 2026-08-16T09:40; F2 LY 357 departs 2026-08-16T06:05
  stays: H1 Airport Plaza 2.4 km; H2 City Central Inn 11.2 km
```

It goes into both Supervisor prompts. `REFINE_SYSTEM_PROMPT` gains: when the options already on the
table answer the question, return an empty `needs`. That is what makes "what else was there?" cost
one call.

---

## 7. Voice and failure text (S11)

`COMPOSE_SYSTEM_PROMPT` opens with an identity and a ban:

> You are Wingman, writing directly to one passenger whose flight has just been disrupted.
>
> You are a single assistant. Never mention agents, modules, tools, searches, internal steps, or that
> any work was divided up — the passenger is talking to you, not to a system.

Every occurrence of "crew" is deleted from the system prompt (`:80,86,87`) and the compose prompt
header becomes `Findings:` (`:284`). The model echoed "crew" because we wrote it there.

The prompt also gains caveat instructions matching §5: state anything under *Before you act on this*
ahead of the recommendation, and close on the *Things I need from you* question rather than the
generic "ask me to compare any of these".

Failure text stops being an interpolated exception:

```python
# lib/llm.py
class LLMError(RuntimeError):
    def __init__(self, message, steps=None, passenger_message=None):
        ...
        self.passenger_message = passenger_message
```

Raise sites gaining a `passenger_message`: `flight_agent.py:212` (`route_problem`'s reason),
`flight_agent.py:224` and `accommodation_agent.py:200` (`NO_LIVE_DATA`). Everything else leaves it
`None` and gets the fallback.

```python
FAILURE_MESSAGES = {
    "flight": "I could not get onward flight options just now.",
    "stay":   "I could not find somewhere to stay just now.",
    "rights": "I could not work out what you are owed just now.",
}
```

`dispatch` records `getattr(exc, "passenger_message", None) or FAILURE_MESSAGES[key]`, and adds a
`logger.exception` so a bare exception's cause survives in the Vercel logs. `steps[]` keeps it too,
which is where the guidelines put the debugging trace.

---

## 8. Interface changes (locked — `PROJECT_PLAN.md` §1)

```python
# request, as extracted
{"airline", "flight_number", "origin", "destination",
 "disruption": "delayed" | "cancelled" | "denied_boarding",
 "stranded_at", "party_size", "arrive_by": iso8601 | None,
 "incident_time": iso8601 | None,                          # new
 "conflicts": [{"field", "stated", "reason"}],             # new
 "assumptions": [str],                                     # new
 "needs": [...], "local_now": iso8601, "missing": [...]}

supervisor.run(prompt, history, *, local_time=None) -> tuple[str, list[dict], dict]   # was 2-tuple

# a stored turn
{"prompt": str, "response": str, "results": dict}          # results new
```

---

## 9. Impact elsewhere

- **`scripts/run_search_agents_live.py`** unpacks a 2-tuple at `:89` and monkeypatches
  `_extract_request` at `:78-80` purely to force `local_now` after extraction — which would compute
  conflicts against the wrong clock. With `local_time` a parameter, `build_request_patch` loses its
  `local_now` line and the script passes the scenario clock directly. It gets shorter.
- **`tests/test_search_runner.py`** fakes `_extract_request`; its stub request needs the new keys.
- **`tests/conftest.py`** — `_refine_response` gains `incident_time` and `conflicts`; the
  `"What the crew came back with:"` marker in `_compose_response`/`_supervisor_response` becomes
  `"Findings:"`.
- **`lib/agents/documentation_agent.py`** `_user_prompt` gains one line for `incident_time`.
  Additive; coordinate with Person C.
- **`lib/agents/accommodation_agent.py`** — delete `area` and `meals_included` from `_enrich` and the
  `_area_text` helper, once §5 lands. Coordinate with Person B.
- **Docs updated in the same commit** (`CLAUDE.md` §7): `PROJECT_PLAN.md` §1 (shapes above) and §6
  (decisions log); `CLAUDE.md` §6, which states `Supervisor.run(prompt, history)`;
  `docs/search-agents-capabilities.md:93` and the payload spec §7, both of which track the deprecated
  fields as outstanding; `agent_info.description`, for the sanity-check behaviour.
- **GUI** sends `local_time` on every call. No rendering change: `public/index.html` renders steps
  with `JSON.stringify`.

---

## 10. Testing

Updated: `test_one_agent_failing_does_not_lose_the_rest_of_the_plan` asserts `"Pinecone unreachable"`
reaches the passenger today — it inverts to assert it does **not**, and that the fixed sentence does.
Every `supervisor.run` call site in the suite unpacks three values.

New:

- an incident date far from `local_now` blocks dispatch and asks, at a cost of one step
- a model-reported conflict on a blocking field blocks; on a soft field it proceeds, states the
  assumption, and nulls the field
- `origin == destination` is caught at the gate, before any agent runs
- a blocking conflict with empty `needs` does **not** interrogate
- `local_time` is honoured; an unparseable one falls back to the server clock
- the second flight option and the second stay reach the digest
- `phone`, `distance_km`, `terminal` and `arrives_next_day` reach the digest
- `meals: "unknown"` renders as not-confirmed, and `meals_included` is read nowhere
- `CONFIRM:` / `ASK:` / `NOTE:` land in their own blocks, prefix stripped, and
  `DocumentationAgent.caveats` land in the evidence block instead
- results are stored on the turn, and a follow-up answerable from them dispatches nobody
- a failure with `passenger_message` shows it; one without shows the fixed sentence; neither shows
  the exception

New tests reach the agents through `fake_search_data`, which both agents now require — they refuse
without candidates rather than inventing options.

---

## 11. Risks

**The composing prompt grows.** More options and more fields per option is the point, but it is also
tokens on every turn. Mitigated by `MAX_DIGEST_OPTIONS = 3`, identity-only prior results, and one
turn of them. Measure a real turn against the pre-change baseline before submission.

**`incident_time` thresholds are judgement calls.** 30 days ahead and 14 behind are chosen to catch
the motivating case without flagging a legitimately pre-announced cancellation. They are module
constants, so a live run that flags a real case is a one-line change.

**Three files outside this owner's territory move**: `lib/llm.py` (S11), `accommodation_agent.py`
(deprecated-field deletion, already agreed in the payload spec §7) and one line of
`documentation_agent.py`. All three are additive or already scheduled; none changes behaviour for
their owners. They belong in the handover conversation, not just the PR.
