# Search-Agent Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate `FlightAgent` and `AccommodationAgent` against a real model and real data, and write down exactly what they can and cannot do.

**Architecture:** Declarative scenarios in `evals/`, a budget-gated live runner that drives `supervisor.run` with `DocumentationAgent` faked, and an offline evaluator that mechanically checks the saved artifact. The evaluator is built and tested *before* the runner so no money is spent proving checks that were never exercised.

**Tech Stack:** Python 3.12+, `pytest`, `httpx`, `python-dotenv`. No new dependencies.

**Spec:** [`../specs/2026-08-10-search-agent-validation-design.md`](../specs/2026-08-10-search-agent-validation-design.md)

## Global Constraints

- Module names locked: `Supervisor`, `FlightAgent`, `AccommodationAgent`, `DocumentationAgent`.
- Every LLM call produces exactly one `steps[]` entry; tool/HTTP calls produce none.
- Budget ceiling: **$1 of LLMod spend**, ~20 chat calls expected. `--max-llm-calls` default **30**, abort rather than exceed.
- AeroDataBox: **592/600 units remaining**, 2 units per call. `--all` runs in one process so the route cache is shared.
- Never name a booking or flight-search website in user-facing text.
- `pytest` must stay green throughout (109 tests at branch point).
- Tests must pass with no API keys; only `scripts/run_search_agents_live.py` may spend.
- Artifacts go to `live-test-output/` (already gitignored).
- Follow Person C's conventions: `{"schema_version": 1, "cases": [...]}`, `--confirm-paid-calls`, artifact dict with `status`/`payload`/`steps`/`chat_usage`.

## File Structure

| File | Responsibility |
|---|---|
| `evals/search_agent_cases.json` | The 12 scenarios, declarative |
| `evals/search_cases.py` | Loader + validation, mirroring `documentation_cases.py` |
| `evals/search_artifact.py` | The mechanical checks, importable and testable offline |
| `scripts/run_search_agents_live.py` | Budget-gated live runner |
| `scripts/evaluate_search_artifact.py` | CLI wrapper over `evals/search_artifact.py` |
| `tests/test_search_cases.py` | Loader tests |
| `tests/test_search_artifact.py` | Check tests, against synthetic artifacts |
| `docs/search-agents-capabilities.md` | The capability/limitation map |

---

### Task 1: Scenario definitions and loader

**Files:**
- Create: `evals/search_agent_cases.json`, `evals/search_cases.py`
- Test: `tests/test_search_cases.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `evals.search_cases.load_search_cases(path: Path = CASES_PATH) -> dict[str, dict]`, keyed by `case_id`. Each case has keys `case_id`, `title`, `probes` (list[str]), `prompt` (str), `history` (list[dict]), `request_override` (dict), `local_now` (dict with `offset_days: int` and `time: "HH:MM"`), `live_data` (bool), `checks` (list[str]), `expect` (dict).

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_cases.py`:

```python
"""The scenario file is data, so a typo in it should fail loudly here."""

import json

import pytest

from evals.search_cases import CHECK_NAMES, load_search_cases


def test_loads_all_twelve_scenarios():
    cases = load_search_cases()
    assert len(cases) == 12


def test_every_case_declares_what_it_probes():
    for case_id, case in load_search_cases().items():
        assert case["probes"], f"{case_id} declares no probes"
        assert case["title"], f"{case_id} has no title"


def test_every_declared_check_is_a_real_check():
    for case_id, case in load_search_cases().items():
        unknown = set(case["checks"]) - CHECK_NAMES
        assert not unknown, f"{case_id} names unknown checks: {unknown}"


def test_local_now_is_well_formed():
    for case_id, case in load_search_cases().items():
        local_now = case["local_now"]
        assert isinstance(local_now["offset_days"], int), case_id
        hour, _, minute = local_now["time"].partition(":")
        assert 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59, case_id


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "dupe.json"
    case = {
        "case_id": "same", "title": "t", "probes": ["x"], "prompt": "p",
        "history": [], "request_override": {}, "local_now": {"offset_days": 0, "time": "12:00"},
        "live_data": True, "checks": [], "expect": {},
    }
    path.write_text(json.dumps({"schema_version": 1, "cases": [case, dict(case)]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_search_cases(path)


def test_wrong_schema_version_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        load_search_cases(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search_cases.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.search_cases'`

- [ ] **Step 3: Write the scenario file**

Create `evals/search_agent_cases.json`:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "case_id": "tlv-fra-cancelled",
      "title": "Demo route, cancelled at the gate",
      "probes": ["grounding", "cross_airline", "date_sync", "honesty"],
      "prompt": "LH318 TLV -> FRA was cancelled at the gate",
      "history": [],
      "request_override": {"party_size": 2},
      "local_now": {"offset_days": 0, "time": "22:15"},
      "live_data": true,
      "checks": ["trace_shape", "grounding_flights", "grounding_hotels", "date_sync", "no_booking_site", "price_honesty", "meals_honesty"],
      "expect": {"accommodation_dispatched": true}
    },
    {
      "case_id": "lhr-jfk-cancelled",
      "title": "Long-haul, different region",
      "probes": ["grounding", "coverage_outside_tlv"],
      "prompt": "BA117 LHR -> JFK was cancelled",
      "history": [],
      "request_override": {"party_size": 1},
      "local_now": {"offset_days": 0, "time": "21:00"},
      "live_data": true,
      "checks": ["trace_shape", "grounding_flights", "grounding_hotels", "date_sync", "no_booking_site", "price_honesty", "meals_honesty"],
      "expect": {"accommodation_dispatched": true}
    },
    {
      "case_id": "thin-route",
      "title": "Route with few departures",
      "probes": ["window_widening", "thin_candidates"],
      "prompt": "LY315 TLV -> NCE was cancelled",
      "history": [],
      "request_override": {"party_size": 1},
      "local_now": {"offset_days": 0, "time": "22:00"},
      "live_data": true,
      "checks": ["trace_shape", "grounding_flights", "no_booking_site", "price_honesty"],
      "expect": {}
    },
    {
      "case_id": "overnight-one-night",
      "title": "Late-evening disruption, morning flight",
      "probes": ["date_sync"],
      "prompt": "LH318 TLV -> FRA was cancelled at the gate",
      "history": [],
      "request_override": {"party_size": 2},
      "local_now": {"offset_days": 0, "time": "23:30"},
      "live_data": true,
      "checks": ["trace_shape", "date_sync", "grounding_hotels", "meals_honesty"],
      "expect": {"accommodation_dispatched": true}
    },
    {
      "case_id": "multi-night",
      "title": "Infrequent route, possible multi-night strand",
      "probes": ["date_sync_multi_night"],
      "prompt": "LY315 TLV -> NCE was cancelled",
      "history": [],
      "request_override": {"party_size": 2},
      "local_now": {"offset_days": 0, "time": "23:45"},
      "live_data": true,
      "checks": ["trace_shape", "date_sync"],
      "expect": {}
    },
    {
      "case_id": "same-day-no-hotel",
      "title": "Replacement leaves the same day",
      "probes": ["date_sync_skip"],
      "prompt": "LH318 TLV -> FRA was cancelled at the gate",
      "history": [],
      "request_override": {"party_size": 1},
      "local_now": {"offset_days": 0, "time": "05:00"},
      "live_data": true,
      "checks": ["trace_shape", "grounding_flights", "no_accommodation"],
      "expect": {"accommodation_dispatched": false}
    },
    {
      "case_id": "degraded-flights",
      "title": "Live data switched off",
      "probes": ["degraded_labelling"],
      "prompt": "LH318 TLV -> FRA was cancelled at the gate",
      "history": [],
      "request_override": {"party_size": 1},
      "local_now": {"offset_days": 0, "time": "22:15"},
      "live_data": false,
      "checks": ["trace_shape", "degraded_labelling", "no_booking_site"],
      "expect": {}
    },
    {
      "case_id": "sparse-osm-airport",
      "title": "Airport OpenStreetMap barely covers",
      "probes": ["hotel_coverage_limits"],
      "prompt": "LY123 VDA -> TLV was cancelled",
      "history": [],
      "request_override": {"party_size": 1, "stranded_at": "VDA"},
      "local_now": {"offset_days": 0, "time": "22:15"},
      "live_data": true,
      "checks": ["trace_shape"],
      "expect": {}
    },
    {
      "case_id": "baggage-followup",
      "title": "Ski bag question on a follow-up turn",
      "probes": ["deferral", "scope_discipline"],
      "prompt": "can I take my ski bag on the 09:40?",
      "history": [
        {"prompt": "LH318 TLV -> FRA was cancelled at the gate", "response": "Onward flight: LH 687, TLV to FRA."}
      ],
      "request_override": {"party_size": 2, "airline": "LH", "flight_number": "LH318", "origin": "TLV", "destination": "FRA", "stranded_at": "TLV", "disruption": "cancelled"},
      "local_now": {"offset_days": 0, "time": "22:20"},
      "live_data": true,
      "checks": ["trace_shape", "deferral", "history_reached_agents"],
      "expect": {}
    },
    {
      "case_id": "price-question",
      "title": "Passenger asks what it will cost",
      "probes": ["price_honesty"],
      "prompt": "how much will the new flight cost me?",
      "history": [
        {"prompt": "LH318 TLV -> FRA was cancelled at the gate", "response": "Onward flight: LH 687, TLV to FRA."}
      ],
      "request_override": {"party_size": 2, "airline": "LH", "flight_number": "LH318", "origin": "TLV", "destination": "FRA", "stranded_at": "TLV", "disruption": "cancelled"},
      "local_now": {"offset_days": 0, "time": "22:25"},
      "live_data": true,
      "checks": ["trace_shape", "price_honesty", "no_asserted_fare", "history_reached_agents"],
      "expect": {}
    },
    {
      "case_id": "earlier-flight-followup",
      "title": "Asks for something earlier",
      "probes": ["multi_turn"],
      "prompt": "is there anything earlier than that?",
      "history": [
        {"prompt": "LH318 TLV -> FRA was cancelled at the gate", "response": "Onward flight: LH 687, departs 16:30."}
      ],
      "request_override": {"party_size": 2, "airline": "LH", "flight_number": "LH318", "origin": "TLV", "destination": "FRA", "stranded_at": "TLV", "disruption": "cancelled"},
      "local_now": {"offset_days": 0, "time": "22:30"},
      "live_data": true,
      "checks": ["trace_shape", "grounding_flights", "history_reached_agents"],
      "expect": {}
    },
    {
      "case_id": "compare-options",
      "title": "Asks to compare two options",
      "probes": ["multi_turn"],
      "prompt": "compare the two options for me",
      "history": [
        {"prompt": "LH318 TLV -> FRA was cancelled at the gate", "response": "Onward flight: LH 687, departs 16:30."}
      ],
      "request_override": {"party_size": 2, "airline": "LH", "flight_number": "LH318", "origin": "TLV", "destination": "FRA", "stranded_at": "TLV", "disruption": "cancelled"},
      "local_now": {"offset_days": 0, "time": "22:35"},
      "live_data": true,
      "checks": ["trace_shape", "grounding_flights", "history_reached_agents"],
      "expect": {}
    }
  ]
}
```

- [ ] **Step 4: Write the loader**

Create `evals/search_cases.py`:

```python
"""Load and validate the search-agent validation scenarios.

Mirrors evals/documentation_cases.py so both halves of the project describe
their evaluation cases the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

CASES_PATH = Path(__file__).with_name("search_agent_cases.json")

CHECK_NAMES = {
    "trace_shape",
    "grounding_flights",
    "grounding_hotels",
    "date_sync",
    "no_accommodation",
    "no_booking_site",
    "price_honesty",
    "no_asserted_fare",
    "meals_honesty",
    "degraded_labelling",
    "deferral",
    "history_reached_agents",
}

REQUIRED_FIELDS = {
    "case_id", "title", "probes", "prompt", "history",
    "request_override", "local_now", "live_data", "checks", "expect",
}


def load_search_cases(path: Path = CASES_PATH) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported search evaluation schema version")

    cases: dict[str, dict] = {}
    for case in data.get("cases", []):
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("A case is missing case_id")
        if case_id in cases:
            raise ValueError(f"duplicate case_id: {case_id!r}")

        missing = REQUIRED_FIELDS - set(case)
        if missing:
            raise ValueError(f"{case_id} is missing: {', '.join(sorted(missing))}")
        if not case["probes"] or not case["title"]:
            raise ValueError(f"{case_id} must declare a title and what it probes")

        unknown = set(case["checks"]) - CHECK_NAMES
        if unknown:
            raise ValueError(f"{case_id} names unknown checks: {sorted(unknown)}")

        local_now = case["local_now"]
        if not isinstance(local_now.get("offset_days"), int):
            raise ValueError(f"{case_id} local_now.offset_days must be an int")
        hour, _, minute = str(local_now.get("time", "")).partition(":")
        if not (hour.isdigit() and minute.isdigit()):
            raise ValueError(f"{case_id} local_now.time must be HH:MM")

        cases[case_id] = case

    if not cases:
        raise ValueError("No search evaluation cases were found")
    return cases
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_search_cases.py -q`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add evals/search_agent_cases.json evals/search_cases.py tests/test_search_cases.py
git commit -m "Define the search-agent validation scenarios

Twelve cases, half of them aimed at the edges: degraded mode, an airport
OpenStreetMap barely covers, a thin route, and follow-ups that must be
deferred rather than answered.

local_now is explicit per case because the Supervisor's stub derives it from
datetime.now(), so a scenario about a late-evening disruption cannot be
driven from the prompt alone."
```

---

### Task 2: The offline evaluator

Built **before** the runner deliberately: the checks decide whether the live run was a pass, so proving them against synthetic artifacts costs nothing and avoids spending money on checks that were never exercised.

**Files:**
- Create: `evals/search_artifact.py`, `scripts/evaluate_search_artifact.py`
- Test: `tests/test_search_artifact.py`

**Interfaces:**
- Consumes: `evals.search_cases.load_search_cases`.
- Produces: `evals.search_artifact.evaluate(artifact: dict, case: dict) -> list[dict]`, each result `{"check": str, "status": "pass"|"fail"|"skip", "detail": str}`. Also `evals.search_artifact.candidates_from_prompt(user_prompt: str) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_artifact.py`:

```python
"""The checks that decide whether a live run passed.

Run against synthetic artifacts so they are proven before any money is spent.
"""

import json

from evals.search_artifact import candidates_from_prompt, evaluate

FLIGHT_CANDIDATES = [
    {"flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH", "origin": "TLV",
     "destination": "FRA", "depart": "2026-08-11T16:30+03:00", "arrive": "2026-08-11T20:10+02:00",
     "status": "Expected", "aircraft": "Airbus A320", "terminal": "3"},
]
HOTEL_CANDIDATES = [
    {"name": "Airport Plaza", "distance_km": 2.4, "stars": "4", "breakfast": None, "area": "Lod"},
]


def flight_prompt(candidates=FLIGHT_CANDIDATES):
    body = ("Candidates (real, verified departures - choose only from these):\n"
            + json.dumps(candidates)) if candidates else "No live schedule data was available"
    return "Route: TLV -> FRA\nLocal time now: 2026-08-10T22:15\n\n" + body


def hotel_prompt(candidates=HOTEL_CANDIDATES, check_in="2026-08-10",
                 check_out="2026-08-11", nights=1):
    body = ("Hotels (real, near the airport - choose only from these):\n"
            + json.dumps(candidates)) if candidates else "No live hotel data was available"
    return (f"Check in: {check_in}\nCheck out: {check_out}\nNights: {nights}\n\n" + body)


def step(module, user_prompt, response):
    return {"module": module,
            "prompt": {"system_prompt": "sys", "user_prompt": user_prompt},
            "response": response}


def artifact_with(flight_options, hotel_options, flight_prompt_text=None,
                  hotel_prompt_text=None, response="Onward flight: LH 687."):
    flight_payload = {"options": flight_options, "recommended_id": "F1"}
    steps = [step("Supervisor", "refine", {}),
             step("FlightAgent", flight_prompt_text or flight_prompt(), flight_payload)]
    if hotel_options is not None:
        steps.append(step("AccommodationAgent", hotel_prompt_text or hotel_prompt(),
                          {"options": hotel_options, "recommended_id": "H1"}))
    steps.append(step("Supervisor", "compose", {"text": response}))
    return {"status": "ok", "response": response, "steps": steps}


GOOD_FLIGHT = {"id": "F1", "airline": "Lufthansa", "flight_number": "LH 687",
               "origin": "TLV", "destination": "FRA",
               "depart": "2026-08-11T16:30+03:00", "arrive": "2026-08-11T20:10+02:00",
               "stops": 0, "fare_conditions": "See your Contract of Carriage.",
               "notes": "Nonstop."}
GOOD_HOTEL = {"id": "H1", "name": "Airport Plaza", "area": "2.4 km from the terminal",
              "check_in": "2026-08-10", "check_out": "2026-08-11", "nights": 1,
              "price_estimate": "Roughly EUR 120 (estimate)", "meals_included": False,
              "notes": "Meals not confirmed - check at the desk."}


def case_with(*checks):
    return {"case_id": "t", "checks": list(checks), "expect": {}}


def status_of(results, name):
    return next(r["status"] for r in results if r["check"] == name)


# --- candidate parsing ----------------------------------------------------------


def test_candidates_are_recovered_from_the_prompt():
    assert candidates_from_prompt(flight_prompt())[0]["flight"] == "LH 687"


def test_degraded_prompt_has_no_candidates():
    assert candidates_from_prompt(flight_prompt(candidates=None)) == []


# --- grounding ------------------------------------------------------------------


def test_grounding_passes_when_the_flight_came_from_the_candidates():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]),
                       case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "pass"


def test_grounding_fails_on_an_invented_flight():
    invented = dict(GOOD_FLIGHT, flight_number="XX 999")
    results = evaluate(artifact_with([invented], [GOOD_HOTEL]),
                       case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "fail"


def test_grounding_ignores_spacing_differences_in_flight_numbers():
    spaced = dict(GOOD_FLIGHT, flight_number="LH687")
    results = evaluate(artifact_with([spaced], [GOOD_HOTEL]),
                       case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "pass"


def test_hotel_grounding_fails_on_an_invented_property():
    invented = dict(GOOD_HOTEL, name="Imaginary Suites")
    results = evaluate(artifact_with([GOOD_FLIGHT], [invented]),
                       case_with("grounding_hotels"))
    assert status_of(results, "grounding_hotels") == "fail"


def test_grounding_is_skipped_when_there_were_no_candidates():
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=flight_prompt(candidates=None))
    results = evaluate(art, case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "skip"


# --- date sync ------------------------------------------------------------------


def test_date_sync_passes_when_the_hotel_matches_the_window():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]), case_with("date_sync"))
    assert status_of(results, "date_sync") == "pass"


def test_date_sync_fails_on_the_wrong_nights():
    wrong = dict(GOOD_HOTEL, check_out="2026-08-13", nights=3)
    results = evaluate(artifact_with([GOOD_FLIGHT], [wrong]), case_with("date_sync"))
    assert status_of(results, "date_sync") == "fail"


# --- honesty --------------------------------------------------------------------


def test_price_honesty_fails_on_an_unhedged_figure():
    blunt = dict(GOOD_HOTEL, price_estimate="EUR 120")
    results = evaluate(artifact_with([GOOD_FLIGHT], [blunt]), case_with("price_honesty"))
    assert status_of(results, "price_honesty") == "fail"


def test_no_asserted_fare_fails_when_flightagent_quotes_money():
    quoting = dict(GOOD_FLIGHT, fare_conditions="Rebooking costs EUR 90.")
    results = evaluate(artifact_with([quoting], [GOOD_HOTEL]), case_with("no_asserted_fare"))
    assert status_of(results, "no_asserted_fare") == "fail"


def test_meals_honesty_fails_when_meals_are_claimed_without_evidence():
    claimed = dict(GOOD_HOTEL, meals_included=True, notes="Breakfast included.")
    results = evaluate(artifact_with([GOOD_FLIGHT], [claimed]), case_with("meals_honesty"))
    assert status_of(results, "meals_honesty") == "fail"


def test_meals_honesty_passes_when_the_candidate_data_supports_it():
    art = artifact_with(
        [GOOD_FLIGHT], [dict(GOOD_HOTEL, meals_included=True, notes="Breakfast from 05:30.")],
        hotel_prompt_text=hotel_prompt(
            candidates=[dict(HOTEL_CANDIDATES[0], breakfast="yes")]),
    )
    results = evaluate(art, case_with("meals_honesty"))
    assert status_of(results, "meals_honesty") == "pass"


def test_no_booking_site_fails_when_one_is_named():
    leaky = dict(GOOD_HOTEL, notes="Cheaper on booking.com")
    results = evaluate(artifact_with([GOOD_FLIGHT], [leaky]), case_with("no_booking_site"))
    assert status_of(results, "no_booking_site") == "fail"


def test_deferral_fails_when_a_baggage_allowance_is_asserted():
    asserting = dict(GOOD_FLIGHT, notes="Your ski bag is fine, allowance is 23kg.")
    results = evaluate(artifact_with([asserting], [GOOD_HOTEL]), case_with("deferral"))
    assert status_of(results, "deferral") == "fail"


def test_deferral_passes_when_it_points_at_the_contract():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]), case_with("deferral"))
    assert status_of(results, "deferral") == "pass"


# --- degraded and trace ---------------------------------------------------------


def test_degraded_labelling_requires_the_illustrative_note():
    art = artifact_with([GOOD_FLIGHT], None,
                        flight_prompt_text=flight_prompt(candidates=None))
    results = evaluate(art, case_with("degraded_labelling"))
    assert status_of(results, "degraded_labelling") == "fail"

    labelled = dict(GOOD_FLIGHT, notes="Illustrative option - not live availability.")
    art = artifact_with([labelled], None,
                        flight_prompt_text=flight_prompt(candidates=None))
    results = evaluate(art, case_with("degraded_labelling"))
    assert status_of(results, "degraded_labelling") == "pass"


def test_trace_shape_requires_one_step_per_agent():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]), case_with("trace_shape"))
    assert status_of(results, "trace_shape") == "pass"


def test_trace_shape_fails_on_an_unknown_module():
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL])
    art["steps"].append(step("MysteryAgent", "x", {}))
    results = evaluate(art, case_with("trace_shape"))
    assert status_of(results, "trace_shape") == "fail"


def test_no_accommodation_passes_when_the_stay_was_skipped():
    results = evaluate(artifact_with([GOOD_FLIGHT], None), case_with("no_accommodation"))
    assert status_of(results, "no_accommodation") == "pass"


def test_history_reached_agents_checks_every_agent_prompt():
    earlier = "\nEarlier in this conversation:\n  passenger: hi"
    with_history = {"case_id": "t", "checks": ["history_reached_agents"], "expect": {},
                    "history": [{"prompt": "hi", "response": "ok"}]}

    # Both agents append history, so both prompts must carry it.
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=flight_prompt() + earlier,
                        hotel_prompt_text=hotel_prompt() + earlier)
    assert status_of(evaluate(art, with_history), "history_reached_agents") == "pass"


def test_history_reached_agents_fails_when_one_agent_missed_it():
    earlier = "\nEarlier in this conversation:\n  passenger: hi"
    with_history = {"case_id": "t", "checks": ["history_reached_agents"], "expect": {},
                    "history": [{"prompt": "hi", "response": "ok"}]}

    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=flight_prompt() + earlier)
    assert status_of(evaluate(art, with_history), "history_reached_agents") == "fail"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search_artifact.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.search_artifact'`

- [ ] **Step 3: Write the evaluator**

Create `evals/search_artifact.py`:

```python
"""Mechanical checks over a saved live-run artifact.

Every check reads only the artifact, so it runs offline with no keys. The
candidates the agent was given are inside each step's user_prompt, which is
what makes grounding verifiable after the fact.

Checks assert *properties*, never specific flight numbers or counts: real
schedules change daily and a test pinned to today's timetable is a test that
fails tomorrow for no reason.
"""

from __future__ import annotations

import json
import re

MODULES = {"Supervisor", "FlightAgent", "AccommodationAgent", "DocumentationAgent"}

FLIGHT_MARKER = "Candidates (real, verified departures"
HOTEL_MARKER = "Hotels (real, near the airport"

BOOKING_SITES = (
    "booking.com", "expedia", "kayak", "skyscanner", "agoda", "hotels.com",
    "trivago", "momondo", "priceline", "orbitz", "travelocity", "airbnb",
    "tripadvisor", "kiwi.com", "opodo", "edreams",
)

HEDGES = ("estimate", "approx", "around", "roughly", "~", "about", "typically")
MONEY = re.compile(r"(?:eur|usd|gbp|ils|nis|\$|€|£|₪)\s?\d|\d+\s?(?:eur|usd|gbp|ils|nis)", re.I)
BAGGAGE_ASSERTION = re.compile(r"\d+\s?kg|\ballowance is\b|\byou may (?:bring|carry)\b", re.I)
CONTRACT_POINTER = re.compile(r"contract of carriage|conditions of carriage|entitlement", re.I)


def candidates_from_prompt(user_prompt: str) -> list[dict]:
    """The candidate list the agent was actually given, or [] in degraded mode."""
    for marker in (FLIGHT_MARKER, HOTEL_MARKER):
        index = user_prompt.find(marker)
        if index == -1:
            continue
        start = user_prompt.find("[", index)
        if start == -1:
            continue
        depth, in_string, escaped = 0, False, False
        for position in range(start, len(user_prompt)):
            character = user_prompt[position]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
            elif character == '"':
                in_string = not in_string
            elif not in_string and character == "[":
                depth += 1
            elif not in_string and character == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(user_prompt[start:position + 1])
                    except json.JSONDecodeError:
                        return []
    return []


def _step(artifact: dict, module: str) -> dict | None:
    return next((s for s in artifact.get("steps", []) if s["module"] == module), None)


def _options(step: dict | None) -> list[dict]:
    if not step or not isinstance(step.get("response"), dict):
        return []
    return step["response"].get("options") or []


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _result(check: str, ok: bool, detail: str, skip: bool = False) -> dict:
    return {"check": check, "status": "skip" if skip else ("pass" if ok else "fail"),
            "detail": detail}


def _check_trace_shape(artifact, case):
    steps = artifact.get("steps", [])
    problems = []
    for step in steps:
        if step["module"] not in MODULES:
            problems.append(f"unknown module {step['module']!r}")
        if set(step) != {"module", "prompt", "response"}:
            problems.append(f"{step['module']} has keys {sorted(step)}")
        if set(step.get("prompt", {})) != {"system_prompt", "user_prompt"}:
            problems.append(f"{step['module']} prompt keys {sorted(step.get('prompt', {}))}")
    for module in ("FlightAgent", "AccommodationAgent"):
        count = sum(1 for s in steps if s["module"] == module)
        if count > 1:
            problems.append(f"{module} produced {count} steps, expected at most 1")
    return _result("trace_shape", not problems, "; ".join(problems) or f"{len(steps)} steps, all well-formed")


def _grounding(artifact, module, marker_key, option_key):
    step = _step(artifact, module)
    if not step:
        return _result(f"grounding_{marker_key}", True, f"{module} was not dispatched", skip=True)
    candidates = candidates_from_prompt(step["prompt"]["user_prompt"])
    if not candidates:
        return _result(f"grounding_{marker_key}", True, "degraded run, nothing to ground against", skip=True)

    allowed = {_normalise(c.get("flight") or c.get("name")) for c in candidates}
    invented = [o.get(option_key) for o in _options(step)
                if _normalise(o.get(option_key)) not in allowed]
    return _result(
        f"grounding_{marker_key}", not invented,
        f"invented: {invented}" if invented else f"all {len(_options(step))} from {len(candidates)} candidates",
    )


def _check_grounding_flights(artifact, case):
    return _grounding(artifact, "FlightAgent", "flights", "flight_number")


def _check_grounding_hotels(artifact, case):
    return _grounding(artifact, "AccommodationAgent", "hotels", "name")


def _check_date_sync(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    if not step:
        return _result("date_sync", True, "no stay was needed", skip=True)
    prompt = step["prompt"]["user_prompt"]

    def field(label):
        match = re.search(rf"^{label}:\s*(.+)$", prompt, re.M)
        return match.group(1).strip() if match else None

    wanted_in, wanted_out, wanted_nights = field("Check in"), field("Check out"), field("Nights")
    problems = []
    for option in _options(step):
        if option.get("check_in") != wanted_in or option.get("check_out") != wanted_out:
            problems.append(f"{option.get('id')}: {option.get('check_in')}..{option.get('check_out')}")
        if str(option.get("nights")) != str(wanted_nights):
            problems.append(f"{option.get('id')}: nights {option.get('nights')} != {wanted_nights}")
    return _result("date_sync", not problems,
                   "; ".join(problems) or f"all options on {wanted_in}..{wanted_out} ({wanted_nights}n)")


def _check_no_accommodation(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    return _result("no_accommodation", step is None,
                   "AccommodationAgent ran when no stay was needed" if step else "correctly skipped")


def _check_no_booking_site(artifact, case):
    haystack = json.dumps(artifact.get("steps", []), ensure_ascii=False).lower()
    haystack += str(artifact.get("response", "")).lower()
    named = [site for site in BOOKING_SITES if site in haystack]
    return _result("no_booking_site", not named, f"named: {named}" if named else "none named")


def _check_price_honesty(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    if not step:
        return _result("price_honesty", True, "no stay was proposed", skip=True)
    blunt = [o.get("price_estimate") for o in _options(step)
             if not any(h in str(o.get("price_estimate", "")).lower() for h in HEDGES)]
    return _result("price_honesty", not blunt,
                   f"unhedged: {blunt}" if blunt else "every price is hedged")


def _check_no_asserted_fare(artifact, case):
    step = _step(artifact, "FlightAgent")
    if not step:
        return _result("no_asserted_fare", True, "FlightAgent was not dispatched", skip=True)
    # FlightAgent has no fare data at all, so any money figure it prints is invented.
    quoting = [f"{o.get('id')}: {o.get('fare_conditions')}" for o in _options(step)
               if MONEY.search(str(o.get("fare_conditions", "")) + str(o.get("notes", "")))]
    return _result("no_asserted_fare", not quoting,
                   f"quoted money: {quoting}" if quoting else "no fares asserted")


def _check_meals_honesty(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    if not step:
        return _result("meals_honesty", True, "no stay was proposed", skip=True)
    candidates = {_normalise(c.get("name")): c for c in candidates_from_prompt(step["prompt"]["user_prompt"])}
    problems = []
    for option in _options(step):
        candidate = candidates.get(_normalise(option.get("name")))
        if option.get("meals_included"):
            if not (candidate and candidate.get("breakfast")):
                problems.append(f"{option.get('id')} claims meals with no supporting tag")
        elif "confirm" not in str(option.get("notes", "")).lower():
            problems.append(f"{option.get('id')} does not say meals were unconfirmed")
    return _result("meals_honesty", not problems, "; ".join(problems) or "meals stated honestly")


def _check_degraded_labelling(artifact, case):
    problems = []
    checked = 0
    for module in ("FlightAgent", "AccommodationAgent"):
        step = _step(artifact, module)
        if not step or candidates_from_prompt(step["prompt"]["user_prompt"]):
            continue
        for option in _options(step):
            checked += 1
            if "illustrative" not in str(option.get("notes", "")).lower():
                problems.append(f"{module}/{option.get('id')} is unlabelled")
    if not checked:
        return _result("degraded_labelling", True, "nothing ran degraded", skip=True)
    return _result("degraded_labelling", not problems,
                   "; ".join(problems) or f"all {checked} options labelled illustrative")


def _check_deferral(artifact, case):
    step = _step(artifact, "FlightAgent")
    if not step:
        return _result("deferral", True, "FlightAgent was not dispatched", skip=True)
    text = json.dumps(_options(step), ensure_ascii=False)
    asserted = BAGGAGE_ASSERTION.search(text)
    if asserted:
        return _result("deferral", False, f"asserted a carriage rule: {asserted.group(0)!r}")
    return _result("deferral", bool(CONTRACT_POINTER.search(text)),
                   "points at the Contract of Carriage" if CONTRACT_POINTER.search(text)
                   else "neither asserted nor deferred - no pointer to the contract")


def _check_history_reached_agents(artifact, case):
    if not case.get("history"):
        return _result("history_reached_agents", True, "case has no history", skip=True)
    missing = [s["module"] for s in artifact.get("steps", [])
               if s["module"] in ("FlightAgent", "AccommodationAgent")
               and "Earlier in this conversation" not in s["prompt"]["user_prompt"]]
    return _result("history_reached_agents", not missing,
                   f"history missing from: {missing}" if missing else "history reached every agent")


CHECKS = {
    "trace_shape": _check_trace_shape,
    "grounding_flights": _check_grounding_flights,
    "grounding_hotels": _check_grounding_hotels,
    "date_sync": _check_date_sync,
    "no_accommodation": _check_no_accommodation,
    "no_booking_site": _check_no_booking_site,
    "price_honesty": _check_price_honesty,
    "no_asserted_fare": _check_no_asserted_fare,
    "meals_honesty": _check_meals_honesty,
    "degraded_labelling": _check_degraded_labelling,
    "deferral": _check_deferral,
    "history_reached_agents": _check_history_reached_agents,
}


def evaluate(artifact: dict, case: dict) -> list[dict]:
    """Run the checks this case declares. Returns one result per check."""
    return [CHECKS[name](artifact, case) for name in case.get("checks", [])]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_search_artifact.py -q`
Expected: 22 passed.

- [ ] **Step 5: Write the CLI wrapper**

Create `scripts/evaluate_search_artifact.py`:

```python
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
        print(f"\n=== {run['scenario']} — {case['title']} ({run.get('status')}) ===")
        for result in results:
            mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip"}[result["status"]]
            print(f"  [{mark}] {result['check']}: {result['detail']}")
            failed += result["status"] == "fail"

    print(f"\n{failed} failing check(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Verify the wrapper runs on a synthetic artifact**

`tests/` is not an importable package, so build the synthetic artifact inline rather than
importing the test helpers:

```bash
python -c "
import json, pathlib
candidates = [{'flight': 'LH 687', 'airline': 'Lufthansa', 'origin': 'TLV',
               'destination': 'FRA', 'depart': '2026-08-11T16:30+03:00',
               'arrive': '2026-08-11T20:10+02:00'}]
hotels = [{'name': 'Airport Plaza', 'distance_km': 2.4, 'stars': '4',
           'breakfast': None, 'area': 'Lod'}]
def step(module, prompt, response):
    return {'module': module,
            'prompt': {'system_prompt': 'sys', 'user_prompt': prompt},
            'response': response}
fp = 'Candidates (real, verified departures - choose only from these):\n' + json.dumps(candidates)
hp = ('Check in: 2026-08-10\nCheck out: 2026-08-11\nNights: 1\n\n'
      'Hotels (real, near the airport - choose only from these):\n' + json.dumps(hotels))
art = {
  'scenario': 'tlv-fra-cancelled', 'status': 'ok', 'response': 'Onward flight: LH 687.',
  'steps': [
    step('Supervisor', 'refine', {}),
    step('FlightAgent', fp, {'options': [{'id': 'F1', 'flight_number': 'LH 687',
         'fare_conditions': 'See your Contract of Carriage.', 'notes': 'Nonstop.'}],
         'recommended_id': 'F1'}),
    step('AccommodationAgent', hp, {'options': [{'id': 'H1', 'name': 'Airport Plaza',
         'check_in': '2026-08-10', 'check_out': '2026-08-11', 'nights': 1,
         'price_estimate': 'Roughly EUR 120 (estimate)', 'meals_included': False,
         'notes': 'Meals not confirmed - check at the desk.'}], 'recommended_id': 'H1'}),
    step('Supervisor', 'compose', {'text': 'Onward flight: LH 687.'}),
  ],
}
pathlib.Path('live-test-output').mkdir(exist_ok=True)
pathlib.Path('live-test-output/synthetic.json').write_text(json.dumps(art), encoding='utf-8')
"
python scripts/evaluate_search_artifact.py --artifact live-test-output/synthetic.json
```
Expected: every declared check prints PASS or skip, and it reports `0 failing check(s)`.

- [ ] **Step 7: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass (109 existing + 6 + 20 new).

- [ ] **Step 8: Commit**

```bash
git add evals/search_artifact.py scripts/evaluate_search_artifact.py tests/test_search_artifact.py
git commit -m "Add mechanical checks for search-agent live runs

Built before the runner on purpose: these checks decide whether a live run
passed, so proving them against synthetic artifacts costs nothing and avoids
spending budget on checks that were never exercised.

Grounding is decidable after the fact because the candidates the agent was
given are inside its own user_prompt, so an invented flight number is caught
without re-calling anything.

Checks assert properties, never specific flights or counts - real schedules
change daily and a test pinned to today's timetable fails tomorrow for no
reason."
```

---

### Task 3: The live runner

**Files:**
- Create: `scripts/run_search_agents_live.py`
- Test: `tests/test_search_runner.py`

**Interfaces:**
- Consumes: `evals.search_cases.load_search_cases`, `lib.agents.supervisor`, `lib.llm`.
- Produces: `scripts.run_search_agents_live.build_request_patch(case: dict, original) -> callable` and `run_case(case: dict) -> dict` returning an artifact with keys `scenario`, `status`, `response`, `steps`, `payloads_seen`, `observed`, `chat_usage`, `elapsed_seconds`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_runner.py`:

```python
"""The runner's wiring, exercised with fakes so it costs nothing."""

import importlib.util
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "run_search_agents_live", REPO_ROOT / "scripts" / "run_search_agents_live.py")
runner = importlib.util.module_from_spec(spec)
sys.modules["run_search_agents_live"] = runner
spec.loader.exec_module(runner)


CASE = {
    "case_id": "t", "title": "t", "probes": ["x"],
    "prompt": "LH318 TLV -> FRA was cancelled at the gate",
    "history": [], "request_override": {"party_size": 4},
    "local_now": {"offset_days": 1, "time": "23:30"},
    "live_data": True, "checks": [], "expect": {},
}


def test_local_now_is_injected_from_the_case():
    def original(prompt, history):
        return {"party_size": 1, "local_now": "ignored", "flight_number": "LH318",
                "origin": "TLV", "destination": "FRA", "disruption": "cancelled",
                "stranded_at": "TLV"}, {"module": "Supervisor"}

    patched = runner.build_request_patch(CASE, original)
    request, _ = patched("prompt", [])

    expected_date = date.today() + timedelta(days=1)
    assert request["local_now"].startswith(expected_date.isoformat())
    assert request["local_now"].endswith("23:30:00")


def test_request_override_wins():
    def original(prompt, history):
        return {"party_size": 1, "local_now": "x", "flight_number": "LH318",
                "origin": "TLV", "destination": "FRA", "disruption": "cancelled",
                "stranded_at": "TLV"}, {"module": "Supervisor"}

    request, _ = runner.build_request_patch(CASE, original)("prompt", [])
    assert request["party_size"] == 4


def test_missing_is_recomputed_so_the_gate_still_works():
    def original(prompt, history):
        return {"party_size": 1, "local_now": "x", "flight_number": None,
                "origin": None, "destination": None, "disruption": None,
                "stranded_at": None}, {"module": "Supervisor"}

    case = dict(CASE, request_override={})
    request, _ = runner.build_request_patch(case, original)("prompt", [])
    assert "flight_number" in request["missing"]


def test_call_ceiling_raises_before_overspending():
    with pytest.raises(RuntimeError, match="call ceiling"):
        runner.guard_budget(used=30, ceiling=30)


def test_call_ceiling_allows_headroom():
    runner.guard_budget(used=10, ceiling=30)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_search_runner.py -q`
Expected: FAIL — `scripts/run_search_agents_live.py` does not exist.

- [ ] **Step 3: Write the runner**

Create `scripts/run_search_agents_live.py`:

```python
"""Run explicitly confirmed live search-agent scenarios and save their traces.

Drives the real Supervisor so the date sync - the only thing connecting
FlightAgent and AccommodationAgent - is exercised rather than bypassed.
DocumentationAgent is replaced with the same no-cost fake the unit tests use,
so nothing here spends on Person C's half or needs Pinecone.

Budget: AeroDataBox is 2 units per call from a 600/month allowance, and every
chat call costs against a $13 project ceiling. `--all` runs in one process so
the route cache is shared and repeated routes cost nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, time as clock, timedelta
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.search_cases import load_search_cases  # noqa: E402
from lib import llm  # noqa: E402
from lib.agents import documentation_agent, supervisor  # noqa: E402
from lib.steps import make_step  # noqa: E402
from lib.tools import flights, hotels  # noqa: E402

CASES = load_search_cases()

DOC_PAYLOAD = {"regulation": "EU 261/2004", "entitlements": [],
               "next_actions": [], "caveats": ["Faked for search-agent validation."]}


def fake_documentation_run(request, history):
    """DocumentationAgent is Person C's; validating it here would spend on their half."""
    steps = [make_step("DocumentationAgent", phase, "faked for search validation", DOC_PAYLOAD)
             for phase in ("draft", "critique", "refine")]
    return DOC_PAYLOAD, steps


def guard_budget(used: int, ceiling: int) -> None:
    if used >= ceiling:
        raise RuntimeError(
            f"LLM call ceiling reached ({used}/{ceiling}). Raise --max-llm-calls deliberately.")


def build_request_patch(case: dict, original):
    """Wrap the Supervisor's extraction so a scenario can fix local_now and fields.

    local_now cannot be driven from the prompt: the Supervisor's stub derives it
    from datetime.now(), so 'a late-evening disruption' is not expressible as text.
    Real extraction still runs; only the named fields are overridden.
    """
    when = datetime.combine(
        date.today() + timedelta(days=case["local_now"]["offset_days"]),
        clock.fromisoformat(case["local_now"]["time"]),
    )

    def patched(prompt, history):
        request, step = original(prompt, history)
        request.update(case["request_override"])
        request["local_now"] = when.isoformat(timespec="seconds")
        request["missing"] = [f for f in supervisor.REQUIRED_FIELDS if not request.get(f)]
        return request, step

    return patched


def run_case(case: dict) -> dict:
    os.environ["WINGMAN_LIVE_DATA"] = "1" if case["live_data"] else "0"

    original_extract = supervisor._extract_request
    original_doc_run = documentation_agent.run
    supervisor._extract_request = build_request_patch(case, original_extract)
    documentation_agent.run = fake_documentation_run

    started = time.monotonic()
    artifact = {"scenario": case["case_id"], "title": case["title"],
                "probes": case["probes"], "live_data": case["live_data"],
                "started_at": datetime.now().astimezone().isoformat(),
                "model": llm.TEXT_MODEL}
    try:
        response, steps = supervisor.run(case["prompt"], list(case["history"]))
        artifact.update({"status": "ok", "response": response, "steps": steps})
    except Exception as exc:  # noqa: BLE001 - the artifact must record any failure
        artifact.update({"status": "error", "error": str(exc),
                         "steps": getattr(exc, "steps", [])})
    finally:
        supervisor._extract_request = original_extract
        documentation_agent.run = original_doc_run

    artifact["elapsed_seconds"] = round(time.monotonic() - started, 3)
    artifact["observed"] = _observed(artifact)
    return artifact


def _observed(artifact: dict) -> dict:
    """What the live data actually gave us, so the report can be honest.

    Scenarios like 'thin route' depend on conditions we cannot guarantee on the
    day; recording the real counts lets the report say when one did not reproduce.
    """
    from evals.search_artifact import candidates_from_prompt

    observed = {}
    for module, key in (("FlightAgent", "flight_candidates"),
                        ("AccommodationAgent", "hotel_candidates")):
        step = next((s for s in artifact.get("steps", []) if s["module"] == module), None)
        observed[key] = len(candidates_from_prompt(step["prompt"]["user_prompt"])) if step else None
    stay = next((s for s in artifact.get("steps", []) if s["module"] == "AccommodationAgent"), None)
    observed["accommodation_dispatched"] = stay is not None
    if stay:
        options = (stay.get("response") or {}).get("options") or []
        observed["nights"] = options[0].get("nights") if options else None
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid-calls", action="store_true",
                        help="Required: makes live chat-model and AeroDataBox calls.")
    parser.add_argument("--scenario", choices=tuple(CASES))
    parser.add_argument("--all", action="store_true",
                        help="Run every scenario in one process so the route cache is shared.")
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "live-test-output" / "search_agents_live.json")
    args = parser.parse_args()

    if not args.confirm_paid_calls:
        parser.error("Pass --confirm-paid-calls after reviewing the expected call scope.")
    if not (args.scenario or args.all):
        parser.error("Pass --scenario NAME or --all.")

    load_dotenv(REPO_ROOT / ".env", override=False)

    selected = list(CASES.values()) if args.all else [CASES[args.scenario]]
    runs = []
    for case in selected:
        guard_budget(llm.usage["calls"], args.max_llm_calls)
        print(f"--- {case['case_id']} (llm calls so far: {llm.usage['calls']})", flush=True)
        runs.append(run_case(case))

    bundle = {"started_at": datetime.now().astimezone().isoformat(),
              "model": llm.TEXT_MODEL, "runs": runs, "chat_usage": dict(llm.usage),
              "flight_cache_entries": len(flights._cache),
              "hotel_cache_entries": len(hotels._cache)}

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output), "scenarios": len(runs),
                      "chat_usage": bundle["chat_usage"],
                      "errors": [r["scenario"] for r in runs if r["status"] != "ok"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_search_runner.py -q`
Expected: 5 passed.

- [ ] **Step 5: Confirm the runner refuses to spend without the flag**

Run: `python scripts/run_search_agents_live.py --all`
Expected: exits non-zero with `Pass --confirm-paid-calls after reviewing the expected call scope.` and makes **no** network calls.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_search_agents_live.py tests/test_search_runner.py
git commit -m "Add the budget-gated live runner for the search agents

Drives the real Supervisor so the date sync is exercised rather than
bypassed, and fakes DocumentationAgent so nothing here spends on Person C's
half or needs Pinecone.

local_now is injected per scenario because the Supervisor's stub derives it
from datetime.now(), so 'a late-evening disruption' cannot be expressed as
prompt text.

Records what the live data actually returned - candidate counts, derived
nights - so the report can say honestly when a scenario did not reproduce
its intended condition."
```

---

### Task 4: Run it live, triage, fix hard failures

This is the task that spends money. Everything before it is free and proven.

**Files:**
- Modify (only if a hard check fails): `lib/agents/flight_agent.py`, `lib/agents/accommodation_agent.py`
- Create: `live-test-output/search_agents_live.json` (gitignored)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: a saved artifact bundle and a list of triaged findings for Task 5.

- [ ] **Step 1: Confirm the sparse-OSM airport choice for free**

Overpass costs nothing, so verify `VDA` is genuinely sparse before relying on it:

```bash
WINGMAN_LIVE_DATA=1 python -c "
from lib.tools import hotels, airports
for code in ('VDA', 'ETM', 'HFA'):
    print(code, airports.lookup(code) and airports.lookup(code)['name'], '->', len(hotels.search(code)), 'hotels')
"
```
If `VDA` returns 0–2 hotels, keep it. If it returns many, swap `sparse-osm-airport`'s
`request_override.stranded_at` and `prompt` to whichever code above is sparsest, and note
the substitution in the commit message.

- [ ] **Step 2: Check the quota before spending**

```bash
python -c "
import os, httpx; from dotenv import load_dotenv; load_dotenv('.env')
k, h = os.environ['AERODATABOX_API_KEY'], os.environ['AERODATABOX_API_HOST']
r = httpx.get(f'https://{h}/subscriptions/balance', headers={'x-rapidapi-key': k, 'x-rapidapi-host': h}, timeout=30)
print('units:', r.headers.get('x-ratelimit-api-units-remaining'), '/', r.headers.get('x-ratelimit-api-units-limit'))
"
```
Record the number. Expected around 592.

- [ ] **Step 3: Run all twelve scenarios**

```bash
python scripts/run_search_agents_live.py --confirm-paid-calls --all \
  --output live-test-output/search_agents_live.json
```
Expected: ~20 chat calls, printed running totals, and a bundle written. If the ceiling
trips, stop and report rather than raising it silently.

- [ ] **Step 4: Evaluate the run**

```bash
python scripts/evaluate_search_artifact.py --artifact live-test-output/search_agents_live.json
```
Record every FAIL verbatim, and the units consumed (re-run Step 2 and subtract).

- [ ] **Step 5: Triage each failure**

Classify each FAIL:

- **Hard** — breaks a rule the design commits to: an invented flight or hotel, hotel dates
  that do not match the stay window, a booking site named, a fare asserted by FlightAgent, a
  baggage allowance asserted, an unlabelled degraded option, a malformed trace.
  These get fixed.
- **Soft** — tone, phrasing, ordering, how verbose a note is. These get written into the
  report for the owner to decide on, and are **not** fixed here.

- [ ] **Step 6: Fix each hard failure in the prompt**

Hard failures are almost always a missing or too-weak instruction. Edit the relevant
`SYSTEM_PROMPT` in `lib/agents/flight_agent.py` or `lib/agents/accommodation_agent.py` —
state the rule explicitly and, where the one-shot example contradicts it, fix the example
too, since the example is the stronger signal.

Do not weaken a check to make it pass. If a check itself is wrong — it flags correct
behaviour — fix the check in `evals/search_artifact.py` and add a test for the corrected
behaviour in `tests/test_search_artifact.py`.

- [ ] **Step 7: Re-run only the affected scenarios**

```bash
python scripts/run_search_agents_live.py --confirm-paid-calls --scenario <case_id> \
  --output live-test-output/retry-<case_id>.json
python scripts/evaluate_search_artifact.py --artifact live-test-output/retry-<case_id>.json
```
Repeat per fixed scenario. Each retry is 1–2 chat calls and reuses the cached route.

- [ ] **Step 8: Confirm the unit suite is still green**

Run: `python -m pytest -q`
Expected: all pass. The existing tests guard the payload shapes the Supervisor depends on,
so a prompt fix that broke a shape shows up here.

- [ ] **Step 9: Commit any fixes**

```bash
git add lib/agents/ evals/ tests/
git commit -m "Fix rule violations found by live validation

<one line per fix: which rule the model broke, and what changed in the prompt>"
```
If nothing needed fixing, skip this step and say so explicitly in the report.

---

### Task 5: The capabilities report

**Files:**
- Create: `docs/search-agents-capabilities.md`

**Interfaces:**
- Consumes: the artifacts and triage from Task 4.
- Produces: no code.

- [ ] **Step 1: Write the report**

Create `docs/search-agents-capabilities.md` with these sections, filled from the real run:

1. **What was validated** — date, model, scenario count, chat calls, tokens, AeroDataBox
   units consumed. State the harness commands so anyone can re-run it.
2. **Verified capabilities** — one row per capability, each naming the scenario that proves
   it and the check that passed. Cover: grounding in real departures, cross-airline options,
   the date sync deriving nights from the chosen flight, skipping the stay when none is
   needed, exact hotel distances, degraded-mode labelling, history reaching the agents.
3. **Limitations, grouped by root cause:**
   - *No free data source*: fares, seat availability, baggage and carriage terms.
   - *API shape*: direct flights only (departures, not itineraries); 12-hour windows; a
     search is capped at two calls.
   - *OpenStreetMap sparseness*: no prices, no availability, meals almost never known,
     ratings rare, names often local-language.
   - *Quota*: 2 units per call from 600/month.
   - *Blocked on Person A*: `_compose` is a stub, so a follow-up turn does not read as an
     answer to the question asked — the agents receive the history and act on it, but the
     final prose is assembled deterministically.
4. **Scenarios that did not reproduce their condition** — from `observed` in the artifact,
   e.g. if `multi-night` returned a one-night strand because the route ran daily. Say so
   plainly and name the unit test that covers the logic instead
   (`tests/test_supervisor.py::test_stay_window_spans_multiple_nights`).
5. **Failure modes** — what the passenger sees when AeroDataBox is down or out of quota,
   when Overpass 504s, when the LLM returns an unusable option, and when no key is set.
6. **Soft findings** — the style and tone observations from triage, as open questions.
7. **Open items for the team meeting** — including the two that are not Person B's:
   `/api/agent_info` returns `prompt_examples: []`, and its `description` does not state
   that prices are estimates. Both are blocked behind the Supervisor's stubbed seams,
   because capturing a trace now would bake two LLM-less `Supervisor` steps into a graded
   artifact.

Every capability claim must name the scenario that proves it. A claim with no scenario
behind it does not belong in the report.

- [ ] **Step 2: Verify every claim traces to evidence**

Re-read the report against the evaluator output. Delete or requalify any sentence that the
artifacts do not support.

- [ ] **Step 3: Commit**

```bash
git add docs/search-agents-capabilities.md
git commit -m "Record what the search agents can and cannot do

Written from a real run against a real model, not from the design intent.
Every capability names the scenario that proves it; limitations are grouped
by root cause so the team can see which are fixable and which follow from
there being no free source of fares or carriage terms in 2026."
```

---

## Verification before opening the PR

- [ ] `python -m pytest -q` — green, with no API keys configured.
- [ ] `python scripts/evaluate_search_artifact.py --artifact live-test-output/search_agents_live.json`
      — every hard check passes, or the report explains each remaining failure.
- [ ] AeroDataBox units consumed recorded and well inside budget.
- [ ] Chat calls and tokens recorded; total spend inside the $1 approved.
- [ ] Report contains no capability claim without a scenario behind it.
- [ ] PR description carries the items for the team meeting.
