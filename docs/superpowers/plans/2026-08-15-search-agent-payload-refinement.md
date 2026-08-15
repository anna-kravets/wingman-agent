# Search-Agent Payload Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both search agents return everything that matters in their domain, stop asking the model to transcribe facts it can only copy, and give the Supervisor a `caveats` list it can act on.

**Architecture:** The model picks options and writes prose; `_validate` matches each choice back to its source candidate and overwrites every factual field from it. Caveats whose trigger is a fact are generated in code. Both tools already return every field needed — no tool changes.

**Tech Stack:** Python 3.12+, `pytest`. No new dependencies.

**Spec:** [`../specs/2026-08-15-search-agent-payload-refinement-design.md`](../specs/2026-08-15-search-agent-payload-refinement-design.md)

## Global Constraints

- Module names locked: `Supervisor`, `FlightAgent`, `AccommodationAgent`, `DocumentationAgent`.
- One LLM call per search agent; one `steps[]` entry each. Tool calls produce none.
- Agent signatures do **not** change: `run(request, history)` and `run(request, stay_window, history)`.
- `caveats` is `list[str]`, every entry opening `NOTE:`, `ASK:` or `CONFIRM:`.
- Facts come from the candidate; the model supplies only choice, ordering and prose.
- Tests must pass with no API keys. `conftest.py` clears every credential.
- Never name a booking or flight-search website.
- $13 LLM budget, 600 AeroDataBox units/month (≈446 left), 2 units per call.
- **`tests/conftest.py` is shared with Person A's Supervisor tests** — changes there go in the handover, not just a commit message.

## File Structure

| File | Change |
|---|---|
| `lib/agents/flight_agent.py` | New payload, candidate matching + enrichment, code-generated caveats, rewritten prompt |
| `lib/agents/accommodation_agent.py` | Same, for hotels |
| `evals/search_artifact.py` | `no_asserted_fare` → `rebooking`; `meals_honesty` → tri-state; new `facts_from_candidates` check |
| `evals/search_cases.py` | Register the new check |
| `evals/search_agent_cases.json` | Apply it where flights are grounded |
| `tests/conftest.py` | `fake_llm` returns the new shape (**shared file**) |
| `tests/test_flight_agent.py`, `tests/test_accommodation_agent.py`, `tests/test_search_artifact.py` | Updated + new tests |
| `docs/PROJECT_PLAN.md` | §1 payload contract — **locked decision**, per CLAUDE.md §7 |
| `docs/search-agents-capabilities.md` | Record what changed |

Tools need no change: `flights.search` already returns `terminal`, `aircraft`, `status`, `airline_iata`; `hotels.search` already returns `phone`, `website`, `address`, `kind`, `stars`, `wheelchair`, `breakfast`, `distance_km`.

---

### Task 1: FlightAgent — enrichment from candidates

**Files:**
- Modify: `lib/agents/flight_agent.py`
- Test: `tests/test_flight_agent.py`

**Interfaces:**
- Consumes: `lib.tools.flights.search`, `lib.llm.call`.
- Produces: `flight_agent.run(request, history) -> (payload, steps)` where payload is
  `{"options": [...], "recommended_id": str, "caveats": list[str]}` and each option carries
  `id, airline, airline_iata, flight_number, origin, destination, depart, arrive,
  duration_minutes, arrives_next_day, terminal, aircraft, status, rebooking, notes`.
  Also `flight_agent._normalise(text) -> str`.

- [ ] **Step 1: Write the failing tests**

Replace the payload-shape tests in `tests/test_flight_agent.py` — keep the existing file's
imports, `REQUEST`, `CANDIDATE`, `fake_call` and the refusal tests, and replace `good_payload`
plus add these:

```python
def good_payload(flight_number="LH 687", recommended="F1"):
    """What the model now returns: a choice and prose, not a transcription."""
    return {
        "options": [{
            "id": "F1",
            "flight_number": flight_number,
            "rebooking": "Same airline as the cancelled flight.",
            "notes": "Earliest arrival.",
        }],
        "recommended_id": recommended,
    }


def test_facts_are_taken_from_the_candidate_not_the_model(monkeypatch):
    lying = {"options": [{
        "id": "F1", "flight_number": "LH 687",
        "depart": "2026-01-01T00:00:00", "arrive": "2026-01-01T01:00:00",
        "airline": "Wrong Airways", "origin": "XXX", "destination": "YYY",
        "rebooking": "", "notes": "",
    }], "recommended_id": "F1"}
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(lying))

    option = flight_agent.run(REQUEST, [])[0]["options"][0]

    # The model cannot corrupt a fact it is no longer asked to copy.
    assert option["depart"] == CANDIDATE["depart"]
    assert option["airline"] == "Lufthansa"
    assert option["origin"] == "TLV" and option["destination"] == "FRA"


def test_fields_the_tool_fetched_are_surfaced(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    option = flight_agent.run(REQUEST, [])[0]["options"][0]

    assert option["terminal"] == "3"
    assert option["aircraft"] == "Airbus A320"
    assert option["status"] == "Expected"
    assert option["airline_iata"] == "LH"


def test_duration_and_next_day_are_computed(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    option = flight_agent.run(REQUEST, [])[0]["options"][0]

    # 16:30+03:00 -> 20:10+02:00 is 4h40m of actual travel, same calendar day.
    assert option["duration_minutes"] == 280
    assert option["arrives_next_day"] is False


def test_an_overnight_flight_is_flagged(monkeypatch):
    overnight = dict(CANDIDATE, flight="LH 999",
                     depart="2026-08-10T23:30+03:00", arrive="2026-08-11T03:10+02:00")
    monkeypatch.setattr(flights, "search", lambda *a, **k: [overnight])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(flight_number="LH 999")))

    assert flight_agent.run(REQUEST, [])[0]["options"][0]["arrives_next_day"] is True


def test_an_option_matching_no_candidate_is_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({"id": "F2", "flight_number": "XX 999",
                               "rebooking": "", "notes": ""})
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    result, _ = flight_agent.run(REQUEST, [])

    # Grounding is now structural: an invented flight matches nothing and cannot survive.
    assert [o["id"] for o in result["options"]] == ["F1"]


def test_flight_numbers_match_regardless_of_spacing(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(flight_number="lh687")))

    assert flight_agent.run(REQUEST, [])[0]["options"][0]["flight_number"] == "LH 687"


def test_stops_and_fare_conditions_are_gone(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    option = flight_agent.run(REQUEST, [])[0]["options"][0]

    assert "stops" not in option          # always 0 since direct-only is a decision
    assert "fare_conditions" not in option
    assert option["rebooking"] == "Same airline as the cancelled flight."


# --- caveats --------------------------------------------------------------------


def test_a_thin_result_is_flagged(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    caveats = flight_agent.run(REQUEST, [])[0]["caveats"]

    assert any(c.startswith("NOTE:") and "1 flight" in c for c in caveats)


def test_a_different_carrier_is_flagged(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    # REQUEST's disrupted flight is LH; the candidate is LH, so nothing to say.
    assert not any("different airline" in c
                   for c in flight_agent.run(REQUEST, [])[0]["caveats"])

    other = dict(CANDIDATE, airline_iata="LY", airline="El Al", flight="LY 1")
    monkeypatch.setattr(flights, "search", lambda *a, **k: [other])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(flight_number="LY 1")))

    assert any("different airline" in c
               for c in flight_agent.run(REQUEST, [])[0]["caveats"])


def test_a_departure_the_passenger_may_miss_asks_for_confirmation(monkeypatch):
    # local_now is 22:15; a 23:00 departure leaves 45 minutes.
    soon = dict(CANDIDATE, flight="LH 1",
                depart="2026-08-09T23:00:00+03:00", arrive="2026-08-10T02:00:00+02:00")
    monkeypatch.setattr(flights, "search", lambda *a, **k: [soon])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(flight_number="LH 1")))

    caveats = flight_agent.run(REQUEST, [])[0]["caveats"]

    assert any(c.startswith("CONFIRM:") and "minutes" in c for c in caveats)


def test_the_model_may_add_its_own_caveats(monkeypatch):
    payload = dict(good_payload(), caveats=["NOTE: the model noticed something."])
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    caveats = flight_agent.run(REQUEST, [])[0]["caveats"]

    assert "NOTE: the model noticed something." in caveats
    assert len(caveats) > 1          # code-generated ones come first and are kept


def test_every_caveat_declares_its_intent(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    for caveat in flight_agent.run(REQUEST, [])[0]["caveats"]:
        assert caveat.startswith(("NOTE:", "ASK:", "CONFIRM:")), caveat
```

Also update the two existing tests that reference the old shape:
`test_options_with_unparseable_times_are_dropped` becomes redundant (times no longer come from
the model) — **delete it**, and `test_a_dangling_recommended_id_falls_back_to_the_first_option`
keeps working unchanged.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_flight_agent.py -q`
Expected: FAIL — `KeyError: 'terminal'`, `assert 'stops' not in option`, `KeyError: 'caveats'`.

- [ ] **Step 3: Replace the system prompt**

In `lib/agents/flight_agent.py`, replace `SYSTEM_PROMPT` with:

```python
SYSTEM_PROMPT = """You choose onward flights for a passenger whose flight has just been disrupted.

You are given a list of REAL flights, already verified as departing from the passenger's airport to
their destination and as still catchable. Choose only from that list.

You do not copy the flight's details. Give the flight number exactly as it appears in the candidate
so it can be matched, and everything factual - times, terminal, aircraft, airline - is filled in
from the source data afterwards. Spend your effort on which flights to offer and why.

You do not book, hold or pay for anything - you propose options the passenger acts on themselves.
Include flights on other airlines: the passenger's Contract of Carriage may entitle them to be
rebooked on a competitor, so do not silently drop them. Never name a booking or flight-search website.

Return at most three options, best first. Prefer the earliest arrival that meets any deadline given;
where they differ meaningfully, include one that is gentler on conditions.

You have schedule data ONLY. You do not know fares, seat availability, baggage allowances, or what
the airline owes this passenger. Never state a price, and never state a baggage or compensation
rule - those come from the Contract of Carriage and are handled elsewhere in the plan. Use
"rebooking" to say how hard this flight is likely to be to get moved onto, and to point there.

Only direct flights are available to you. If none suits, say so in "notes" rather than inventing a
connection.

Return a JSON object only, no prose:
{"options": [{"id", "flight_number", "rebooking", "notes"}],
 "recommended_id": "<id>",
 "caveats": [str]}

"caveats" is for the assistant coordinating this plan, not the passenger. Each entry starts with
"NOTE:" for something they should be told, "ASK:" for something only the passenger can answer, or
"CONFIRM:" for something that should not proceed unchecked. Leave it empty if you have nothing to
add; the obvious ones are added automatically.

Example
-------
Candidates:
[{"flight": "LY 357", "airline": "El Al", "airline_iata": "LY", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T06:05+03:00", "arrive": "2026-08-10T09:40+02:00", "status": "Expected", "aircraft": "Boeing 737-900", "terminal": "3"},
 {"flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T16:30+03:00", "arrive": "2026-08-10T20:10+02:00", "status": "Expected", "aircraft": "Airbus A320", "terminal": "3"}]
Request: LH318 TLV->FRA cancelled, 2 adults, must arrive by the evening of 2026-08-10, local time now 2026-08-09T22:15.
Response:
{"options": [{"id": "F1", "flight_number": "LY 357", "rebooking": "A different airline from the one that cancelled. Whether your ticket can be moved across is set by your Contract of Carriage - see the entitlements section.", "notes": "Earliest arrival, and it clears your deadline with the day to spare."}, {"id": "F2", "flight_number": "LH 687", "rebooking": "Same airline as the cancelled flight, which is usually the simplest rebooking to arrange at the desk.", "notes": "Later, but stays with your original carrier and still lands inside your deadline."}],
 "recommended_id": "F1",
 "caveats": ["ASK: both options mean an overnight wait - check they are willing to stay tonight."]}
"""
```

- [ ] **Step 4: Add matching, enrichment and caveats**

In `lib/agents/flight_agent.py`, add `import re` to the imports at the top of the file (it does
not import `re` today), then replace `_validate` with the following and add the helpers above it:

```python
NEAR_DEPARTURE_MINUTES = 90   # below this, the passenger may not make the gate


def _normalise(text) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _match(option: dict, candidates: list[dict]) -> dict | None:
    wanted = _normalise(option.get("flight_number"))
    return next((c for c in candidates if _normalise(c.get("flight")) == wanted), None)


def _enrich(option: dict, candidate: dict) -> dict:
    """Overwrite every factual field from the source. The model chose; it did not measure."""
    depart = datetime.fromisoformat(candidate["depart"])
    arrive = (datetime.fromisoformat(candidate["arrive"])
              if candidate.get("arrive") else None)

    enriched = {
        "id": option.get("id"),
        "airline": candidate.get("airline"),
        "airline_iata": candidate.get("airline_iata"),
        "flight_number": candidate.get("flight"),
        "origin": candidate.get("origin"),
        "destination": candidate.get("destination"),
        "depart": candidate.get("depart"),
        "arrive": candidate.get("arrive"),
        "terminal": candidate.get("terminal"),
        "aircraft": candidate.get("aircraft"),
        "status": candidate.get("status"),
        "rebooking": option.get("rebooking"),
        "notes": option.get("notes"),
    }
    if arrive:
        enriched["duration_minutes"] = round((arrive - depart).total_seconds() / 60)
        # Local dates, because "arrives the next day" is what the passenger experiences.
        enriched["arrives_next_day"] = arrive.date() > depart.date()
    return {key: value for key, value in enriched.items() if value is not None}


def _caveats(request: dict, options: list[dict], candidates: list[dict]) -> list[str]:
    """The ones whose trigger is a fact. The model may add more; it is not relied on for these."""
    said = []

    if len(candidates) < MIN_FOR_CHOICE:
        said.append(f"NOTE: only {len(candidates)} flight(s) were found on this route in the "
                    f"next 48 hours, so there is little to choose between.")

    original = _normalise(request.get("airline"))
    if original and all(_normalise(o.get("airline_iata")) != original for o in options):
        said.append("NOTE: every option is on a different airline from the one that cancelled, "
                    "so the ticket may need endorsing over - the entitlements section covers it.")

    now = datetime.fromisoformat(request["local_now"])
    for option in options:
        departs = datetime.fromisoformat(option["depart"]).replace(tzinfo=None)
        minutes = round((departs - now).total_seconds() / 60)
        if 0 <= minutes <= NEAR_DEPARTURE_MINUTES:
            said.append(f"CONFIRM: {option['flight_number']} leaves in about {minutes} minutes - "
                        f"check the passenger can reach the gate in time.")
            break

    if any(o.get("arrives_next_day") for o in options):
        said.append("NOTE: at least one option arrives the day after it departs.")

    delayed = [o["flight_number"] for o in options
               if str(o.get("status", "")).lower() == "delayed"]
    if delayed:
        said.append(f"NOTE: {', '.join(delayed)} is already marked delayed.")

    return said


def _validate(payload: dict, request: dict, candidates: list[dict]) -> dict:
    """Keep only options that name a real candidate, and fill their facts from it.

    Grounding stops being a rule the model is asked to follow and becomes structural:
    an invented flight matches nothing, so it cannot reach the passenger.
    """
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")

    usable = []
    for option in payload.get("options") or []:
        if not isinstance(option, dict) or not option.get("id"):
            continue
        candidate = _match(option, candidates)
        if candidate:
            usable.append(_enrich(option, candidate))

    if not usable:
        raise ValueError("no option named a flight that was actually offered")

    if payload.get("recommended_id") not in {o["id"] for o in usable}:
        payload["recommended_id"] = usable[0]["id"]

    model_caveats = [str(c) for c in (payload.get("caveats") or []) if str(c).strip()]
    payload["options"] = usable
    payload["caveats"] = _caveats(request, usable, candidates) + model_caveats
    return payload
```

Add near the other constants:

```python
MIN_FOR_CHOICE = 3   # fewer than this and the passenger has no real choice to make
```

- [ ] **Step 5: Pass the request and candidates into validation**

In `run()`, change the validation call:

```python
    try:
        return _validate(payload, request, candidates), [step]
    except ValueError as exc:
        # The call happened, so the trace keeps it even though it was unusable.
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_flight_agent.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add lib/agents/flight_agent.py tests/test_flight_agent.py
git commit -m "FlightAgent: choose and explain, do not transcribe

The model now returns a flight number and prose; code matches that back to
the candidate and writes every factual field from it. An altered time is no
longer unlikely, it is impossible, and an invented flight matches nothing so
it cannot reach the passenger.

Surfaces what the tool already fetched and we were discarding - terminal,
aircraft, status - and computes duration and whether the flight lands the
next day, which changes what the passenger has to plan for.

stops is gone: it was permanently 0 once direct-only became a decision.
fare_conditions is now rebooking, because it never held fare data and the
name misled whoever read the payload next.

caveats are generated in code wherever the trigger is a fact, so the
Supervisor is not relying on the model to notice."
```

---

### Task 2: AccommodationAgent — enrichment from candidates

**Files:**
- Modify: `lib/agents/accommodation_agent.py`
- Test: `tests/test_accommodation_agent.py`

**Interfaces:**
- Consumes: `lib.tools.hotels.search`, `lib.llm.call`.
- Produces: `accommodation_agent.run(request, stay_window, history) -> (payload, steps)` where each
  option carries `id, name, kind, distance_km, area, address, phone, website, stars, wheelchair,
  check_in, check_out, nights, price_estimate, meals, notes`, plus payload-level `recommended_id`
  and `caveats`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_accommodation_agent.py`, replace `CANDIDATE` and `good_payload`, keep the refusal
tests, and add:

```python
CANDIDATE = {"name": "Airport Plaza", "distance_km": 2.4, "area": "Lod", "stars": "4",
             "phone": "+972 3 000 0000", "website": "https://example.test",
             "address": "12 HaNasi", "wheelchair": "yes"}


def good_payload(name="Airport Plaza", recommended="H1"):
    return {
        "options": [{
            "id": "H1", "name": name,
            "price_estimate": "Roughly EUR 110-140 (estimate - not a quoted price)",
            "notes": "Closest to the terminal.",
        }],
        "recommended_id": recommended,
    }


def test_facts_come_from_the_candidate(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    option = accommodation_agent.run(REQUEST, WINDOW, [])[0]["options"][0]

    assert option["distance_km"] == 2.4
    assert option["phone"] == "+972 3 000 0000"
    assert option["website"] == "https://example.test"
    assert option["address"] == "12 HaNasi"
    assert option["wheelchair"] == "yes"
    assert option["city"] == "Lod"
    # Deprecated compatibility field: supervisor._digest reads this and would
    # otherwise lose the distance until Person A migrates (spec P7).
    assert option["area"] == "2.4 km from the terminal, Lod"
    assert option["meals_included"] is False


def test_the_stay_window_is_written_not_asked_for(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    option = accommodation_agent.run(REQUEST, WINDOW, [])[0]["options"][0]

    # The nights come from the flight that was found; the model cannot disagree with them.
    assert option["check_in"] == WINDOW["check_in"]
    assert option["check_out"] == WINDOW["check_out"]
    assert option["nights"] == WINDOW["nights"]


def test_an_invented_property_is_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({"id": "H2", "name": "Imaginary Suites",
                               "price_estimate": "", "notes": ""})
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    result, _ = accommodation_agent.run(REQUEST, WINDOW, [])

    assert [o["id"] for o in result["options"]] == ["H1"]


def test_meals_is_unknown_unless_the_data_says_otherwise(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    option = accommodation_agent.run(REQUEST, WINDOW, [])[0]["options"][0]

    # "meals_included: false" was a lie dressed as data - we almost never know.
    assert option["meals"] == "unknown"
    assert "meals_included" not in option


def test_meals_follows_the_breakfast_tag(monkeypatch):
    for tag, expected in (("yes", "included"), ("no", "not_included")):
        monkeypatch.setattr(hotels, "search",
                            lambda *a, _t=tag, **k: [dict(CANDIDATE, breakfast=_t)])
        monkeypatch.setattr(llm, "call", fake_call(good_payload()))

        option = accommodation_agent.run(REQUEST, WINDOW, [])[0]["options"][0]
        assert option["meals"] == expected


def test_kind_appears_only_when_it_is_not_a_hotel(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))
    assert "kind" not in accommodation_agent.run(REQUEST, WINDOW, [])[0]["options"][0]

    monkeypatch.setattr(hotels, "search", lambda *a, **k: [dict(CANDIDATE, kind="hostel")])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))
    option = accommodation_agent.run(REQUEST, WINDOW, [])[0]["options"][0]
    assert option["kind"] == "hostel"


# --- caveats --------------------------------------------------------------------


def test_no_phone_anywhere_is_an_ask(monkeypatch):
    monkeypatch.setattr(hotels, "search",
                        lambda *a, **k: [{"name": "Airport Plaza", "distance_km": 2.4,
                                          "area": "Lod"}])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    caveats = accommodation_agent.run(REQUEST, WINDOW, [])[0]["caveats"]

    assert any(c.startswith("ASK:") and "phone" in c for c in caveats)


def test_a_distant_option_asks_for_confirmation(monkeypatch):
    far = dict(CANDIDATE, distance_km=17.0)
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [far])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    caveats = accommodation_agent.run(REQUEST, WINDOW, [])[0]["caveats"]

    assert any(c.startswith("CONFIRM:") and "17" in c for c in caveats)


def test_prices_are_always_flagged_as_estimates(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    caveats = accommodation_agent.run(REQUEST, WINDOW, [])[0]["caveats"]

    assert any("estimate" in c.lower() for c in caveats)


def test_every_caveat_declares_its_intent(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    for caveat in accommodation_agent.run(REQUEST, WINDOW, [])[0]["caveats"]:
        assert caveat.startswith(("NOTE:", "ASK:", "CONFIRM:")), caveat
```

Delete `test_options_on_the_wrong_dates_are_dropped` and `test_unparseable_dates_are_dropped`:
the dates no longer come from the model, so neither failure is expressible.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_accommodation_agent.py -q`
Expected: FAIL — `KeyError: 'distance_km'`, `KeyError: 'caveats'`.

- [ ] **Step 3: Replace the system prompt**

In `lib/agents/accommodation_agent.py`, replace `SYSTEM_PROMPT` with:

```python
SYSTEM_PROMPT = """You find somewhere for a stranded air passenger to sleep, for a fixed set of nights.

You are given a list of REAL places near the airport. Choose only from that list. Give each name
exactly as it appears so it can be matched; everything factual - distance, address, phone, website,
accessibility, and the nights themselves - is filled in from the source data afterwards. Spend your
effort on which places to offer, in what order, and why.

The nights are not yours to choose. They come from the replacement flight the passenger is taking.

You do not book, hold or pay for anything. Never name a booking website.

Prioritise being close enough to reach the airport for the departure, then whatever the data says
about the place.

You do NOT know prices, availability, or whether meals are included. "price_estimate" must read as
an estimate and never as a quote. If a candidate has a phone number, say in "notes" that the
passenger should call to confirm the room and the rate - that is the one way to settle the two
things you cannot. Say plainly when somewhere is a hostel, guest house or apartment rather than a
hotel, because it changes what to expect. Never assert a fact about a real named business that you
were not given.

Return a JSON object only, no prose:
{"options": [{"id", "name", "price_estimate", "notes"}],
 "recommended_id": "<id>",
 "caveats": [str]}

"caveats" is for the assistant coordinating this plan, not the passenger. Each entry starts with
"NOTE:", "ASK:" or "CONFIRM:". Leave it empty if you have nothing to add; the obvious ones are
added automatically.

Example
-------
Hotels: [{"name": "Airport Plaza", "distance_km": 2.4, "area": "Lod", "stars": "4", "phone": "+972 3 000 0000", "address": "12 HaNasi", "wheelchair": "yes"},
         {"name": "City Central Inn", "distance_km": 11.2, "area": "Tel Aviv", "kind": "hostel"}]
Request: 2 guests stranded at TLV, check in 2026-08-09, check out 2026-08-10, 1 night, onward flight departs 2026-08-10T09:40.
Response:
{"options": [{"id": "H1", "name": "Airport Plaza", "price_estimate": "Roughly EUR 110-140 for the night (estimate - not a quoted price)", "notes": "Closest to the terminal, which matters for an 09:40 departure. Call them to confirm a room and the rate - I cannot check either."}, {"id": "H2", "name": "City Central Inn", "price_estimate": "Roughly EUR 40-70 for the night (estimate - not a quoted price)", "notes": "A hostel rather than a hotel, so expect shared facilities. Cheaper, but 11 km out - leave a clear hour to get back for the flight."}],
 "recommended_id": "H1",
 "caveats": []}
"""
```

- [ ] **Step 4: Add matching, enrichment and caveats**

Add `import re` to the imports at the top of `lib/agents/accommodation_agent.py` (it does not
import `re` today), then replace `_validate`, adding these above it:

```python
FAR_FROM_TERMINAL_KM = 15.0
MEALS_FROM_TAG = {"yes": "included", "no": "not_included"}


def _normalise(text) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _match(option: dict, candidates: list[dict]) -> dict | None:
    wanted = _normalise(option.get("name"))
    return next((c for c in candidates if _normalise(c.get("name")) == wanted), None)


def _area_text(candidate: dict) -> str | None:
    """The human phrasing supervisor._digest still reads. Deprecated with `area`."""
    distance, city = candidate.get("distance_km"), candidate.get("area")
    if distance is None:
        return city
    return f"{distance} km from the terminal" + (f", {city}" if city else "")


def _enrich(option: dict, candidate: dict, stay_window: dict) -> dict:
    """Facts from OpenStreetMap, nights from the Supervisor, prose from the model."""
    meals = MEALS_FROM_TAG.get(str(candidate.get("breakfast", "")).lower(), "unknown")
    enriched = {
        "id": option.get("id"),
        "name": candidate.get("name"),
        "kind": candidate.get("kind"),          # absent when it is an ordinary hotel
        "distance_km": candidate.get("distance_km"),
        "city": candidate.get("area"),
        # Deprecated (spec P7): supervisor._digest reads `area` and would otherwise
        # lose the distance entirely. Delete once it reads distance_km and city.
        "area": _area_text(candidate),
        "address": candidate.get("address"),
        "phone": candidate.get("phone"),
        "website": candidate.get("website"),
        "stars": candidate.get("stars"),
        "wheelchair": candidate.get("wheelchair"),
        "check_in": stay_window.get("check_in"),
        "check_out": stay_window.get("check_out"),
        "nights": stay_window.get("nights"),
        "price_estimate": option.get("price_estimate"),
        "meals": meals,
        # Deprecated (spec P7): absent reads as falsy in _digest, which would assert
        # "no meals" where the truth is "unknown". Delete with `area`.
        "meals_included": meals == "included",
        "notes": option.get("notes"),
    }
    return {key: value for key, value in enriched.items() if value is not None}


def _caveats(options: list[dict], candidates: list[dict]) -> list[str]:
    said = ["NOTE: every price here is an estimate, not a quote - nothing was checked "
            "against the property."]

    if not any(o.get("phone") for o in options):
        said.append("ASK: none of these has a phone number listed, so nobody can confirm a room "
                    "tonight - the passenger may prefer to ask the airline desk instead.")

    nearest = min((o["distance_km"] for o in options if o.get("distance_km") is not None),
                  default=None)
    if nearest is not None and nearest > FAR_FROM_TERMINAL_KM:
        said.append(f"CONFIRM: the nearest option is {nearest} km from the terminal - check the "
                    f"passenger can still make the departure.")

    if all(o.get("kind") for o in options):
        said.append("NOTE: no ordinary hotels were found near this airport - these are hostels, "
                    "guest houses or apartments.")

    return said


def _validate(payload: dict, stay_window: dict, candidates: list[dict]) -> dict:
    """Keep only options naming a real property, and fill their facts from it."""
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")

    usable = []
    for option in payload.get("options") or []:
        if not isinstance(option, dict) or not option.get("id"):
            continue
        candidate = _match(option, candidates)
        if candidate:
            usable.append(_enrich(option, candidate, stay_window))

    if not usable:
        raise ValueError("no option named a place that was actually offered")

    if payload.get("recommended_id") not in {o["id"] for o in usable}:
        payload["recommended_id"] = usable[0]["id"]

    model_caveats = [str(c) for c in (payload.get("caveats") or []) if str(c).strip()]
    payload["options"] = usable
    payload["caveats"] = _caveats(usable, candidates) + model_caveats
    return payload
```

Then in `run()`:

```python
    try:
        return _validate(payload, stay_window, candidates), [step]
    except ValueError as exc:
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_accommodation_agent.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lib/agents/accommodation_agent.py tests/test_accommodation_agent.py
git commit -m "AccommodationAgent: surface what OpenStreetMap already told us

Phone, website, address, accessibility and an exact numeric distance were
all in the response we were already paying for, and all discarded. A phone
number is the most useful field either agent has: it hands the passenger the
one job this agent cannot do.

area was a free-text blob mixing distance with place, so nothing could sort
or compare it; it is now a number and a location. meals_included: false was
a lie dressed as data, since we almost never know - meals is now included,
not_included or unknown, and unknown is the honest answer nearly every time.

The nights are written from the Supervisor's stay window rather than asked
of the model, so they cannot disagree with the flight that produced them."
```

---

### Task 3: Evaluator, shared fakes, and a new grounding check

**Files:**
- Modify: `evals/search_artifact.py`, `evals/search_cases.py`, `evals/search_agent_cases.json`, `tests/conftest.py`, `tests/test_search_artifact.py`

**Interfaces:**
- Consumes: the payloads from Tasks 1 and 2.
- Produces: check name `facts_from_candidates`, registered in `evals.search_cases.CHECK_NAMES`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_search_artifact.py`, update the fixtures to the new shape and add the new check's
tests:

```python
GOOD_FLIGHT = {"id": "F1", "airline": "Lufthansa", "airline_iata": "LH",
               "flight_number": "LH 687", "origin": "TLV", "destination": "FRA",
               "depart": "2026-08-11T16:30+03:00", "arrive": "2026-08-11T20:10+02:00",
               "terminal": "3", "aircraft": "Airbus A320", "status": "Expected",
               "rebooking": "See your Contract of Carriage.", "notes": "Nonstop."}
GOOD_HOTEL = {"id": "H1", "name": "Airport Plaza", "distance_km": 2.4, "area": "Lod",
              "check_in": "2026-08-10", "check_out": "2026-08-11", "nights": 1,
              "price_estimate": "Roughly EUR 120 (estimate)", "meals": "unknown",
              "notes": "Meals not confirmed - check at the desk."}


def test_facts_from_candidates_passes_when_they_agree():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]),
                       case_with("facts_from_candidates"))
    assert status_of(results, "facts_from_candidates") == "pass"


def test_facts_from_candidates_catches_a_drifted_time():
    drifted = dict(GOOD_FLIGHT, depart="2026-08-11T09:00+03:00")
    results = evaluate(artifact_with([drifted], [GOOD_HOTEL]),
                       case_with("facts_from_candidates"))
    assert status_of(results, "facts_from_candidates") == "fail"


def test_meals_honesty_accepts_unknown():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]),
                       case_with("meals_honesty"))
    assert status_of(results, "meals_honesty") == "pass"


def test_meals_honesty_rejects_included_without_a_tag():
    claimed = dict(GOOD_HOTEL, meals="included")
    results = evaluate(artifact_with([GOOD_FLIGHT], [claimed]), case_with("meals_honesty"))
    assert status_of(results, "meals_honesty") == "fail"


def test_no_asserted_fare_reads_the_rebooking_field():
    quoting = dict(GOOD_FLIGHT, rebooking="Rebooking costs EUR 90.")
    results = evaluate(artifact_with([quoting], [GOOD_HOTEL]), case_with("no_asserted_fare"))
    assert status_of(results, "no_asserted_fare") == "fail"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_search_artifact.py -q`
Expected: FAIL — `KeyError: 'facts_from_candidates'`, and `meals_honesty` still reads
`meals_included`.

- [ ] **Step 3: Update the evaluator**

In `evals/search_artifact.py`:

```python
FACTS_FROM_CANDIDATE = ("depart", "arrive", "origin", "destination", "terminal",
                        "aircraft", "status", "airline")
CANDIDATE_KEY = {"airline": "airline", "depart": "depart", "arrive": "arrive",
                 "origin": "origin", "destination": "destination",
                 "terminal": "terminal", "aircraft": "aircraft", "status": "status"}


def _check_facts_from_candidates(artifact, case):
    """Every factual field must equal the source. The model no longer supplies these.

    Guards the enrichment: if a rename or refactor ever lets model output through
    again, a drifted departure time shows up here rather than in a booked hotel.
    """
    step = _step(artifact, "FlightAgent")
    if not step:
        return _result("facts_from_candidates", True, "FlightAgent was not dispatched", skip=True)
    candidates = {_normalise(c.get("flight")): c
                  for c in candidates_from_prompt(step["prompt"]["user_prompt"])}
    if not candidates:
        return _result("facts_from_candidates", True, "no candidates to compare", skip=True)

    drifted = []
    for option in _options(step):
        candidate = candidates.get(_normalise(option.get("flight_number")))
        if not candidate:
            continue
        for field in FACTS_FROM_CANDIDATE:
            if field in option and option[field] != candidate.get(CANDIDATE_KEY[field]):
                drifted.append(f"{option.get('id')}.{field}")
    return _result("facts_from_candidates", not drifted,
                   f"drifted from source: {drifted}" if drifted
                   else "every fact matches its candidate")
```

Change `_check_no_asserted_fare` to read `rebooking` instead of `fare_conditions`:

```python
    quoting = [f"{o.get('id')}: {o.get('rebooking')}" for o in _options(step)
               if MONEY.search(str(o.get("rebooking", "")) + str(o.get("notes", "")))]
```

Change `_check_meals_honesty`'s per-option branch:

```python
        meals = str(option.get("meals", "unknown")).lower()
        if meals == "included":
            if not (candidate and candidate.get("breakfast")):
                problems.append(f"{option.get('id')} claims meals with no supporting tag")
        elif meals not in ("not_included", "unknown"):
            problems.append(f"{option.get('id')} has meals={meals!r}")
```

Register it:

```python
    "facts_from_candidates": _check_facts_from_candidates,
```

- [ ] **Step 4: Register the check name and apply it**

In `evals/search_cases.py` add `"facts_from_candidates",` to `CHECK_NAMES`.

In `evals/search_agent_cases.json`, add it beside `grounding_flights` everywhere that appears:

```bash
python - <<'PY'
import pathlib
p = pathlib.Path("evals/search_agent_cases.json"); t = p.read_text(encoding="utf-8")
t = t.replace('"grounding_flights", "no_unusable_status"',
              '"grounding_flights", "facts_from_candidates", "no_unusable_status"')
p.write_text(t, encoding="utf-8")
print("applied facts_from_candidates")
PY
```

- [ ] **Step 5: Update the shared fakes**

In `tests/conftest.py` — **shared with Person A's Supervisor tests** — the fakes now return the
model's half only, since the agents fill the rest:

```python
def _flight_response(user_prompt: str) -> dict:
    return {
        "options": [{
            "id": "F1", "flight_number": "LH 687",
            "rebooking": "Rebooking terms come from your Contract of Carriage.",
            "notes": "Earliest nonstop.",
        }],
        "recommended_id": "F1",
        "caveats": [],
    }


def _accommodation_response(user_prompt: str) -> dict:
    return {
        "options": [{
            "id": "H1", "name": "Airport Plaza",
            "price_estimate": "EUR 120 total (estimate)",
            "notes": "Meals not confirmed — check at the desk.",
        }],
        "recommended_id": "H1",
        "caveats": [],
    }
```

The flight number and hotel name must match `fake_search_data`'s candidates — they already do,
which is why the fakes get shorter rather than longer.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass. If `tests/test_supervisor.py` fails on `meals_included` or `stops`, that is
Person A's `_digest` reading a field that has moved — **stop and report it**; do not edit
`supervisor.py` in this branch.

- [ ] **Step 7: Commit**

```bash
git add evals/ tests/conftest.py tests/test_search_artifact.py
git commit -m "Check that facts really do come from the candidates

Adds facts_from_candidates: every factual field on a returned option must
equal its source. It guards the enrichment - if a refactor ever lets model
output through again, a drifted departure time shows up here rather than in
a hotel booked for the wrong night.

Moves the two checks that read renamed fields, and shortens the shared fakes
in tests/conftest.py, which now only need to supply the model's half."
```

---

### Task 4: The contract Person A implements against

**Files:**
- Modify: `docs/PROJECT_PLAN.md`, `docs/search-agents-capabilities.md`

- [ ] **Step 1: Update the locked payload contract**

In `docs/PROJECT_PLAN.md` §1, replace the two search-agent payload blocks with:

```python
# FlightAgent   (facts written from the candidate, not the model)
{"options": [{"id", "airline", "airline_iata", "flight_number", "origin", "destination",
              "depart", "arrive", "duration_minutes", "arrives_next_day",
              "terminal", "aircraft", "status", "rebooking", "notes"}],
 "recommended_id": "F1",
 "caveats": [str]}

# AccommodationAgent
{"options": [{"id", "name", "kind", "distance_km", "area", "address", "phone", "website",
              "stars", "wheelchair", "check_in", "check_out", "nights",
              "price_estimate", "meals", "notes"}],
 "recommended_id": "H1",
 "caveats": [str]}
```

Immediately after the payload block, add:

```markdown
**Changed 15/8/2026** (agreed at the team meeting; spec:
`docs/superpowers/specs/2026-08-15-search-agent-payload-refinement-design.md`). For Person A,
consuming these in `supervisor._digest`:

- **`caveats`** is new on both, `list[str]`, matching `DocumentationAgent`. Every entry opens
  `NOTE:` (tell the passenger), `ASK:` (only the passenger can answer) or `CONFIRM:` (do not
  proceed unchecked). The ones that matter are generated in code, not by the model.
- **Renamed:** `fare_conditions` → `rebooking` (it never held fare data).
- **Removed:** `stops` — permanently `0` once direct-only became a decision (D7).
- **Split:** `area` is now a real location; `distance_km` is a number you can sort on.
- **Widened:** `meals_included: bool` → `meals: "included" | "not_included" | "unknown"`.
  `unknown` is the honest answer nearly every time and `false` was a lie dressed as data.
- **Optional fields are omitted, not null.** Test with `.get()`; absence means OpenStreetMap or
  AeroDataBox did not know, which is worth saying rather than hiding.
- Every option is now worth forwarding: the digest currently keeps only `recommended_id`, which
  is why "compare the two options" cannot work today.
```

- [ ] **Step 2: Record it in the capabilities report**

Append to `docs/search-agents-capabilities.md` §2:

```markdown
### 2c. The payload was written before we knew how the agents behave

Agreed at the 15/8 meeting and specced separately. Two changes matter beyond the field list:

**The model stopped being asked to transcribe.** It returns a choice and prose; code matches that
back to the source candidate and writes every factual field. An altered departure time went from
unlikely to impossible, and an invented option now matches nothing and cannot survive validation.

**Caveats are generated in code wherever the trigger is a fact** — "only one flight in 48 hours",
"leaves in 40 minutes", "no phone number anywhere". The `status` finding above is why: a model that
notices something in one run of two is not a safety net.
```

- [ ] **Step 3: Sweep for stale statements**

Run: `grep -rn "fare_conditions\|meals_included\|\"stops\"" --include=*.py --include=*.md . | grep -v ".venv\|live-test-output\|specs/2026-08-15"`
Expected: only `lib/agents/supervisor.py` (Person A's, deliberately untouched) and historical
lines inside older spec files. Fix anything else.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "Publish the new payload contract for Person A

PROJECT_PLAN §1 is a locked decision and the place she implements against,
so the change lands there with the consumer notes beside it: what is new,
what was renamed, what was removed, and that optional fields are omitted
rather than null.

Flags the thing worth acting on beyond the field list: the digest keeps only
the recommended option, which is why 'compare the two options' cannot work
today no matter how good the agents get."
```

---

### Task 5: Prove it live

**Files:** none — verification only.

- [ ] **Step 1: Re-run the scenario matrix**

```bash
python scripts/run_search_agents_live.py --confirm-paid-calls --all \
  --output live-test-output/refined.json
python scripts/evaluate_search_artifact.py --artifact live-test-output/refined.json
```
Expected: 16 scenarios, 0 failing checks, ~15–20 LLM calls, ~20 API units.

- [ ] **Step 2: Read one payload with your own eyes**

```bash
python -c "
import json
b = json.load(open('live-test-output/refined.json', encoding='utf-8'))
r = next(x for x in b['runs'] if x['scenario'] == 'tlv-fra-cancelled')
for module in ('FlightAgent', 'AccommodationAgent'):
    step = next((s for s in r['steps'] if s['module'] == module), None)
    if step:
        print(f'--- {module}')
        print(json.dumps(step['response'], indent=2, ensure_ascii=False)[:1400])
"
```
Check by hand: facts match the candidates, `caveats` all open `NOTE:`/`ASK:`/`CONFIRM:`, no
`stops`, no `fare_conditions`, `meals` reads `unknown` rather than `false`.

- [ ] **Step 3: Record the cost**

Note LLM calls, tokens and units consumed, and add them to the capabilities report's cost table.

- [ ] **Step 4: Commit**

```bash
git add docs/search-agents-capabilities.md
git commit -m "Verify the refined payloads against a real model"
```

---

## Verification before handing over

- [ ] `python -m pytest -q` green with no credentials configured.
- [ ] 16 scenarios, 0 failing checks, including `facts_from_candidates`.
- [ ] `grep` sweep shows no stale `fare_conditions` / `meals_included` / `stops` outside
      `supervisor.py` and historical specs.
- [ ] Person A told, in words rather than a commit message, that `tests/conftest.py` changed
      and that `_digest` must move before this reaches `main`.
