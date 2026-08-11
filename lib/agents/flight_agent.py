"""FlightAgent — onward options after a disruption, grounded in real schedules.

Pattern: role prompt + one-shot example (`docs/PROJECT_PLAN.md` §1). The example
now shows the model *choosing from* verified candidates rather than inventing
options, because `lib/tools/flights.py` supplies real departures.

Scope is deliberately narrow (design doc D5): schedule data cannot answer what
a ticket allows, so baggage, fare and compensation questions are deferred to
DocumentationAgent, which can cite the Contract of Carriage clause.
"""

import json
from datetime import datetime

from lib import llm
from lib.tools import flights

MODULE = "FlightAgent"

SYSTEM_PROMPT = """You choose onward flights for a passenger whose flight has just been disrupted.

You are given a list of REAL flights, already verified as departing from the passenger's airport to
their destination. Choose only from that list. Never invent a flight, and never alter a time, flight
number or airline.

You do not book, hold or pay for anything — you propose options the passenger acts on themselves.
Include flights on other airlines: the passenger's Contract of Carriage may entitle them to be
rebooked on a competitor, so do not silently drop them. Never name a booking or flight-search website.

Return at most three options. Prefer the earliest arrival that meets any deadline given; where they
differ meaningfully, include one that is gentler on conditions.

You have schedule data ONLY. You do not know fares, seat availability, baggage allowances, or what
the airline owes this passenger. Never state a price as fact, and never state a baggage or
compensation rule — those come from the passenger's Contract of Carriage and are handled elsewhere
in the plan. Use "fare_conditions" to point there, not to assert terms.

Only direct flights are available to you. If none suits, say so in "notes" rather than inventing a
connection.

Return a JSON object only, no prose:
{"options": [{"id", "airline", "flight_number", "origin", "destination", "depart", "arrive",
              "stops", "fare_conditions", "notes"}],
 "recommended_id": "<id>"}

"depart" and "arrive" must be copied exactly from the candidate you chose, in ISO 8601 local time.
They decide which nights the passenger is stranded, so an altered time books the wrong hotel.

Example
-------
Candidates:
[{"flight": "LY 357", "airline": "El Al", "airline_iata": "LY", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T06:05+03:00", "arrive": "2026-08-10T09:40+02:00", "status": "Expected", "aircraft": "Boeing 737-900", "terminal": "3"},
 {"flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T16:30+03:00", "arrive": "2026-08-10T20:10+02:00", "status": "Expected", "aircraft": "Airbus A320", "terminal": "3"}]
Request: LH318 TLV->FRA cancelled, 2 adults, must arrive by the evening of 2026-08-10, local time now 2026-08-09T22:15.
Response:
{"options": [{"id": "F1", "airline": "El Al", "flight_number": "LY 357", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T06:05+03:00", "arrive": "2026-08-10T09:40+02:00", "stops": 0, "fare_conditions": "A different airline from the one that cancelled. Whether your ticket can be moved across is set by your Contract of Carriage - see the entitlements section.", "notes": "Earliest arrival, and it clears your deadline with the day to spare."}, {"id": "F2", "airline": "Lufthansa", "flight_number": "LH 687", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T16:30+03:00", "arrive": "2026-08-10T20:10+02:00", "stops": 0, "fare_conditions": "Same airline as the cancelled flight, which is usually the simplest rebooking to arrange at the desk.", "notes": "Later, but stays with your original carrier and still lands inside your deadline."}],
 "recommended_id": "F1"}
"""

DEGRADED_NOTE = (
    "No live schedule data was available for this route. Propose plausible options from your own "
    "knowledge, and put 'Illustrative option - not live availability.' in the notes of every one."
)


def _user_prompt(request: dict, history: list[dict], candidates: list[dict]) -> str:
    lines = [
        f"Flight: {request.get('airline')} {request.get('flight_number')}",
        f"Route: {request.get('origin')} -> {request.get('destination')}",
        f"What happened: {request.get('disruption')}",
        f"Passenger is at: {request.get('stranded_at')}",
        f"Party size: {request.get('party_size')}",
        f"Must arrive by: {request.get('arrive_by') or 'as soon as possible'}",
        f"Local time now: {request.get('local_now')}",
        "",
    ]
    if candidates:
        lines.append("Candidates (real, verified departures - choose only from these):")
        lines.append(json.dumps(candidates, ensure_ascii=False))
    else:
        lines.append(DEGRADED_NOTE)

    if history:
        lines.append("")
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"  passenger: {turn['prompt']}")
            lines.append(f"  you: {turn['response']}")
    return "\n".join(lines)


def _validate(payload: dict) -> dict:
    """Drop anything the Supervisor cannot use.

    `supervisor._stay_window` calls `datetime.fromisoformat` on "depart", so an
    unparseable time costs the passenger the hotel as well as the flight.
    """
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")

    usable = []
    for option in payload.get("options") or []:
        if not isinstance(option, dict) or not option.get("id"):
            continue
        try:
            datetime.fromisoformat(option.get("depart") or "")
            datetime.fromisoformat(option.get("arrive") or "")
        except (TypeError, ValueError):
            continue
        usable.append(option)

    if not usable:
        raise ValueError("no option came back with a usable departure time")

    payload["options"] = usable
    if payload.get("recommended_id") not in {o["id"] for o in usable}:
        payload["recommended_id"] = usable[0]["id"]
    return payload


def run(request: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). See `docs/PROJECT_PLAN.md` §1 for both shapes."""
    after = datetime.fromisoformat(request["local_now"])
    candidates = flights.search(request.get("origin"), request.get("destination"), after)

    payload, step = llm.call(
        MODULE, SYSTEM_PROMPT, _user_prompt(request, history, candidates), expect_json=True
    )
    try:
        return _validate(payload), [step]
    except ValueError as exc:
        # The call happened, so the trace keeps it even though it was unusable.
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
