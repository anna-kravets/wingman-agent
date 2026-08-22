"""FlightAgent — onward options after a disruption, grounded in real schedules.

The role prompt and one-shot example teach the model to choose from verified
candidates rather than inventing options, because `lib/tools/flights.py` supplies
real departures.

Scope is deliberately narrow: schedule data cannot answer what a ticket allows,
so baggage, fare and compensation questions are deferred to
DocumentationAgent, which can cite the Contract of Carriage clause.
"""

import json
import re
from datetime import datetime

from lib import llm
from lib.tools import flights

MODULE = "FlightAgent"

MIN_FOR_CHOICE = 3            # fewer than this and the passenger has no real choice
NEAR_DEPARTURE_MINUTES = 90   # below this, they may not make the gate
MAX_COMPLETION_TOKENS = 10_000

SYSTEM_PROMPT = """You choose onward flights for a passenger whose flight has just been disrupted.

You are given a list of REAL scheduled flights, verified as departing from the passenger's airport
to their destination after the current time. Choose only from that list. A very near departure is
not necessarily catchable: the passenger still needs the airline to confirm seats and complete the
rebooking, so prefer a realistically actionable option over one that is about to close.

You do not copy the flight's details. Give the flight number exactly as it appears in the candidate
so it can be matched, and everything factual - times, terminal, aircraft, airline - is filled in
from the source data afterwards. Spend your effort on which flights to offer and why.

You do not book, hold or pay for anything - you propose options the passenger acts on themselves.
Include flights on other airlines: the passenger's Contract of Carriage may entitle them to be
rebooked on a competitor, so do not silently drop them. Never name a booking or flight-search website.

Return at most three options, best first. Prefer the earliest arrival that meets any deadline given;
where they differ meaningfully, include one that is gentler on conditions.

Do not ask whether everyone in the stated party still intends to travel together unless the
passenger has actually suggested splitting the party. Never tell the passenger to "proceed with"
an internal option id such as F1; describe the real flight instead.

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

NO_LIVE_DATA = (
    "live flight schedules were not available for this route, so no departure could be "
    "verified. Nothing was invented - check your airline's app or the departures board for "
    "the next flight."
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
    lines.append("Candidates (real, verified departures - choose only from these):")
    lines.append(json.dumps(candidates, ensure_ascii=False))

    if history:
        lines.append("")
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"  passenger: {turn['prompt']}")
            lines.append(f"  you: {turn['response']}")
    return "\n".join(lines)


def _normalise(text) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _original_iata(request: dict) -> str:
    """Best available carrier code for the disrupted flight.

    Intake usually returns a human airline name ("Lufthansa"), while schedule data
    carries IATA ("LH"). The flight number is the reliable bridge between them.
    """
    flight_number = _normalise(request.get("flight_number"))
    match = re.match(r"^([A-Z0-9]{2})(?=\d)", flight_number)
    if match:
        return match.group(1)
    airline = _normalise(request.get("airline"))
    return airline if len(airline) in (2, 3) else ""


def _minutes_until(request: dict, option: dict) -> int:
    """Minutes from the passenger's local clock to an option's departure."""
    now = datetime.fromisoformat(request["local_now"])
    departs = datetime.fromisoformat(option["depart"])
    # local_now deliberately has no offset; AeroDataBox's departure does. Both
    # describe the origin airport's local clock, so compare their wall times.
    if departs.tzinfo is not None and now.tzinfo is None:
        departs = departs.replace(tzinfo=None)
    elif departs.tzinfo is None and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    return round((departs - now).total_seconds() / 60)


def _match(option: dict, candidates: list[dict]) -> dict | None:
    wanted = _normalise(option.get("flight_number"))
    return next((c for c in candidates if _normalise(c.get("flight")) == wanted), None)


def _enrich(option: dict, candidate: dict) -> dict:
    """Overwrite every factual field from the source. The model chose; it did not measure."""
    depart = datetime.fromisoformat(candidate["depart"])
    arrive = datetime.fromisoformat(candidate["arrive"]) if candidate.get("arrive") else None

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
    """The ones whose trigger is a fact.

    Generated here rather than asked of the model: live validation caught it
    flagging a departed flight in one run and ignoring it in the next, from
    identical data. The Supervisor should not depend on it noticing.
    """
    said = []

    if len(candidates) < MIN_FOR_CHOICE:
        said.append(f"NOTE: only {len(candidates)} flight(s) were found on this route in the "
                    f"next 48 hours, so there is little to choose between.")

    original = _original_iata(request)
    if original and all(_normalise(o.get("airline_iata")) != original for o in options):
        said.append("NOTE: every option is on a different airline from the one that cancelled, "
                    "so the ticket may need endorsing over - your Contract of Carriage covers "
                    "whether that's automatic.")

    for option in options:
        minutes = _minutes_until(request, option)
        if 0 <= minutes <= NEAR_DEPARTURE_MINUTES:
            said.append(
                f"CONFIRM: {option['flight_number']} leaves in about {minutes} minutes. "
                "Treat it as an urgent possibility, not a reliable replacement, until the "
                "airline confirms seats, completes the rebooking, and says you can still board."
            )
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

    # The model naturally gravitates to the earliest timestamp. That is unsafe when
    # the flight leaves before a disrupted passenger can reasonably be rebooked and
    # reach the gate. Keep the urgent flight visible, but base the recovery plan (and
    # therefore the hotel nights) on the first option outside the risk window.
    recommended = next(o for o in usable if o["id"] == payload["recommended_id"])
    if _minutes_until(request, recommended) <= NEAR_DEPARTURE_MINUTES:
        actionable = [o for o in usable
                      if _minutes_until(request, o) > NEAR_DEPARTURE_MINUTES]
        if actionable:
            actionable.sort(key=lambda o: o["depart"])
            payload["recommended_id"] = actionable[0]["id"]

    model_caveats = [str(c) for c in (payload.get("caveats") or []) if str(c).strip()]
    # These two recurrent model inventions add no passenger information and can be
    # actively harmful. Party size does not imply a plan to split, and internal ids
    # are meaningless outside the trace.
    model_caveats = [c for c in model_caveats
                     if not ("travel together" in c.lower() and "confirm" in c.lower())
                     and not re.search(r"\bproceed with f\d+\b", c, re.IGNORECASE)]
    payload["options"] = usable
    payload["caveats"] = _caveats(request, usable, candidates) + model_caveats
    return payload


def run(request: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Return the selected flight payload and its ordered LLM trace."""
    origin, destination = request.get("origin"), request.get("destination")

    # Decided offline and free: a route that cannot exist should not cost 8 units
    # discovering that, and "live schedules were unavailable" is the wrong thing
    # to tell someone who asked for TLV to TLV.
    problem = flights.route_problem(origin, destination)
    if problem:
        raise llm.LLMError(f"{MODULE}: {problem}", steps=[], passenger_message=problem)

    after = datetime.fromisoformat(request["local_now"])
    candidates = flights.search(origin, destination, after)

    if not candidates:
        # No LLM call at all. With nothing verified to choose from, the only honest
        # answer is that we do not know: a fabricated flight number for a real airline
        # is actionable, and a stressed passenger may go and ask for it at the desk.
        # Live validation on 10/8/2026 found the model refuses here anyway, which is
        # why the earlier "degrade to labelled illustrative options" decision was
        # reversed. `steps` is empty because no call was made.
        raise llm.LLMError(f"{MODULE}: {NO_LIVE_DATA}", steps=[], passenger_message=NO_LIVE_DATA)

    payload, step = llm.call(
        MODULE, SYSTEM_PROMPT, _user_prompt(request, history, candidates), expect_json=True,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    try:
        return _validate(payload, request, candidates), [step]
    except ValueError as exc:
        # The call happened, so the trace keeps it even though it was unusable.
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
