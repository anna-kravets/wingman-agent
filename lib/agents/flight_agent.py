"""FlightAgent — onward options after a disruption.

Pattern: role prompt + one-shot example (`docs/PROJECT_PLAN.md` §1).

STUB — see `IS_STUB`. Returns canned options without calling the LLM so the
Supervisor and the full `/api/execute` trace can be exercised before the LLMod key
exists. Person B replaces the body of `run()`; the prompts, signature and payload
shape are already what the real version should use, so nothing around it changes.
"""

from datetime import datetime, timedelta

from lib.steps import make_step

IS_STUB = True

MODULE = "FlightAgent"

SYSTEM_PROMPT = """You find onward flights for a passenger whose flight has just been disrupted.

You do not book, hold, or pay for anything. You propose options the passenger acts on themselves.
Search across airlines, not only the one that failed them — the passenger's Contract of Carriage may
entitle them to be rebooked on a competitor, so do not silently rule it out.

Prefer the earliest arrival that still gets them there in time, but return a range: a fastest
option, a cheapest option, and one with the most forgiving conditions. Never name a specific
booking or flight-search website.

Return a JSON object only, no prose:
{"options": [{"id", "airline", "flight_number", "origin", "destination", "depart", "arrive",
              "stops", "fare_conditions", "notes"}],
 "recommended_id": "<id>"}

"depart" and "arrive" are ISO 8601 local times — they are used to work out which nights the
passenger is stranded, so they must be exact.

Example
-------
Request: LH318 TLV->FRA cancelled, passenger stuck at TLV, 2 adults, needs to arrive by 2026-08-09 evening, local time now 2026-08-08T22:15.
Response:
{"options": [
  {"id": "F1", "airline": "LH", "flight_number": "LH687", "origin": "TLV", "destination": "FRA",
   "depart": "2026-08-09T09:40:00", "arrive": "2026-08-09T13:05:00", "stops": 0,
   "fare_conditions": "Rebooking on the original ticket, no fare difference for a cancellation.",
   "notes": "Earliest nonstop. Two seats confirmed available at time of search."},
  {"id": "F2", "airline": "OS", "flight_number": "OS858", "origin": "TLV", "destination": "FRA",
   "depart": "2026-08-09T06:20:00", "arrive": "2026-08-09T11:55:00", "stops": 1,
   "fare_conditions": "Different carrier — needs the airline to endorse the ticket over.",
   "notes": "Arrives sooner but connects in VIE with a 55-minute layover."}],
 "recommended_id": "F1"}
"""


def _user_prompt(request: dict, history: list[dict]) -> str:
    lines = [
        f"Flight: {request.get('airline')} {request.get('flight_number')}",
        f"Route: {request.get('origin')} -> {request.get('destination')}",
        f"What happened: {request.get('disruption')}",
        f"Passenger is at: {request.get('stranded_at')}",
        f"Party size: {request.get('party_size')}",
        f"Must arrive by: {request.get('arrive_by') or 'as soon as possible'}",
        f"Local time now: {request.get('local_now')}",
    ]
    if history:
        lines.append("")
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"  passenger: {turn['prompt']}")
            lines.append(f"  you: {turn['response']}")
    return "\n".join(lines)


def run(request: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). See `docs/PROJECT_PLAN.md` §1 for both shapes."""
    user_prompt = _user_prompt(request, history)

    # Person B: replace everything below with a single lib.llm.call(...) and drop IS_STUB.
    now = datetime.fromisoformat(request["local_now"])
    depart = (now + timedelta(days=1)).replace(hour=9, minute=40, second=0, microsecond=0)
    payload = {
        "options": [
            {
                "id": "F1",
                "airline": request.get("airline") or "XX",
                "flight_number": "STUB100",
                "origin": request.get("origin"),
                "destination": request.get("destination"),
                "depart": depart.isoformat(),
                "arrive": (depart + timedelta(hours=3, minutes=25)).isoformat(),
                "stops": 0,
                "fare_conditions": "STUB DATA — not a real availability search.",
                "notes": "STUB DATA — replaced by Person B in Phase 2.",
            }
        ],
        "recommended_id": "F1",
    }
    return payload, [make_step(MODULE, SYSTEM_PROMPT, user_prompt, payload)]
