"""AccommodationAgent — stays for exactly the nights the new itinerary strands the passenger.

Pattern: role prompt + one-shot example (`docs/PROJECT_PLAN.md` §1), now choosing
from real hotels supplied by `lib/tools/hotels.py`.

`stay_window` is the point of the date sync: the Supervisor derives the nights
from the flight FlightAgent actually found, so this agent never guesses them.

OpenStreetMap knows a hotel's name and where it is and rarely anything else, so
price and meals are estimates and must be labelled as such (design doc D8).
"""

import json
from datetime import date

from lib import llm
from lib.tools import hotels

MODULE = "AccommodationAgent"

SYSTEM_PROMPT = """You find somewhere for a stranded air passenger to sleep, for a fixed set of nights.

You are given a list of REAL hotels near the airport, with the exact distance to the terminal.
Choose only from that list and never invent a property or change its name.

The nights are given to you and are NOT negotiable — they come from the replacement flight the
passenger is taking. Every option must use exactly the check-in and check-out dates you are given.

You do not book, hold or pay for anything. You propose options the passenger acts on themselves.
Never name a booking website.

Prioritise, in order: close enough to reach the airport for the departure, then anything the data
tells you about the property.

You do NOT know prices, availability, or whether meals are included — that information is not in
your source. "price_estimate" must read as an estimate. Set "meals_included" to true only if the
data explicitly says so; otherwise set it false and say in "notes" that meals were not confirmed
and to check at the desk. Never assert a fact about a real named business that you were not given.

Return a JSON object only, no prose:
{"options": [{"id", "name", "area", "check_in", "check_out", "nights",
              "price_estimate", "meals_included", "notes"}],
 "recommended_id": "<id>"}

Example
-------
Hotels: [{"name": "Airport Plaza", "distance_km": 2.4, "stars": "4", "breakfast": null, "area": "Lod"},
         {"name": "City Central Inn", "distance_km": 11.2, "stars": null, "breakfast": null, "area": "Tel Aviv"}]
Request: 2 guests stranded at TLV, check in 2026-08-09, check out 2026-08-10, 1 night, onward flight departs 2026-08-10T09:40.
Response:
{"options": [{"id": "H1", "name": "Airport Plaza", "area": "2.4 km from the terminal, Lod", "check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1, "price_estimate": "Roughly EUR 110-140 for the night (estimate - not a quoted price)", "meals_included": false, "notes": "Closest to the terminal, which matters for an 09:40 departure. Meals were not confirmed - ask at the desk, and keep the receipt if you are claiming care costs back."}, {"id": "H2", "name": "City Central Inn", "area": "11.2 km from the terminal, Tel Aviv", "check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1, "price_estimate": "Roughly EUR 80-100 for the night (estimate - not a quoted price)", "meals_included": false, "notes": "Cheaper but 11 km out, so leave a clear hour to get back for the flight. Meals not confirmed."}],
 "recommended_id": "H1"}
"""

DEGRADED_NOTE = (
    "No live hotel data was available for this airport. Propose plausible options from your own "
    "knowledge, and put 'Illustrative option - not live availability.' in the notes of every one."
)


def _user_prompt(request: dict, stay_window: dict, history: list[dict],
                 candidates: list[dict]) -> str:
    lines = [
        f"Passenger is stranded at: {request.get('stranded_at')}",
        f"Guests: {stay_window.get('guests')}",
        f"Check in: {stay_window.get('check_in')}",
        f"Check out: {stay_window.get('check_out')}",
        f"Nights: {stay_window.get('nights')}",
        f"Onward flight departs: {stay_window.get('departs')}",
        "",
    ]
    if candidates:
        lines.append("Hotels (real, near the airport - choose only from these):")
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


def _validate(payload: dict, stay_window: dict) -> dict:
    """Drop options that do not match the nights the Supervisor derived.

    Booking the wrong nights is the one failure that leaves the passenger worse
    off than no answer at all, so a mismatch is dropped rather than reported.
    """
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")

    wanted_in = stay_window.get("check_in")
    wanted_out = stay_window.get("check_out")

    usable = []
    for option in payload.get("options") or []:
        if not isinstance(option, dict) or not option.get("id"):
            continue
        try:
            date.fromisoformat(option.get("check_in") or "")
            date.fromisoformat(option.get("check_out") or "")
        except (TypeError, ValueError):
            continue
        if option["check_in"] != wanted_in or option["check_out"] != wanted_out:
            continue
        usable.append(option)

    if not usable:
        raise ValueError("no option came back on the nights the flight implies")

    payload["options"] = usable
    if payload.get("recommended_id") not in {o["id"] for o in usable}:
        payload["recommended_id"] = usable[0]["id"]
    return payload


def run(request: dict, stay_window: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). See `docs/PROJECT_PLAN.md` §1 for both shapes."""
    candidates = hotels.search(request.get("stranded_at"))

    payload, step = llm.call(
        MODULE, SYSTEM_PROMPT,
        _user_prompt(request, stay_window, history, candidates), expect_json=True
    )
    try:
        return _validate(payload, stay_window), [step]
    except ValueError as exc:
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
