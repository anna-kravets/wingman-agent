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

A candidate may carry extra detail — "phone", "website", "address", "stars", "wheelchair",
"internet", "brand", or "kind" when it is not a hotel (hostel, guest house, apartment). Use only
what is there, never fill a gap. **If a phone number is given, put it in "notes" and tell the
passenger to call and confirm the room and the rate** — that is the one way to settle the two
things you cannot: whether there is a bed free tonight and what it actually costs. Mention step-free
access when the data says so, and say plainly when somewhere is a hostel or apartment rather than a
hotel, because that changes what the passenger should expect.

Return a JSON object only, no prose:
{"options": [{"id", "name", "area", "check_in", "check_out", "nights",
              "price_estimate", "meals_included", "notes"}],
 "recommended_id": "<id>"}

Example
-------
Hotels: [{"name": "Airport Plaza", "distance_km": 2.4, "area": "Lod", "stars": "4", "phone": "+972 3 000 0000", "address": "12 HaNasi", "wheelchair": "yes"},
         {"name": "City Central Inn", "distance_km": 11.2, "area": "Tel Aviv", "kind": "hostel"}]
Request: 2 guests stranded at TLV, check in 2026-08-09, check out 2026-08-10, 1 night, onward flight departs 2026-08-10T09:40.
Response:
{"options": [{"id": "H1", "name": "Airport Plaza", "area": "2.4 km from the terminal - 12 HaNasi, Lod", "check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1, "price_estimate": "Roughly EUR 110-140 for the night (estimate - not a quoted price)", "meals_included": false, "notes": "Closest to the terminal, which matters for an 09:40 departure. Call +972 3 000 0000 to confirm a room and the rate - I cannot check either. Meals were not confirmed, so ask at the desk and keep the receipt if you are claiming care costs back. Step-free access."}, {"id": "H2", "name": "City Central Inn", "area": "11.2 km from the terminal, Tel Aviv", "check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1, "price_estimate": "Roughly EUR 40-70 for the night (estimate - not a quoted price)", "meals_included": false, "notes": "A hostel rather than a hotel, so expect shared facilities. Cheaper, but 11 km out - leave a clear hour to get back for the flight. No phone number listed, so turning up is a gamble at this hour."}],
 "recommended_id": "H1"}
"""

NO_LIVE_DATA = (
    "no hotels could be found near this airport in OpenStreetMap, so nothing could be "
    "verified. Nothing was invented - ask at the airline's desk, which is also where a "
    "hotel voucher would be issued if you are owed one."
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
    lines.append("Hotels (real, near the airport - choose only from these):")
    lines.append(json.dumps(candidates, ensure_ascii=False))

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

    if not candidates:
        # Same reasoning as FlightAgent: no verified property, no invention, no LLM call.
        # A named hotel that does not exist sends a tired passenger to the wrong place.
        raise llm.LLMError(f"{MODULE}: {NO_LIVE_DATA}", steps=[])

    payload, step = llm.call(
        MODULE, SYSTEM_PROMPT,
        _user_prompt(request, stay_window, history, candidates), expect_json=True
    )
    try:
        return _validate(payload, stay_window), [step]
    except ValueError as exc:
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
