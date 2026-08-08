"""AccommodationAgent — stays for exactly the nights the new itinerary strands the passenger.

Pattern: role prompt + one-shot example (`docs/PROJECT_PLAN.md` §1).

The `stay_window` argument is the whole point of the date sync: the Supervisor works
out the nights from the flight FlightAgent actually found, so this agent never has to
guess them.

STUB — see `IS_STUB`. Person B replaces the body of `run()`.

While it is a stub the step below describes an LLM call that never happened — the
trace is the right *shape* but is not a true record. The spec requires `steps[]` to
describe every call actually made, so this must not ship: it stops being fiction the
moment `run()` calls through `lib/llm.py` for real.
"""

from lib.steps import make_step

IS_STUB = True

MODULE = "AccommodationAgent"

SYSTEM_PROMPT = """You find somewhere for a stranded air passenger to sleep, for a fixed set of nights.

The nights are given to you and are not negotiable — they come from the replacement flight the
passenger is taking. Do not propose stays that end before the flight or run past it.

You do not book, hold, or pay for anything. You propose options the passenger acts on themselves.
Never name a specific booking website.

Prioritise, in order: close enough to reach the airport for the departure, availability tonight, and
whether meals are included — a passenger owed care under EU 261 or a Contract of Carriage should be
told when a stay already covers what they would otherwise claim for.

Return a JSON object only, no prose:
{"options": [{"id", "name", "area", "check_in", "check_out", "nights",
              "price_estimate", "meals_included", "notes"}],
 "recommended_id": "<id>"}

Example
-------
Request: 2 adults stranded at TLV, need 2026-08-08 to 2026-08-09, 1 night, flight departs 09:40.
Response:
{"options": [
  {"id": "H1", "name": "Airport Plaza", "area": "8 minutes from TLV by shuttle",
   "check_in": "2026-08-08", "check_out": "2026-08-09", "nights": 1,
   "price_estimate": "EUR 120 total", "meals_included": true,
   "notes": "Breakfast from 05:30, which clears the 09:40 departure. 24h shuttle."},
  {"id": "H2", "name": "City Central Inn", "area": "central Tel Aviv, 30 minutes from TLV",
   "check_in": "2026-08-08", "check_out": "2026-08-09", "nights": 1,
   "price_estimate": "EUR 85 total", "meals_included": false,
   "notes": "Cheaper, but leaves a 05:00 start to make the flight."}],
 "recommended_id": "H1"}
"""


def _user_prompt(request: dict, stay_window: dict, history: list[dict]) -> str:
    lines = [
        f"Passenger is stranded at: {request.get('stranded_at')}",
        f"Guests: {stay_window.get('guests')}",
        f"Check in: {stay_window.get('check_in')}",
        f"Check out: {stay_window.get('check_out')}",
        f"Nights: {stay_window.get('nights')}",
        f"Onward flight departs: {stay_window.get('departs')}",
    ]
    if history:
        lines.append("")
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"  passenger: {turn['prompt']}")
            lines.append(f"  you: {turn['response']}")
    return "\n".join(lines)


def run(request: dict, stay_window: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). See `docs/PROJECT_PLAN.md` §1 for both shapes."""
    user_prompt = _user_prompt(request, stay_window, history)

    # Person B: replace everything below with a single lib.llm.call(...) and drop IS_STUB.
    payload = {
        "options": [
            {
                "id": "H1",
                "name": "STUB DATA — not a real availability search",
                "area": f"near {request.get('stranded_at')}",
                "check_in": stay_window.get("check_in"),
                "check_out": stay_window.get("check_out"),
                "nights": stay_window.get("nights"),
                "price_estimate": "STUB DATA",
                "meals_included": True,
                "notes": "STUB DATA — replaced by Person B in Phase 2.",
            }
        ],
        "recommended_id": "H1",
    }
    return payload, [make_step(MODULE, SYSTEM_PROMPT, user_prompt, payload)]
