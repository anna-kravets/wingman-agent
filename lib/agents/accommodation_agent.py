"""AccommodationAgent — stays for exactly the nights the new itinerary strands the passenger.

Pattern: role prompt + one-shot example (`docs/PROJECT_PLAN.md` §1), now choosing
from real hotels supplied by `lib/tools/hotels.py`.

`stay_window` is the point of the date sync: the Supervisor derives the nights
from the flight FlightAgent actually found, so this agent never guesses them.

OpenStreetMap knows a hotel's name and where it is and rarely anything else, so
price and meals are estimates and must be labelled as such (design doc D8).
"""

import json
import re

from lib import llm
from lib.tools import hotels

MODULE = "AccommodationAgent"

FAR_FROM_TERMINAL_KM = 15.0
MEALS_FROM_TAG = {"yes": "included", "no": "not_included"}

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


def _normalise(text) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _match(option: dict, candidates: list[dict]) -> dict | None:
    wanted = _normalise(option.get("name"))
    return next((c for c in candidates if _normalise(c.get("name")) == wanted), None)


def _enrich(option: dict, candidate: dict, stay_window: dict) -> dict:
    """Facts from OpenStreetMap, nights from the Supervisor, prose from the model."""
    meals = MEALS_FROM_TAG.get(str(candidate.get("breakfast", "")).lower(), "unknown")
    enriched = {
        "id": option.get("id"),
        "name": candidate.get("name"),
        "kind": candidate.get("kind"),          # absent when it is an ordinary hotel
        "distance_km": candidate.get("distance_km"),
        "city": candidate.get("area"),
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
        "notes": option.get("notes"),
    }
    return {key: value for key, value in enriched.items() if value is not None}


def _caveats(options: list[dict]) -> list[str]:
    """The ones whose trigger is a fact, decided here rather than by the model."""
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
    """Keep only options naming a real property, and fill their facts from it.

    The nights come from the stay window the Supervisor derived from the flight,
    not from the model, so they cannot disagree with the flight that produced them.
    """
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
    payload["caveats"] = _caveats(usable) + model_caveats
    return payload


def run(request: dict, stay_window: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). See `docs/PROJECT_PLAN.md` §1 for both shapes."""
    candidates = hotels.search(request.get("stranded_at"))

    if not candidates:
        # Same reasoning as FlightAgent: no verified property, no invention, no LLM call.
        # A named hotel that does not exist sends a tired passenger to the wrong place.
        raise llm.LLMError(f"{MODULE}: {NO_LIVE_DATA}", steps=[], passenger_message=NO_LIVE_DATA)

    payload, step = llm.call(
        MODULE, SYSTEM_PROMPT,
        _user_prompt(request, stay_window, history, candidates), expect_json=True
    )
    try:
        return _validate(payload, stay_window, candidates), [step]
    except ValueError as exc:
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
