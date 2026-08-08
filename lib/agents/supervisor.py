"""Supervisor — reads the passenger's message, works out what they need, runs the crew.

Pattern: question refinement. A passenger standing at a gate writes a short, panicked,
incomplete message, so the Supervisor asks for what is missing *before* dispatching
anyone. That gate is also the cheapest thing in the system: an underspecified turn
costs one LLM call instead of seven.

It also owns the date sync — AccommodationAgent is never asked to guess which nights
the passenger is stranded, it is told, from the departure time of the flight
FlightAgent actually found.

Two seams are stubbed until the LLMod key exists (`_extract_request` and `_compose`,
both marked below). Everything else — dispatch, date sync, the history cap, the
partial-failure policy — is real and covered by `tests/`.
"""

import re
from datetime import datetime

from lib.agents import accommodation_agent, documentation_agent, flight_agent
from lib.steps import make_step

MODULE = "Supervisor"

# How many prior turns reach a prompt. History is re-sent on every call of every turn,
# so an uncapped conversation costs O(n^2) tokens against a $13 project budget
# (`docs/PROJECT_PLAN.md` §7). Six turns is three exchanges of context.
HISTORY_TURNS = 6

REQUIRED_FIELDS = ("flight_number", "origin", "destination", "disruption", "stranded_at")

QUESTIONS = {
    "flight_number": "Which airline and flight number was it?",
    "origin": "Which airport were you flying from, and where to?",
    "destination": "Which airport were you flying from, and where to?",
    "disruption": "What happened exactly — delayed, cancelled, or were you denied boarding?",
    "stranded_at": "Which airport are you at right now?",
}

REFINE_SYSTEM_PROMPT = """You read a message from an air passenger whose flight has just been disrupted and turn it into a structured request.

The passenger is stressed and in a hurry, so the message is usually missing things. Extract what is
there. Do not invent anything — if a field is not stated and cannot be safely inferred, leave it null
and list it in "missing".

Return a JSON object only, no prose:
{"airline", "flight_number", "origin", "destination",
 "disruption": "delayed" | "cancelled" | "denied_boarding" | null,
 "stranded_at", "party_size", "arrive_by", "needs": ["flight", "stay", "rights"],
 "missing": [field names]}

"needs" defaults to all three unless the passenger clearly wants less. Ask for as little as
possible: only fields in REQUIRED_FIELDS actually block the crew from starting.
"""

COMPOSE_SYSTEM_PROMPT = """You write the passenger's recovery plan from the results the crew returned.

Speak to one tired person, not to a user. Short sentences, active voice, no jargon and no
regulation-speak they would have to decode. Lead with the flight, then where they sleep, then what
they are owed and what to do about it — that is the order they need it in.

State amounts and entitlements only if a crew result supports them, and say which document each came
from. If part of the crew failed, say plainly what is missing rather than papering over it.

End by inviting a follow-up: the passenger can compare options or ask about the terms of any one of
them.
"""


def _trim(history: list[dict]) -> list[dict]:
    return history[-HISTORY_TURNS:]


def _recommended(payload: dict) -> dict | None:
    """The option a payload recommends, or its first option, or None."""
    options = payload.get("options") or []
    if not options:
        return None
    wanted = payload.get("recommended_id")
    return next((o for o in options if o.get("id") == wanted), options[0])


def _stay_window(request: dict, flight_payload: dict | None) -> dict | None:
    """The nights the passenger is stranded — the date sync.

    None when no stay is needed (the replacement flight leaves today) or when the
    nights cannot be derived (no flight found). Deliberately does not guess: booking
    the wrong nights is worse than saying a flight is needed first.
    """
    if not flight_payload:
        return None
    option = _recommended(flight_payload)
    if not option or not option.get("depart"):
        return None

    now = datetime.fromisoformat(request["local_now"])
    depart = datetime.fromisoformat(option["depart"])
    if depart.date() <= now.date():
        return None

    return {
        "check_in": now.date().isoformat(),
        "check_out": depart.date().isoformat(),
        "nights": (depart.date() - now.date()).days,
        "guests": request.get("party_size"),
        "departs": option["depart"],
    }


def _extract_request(prompt: str, history: list[dict]) -> tuple[dict, dict]:
    """Question refinement: message -> (request, step).

    STUB. Person A replaces the body with a single lib.llm.call(REFINE_SYSTEM_PROMPT,
    ..., expect_json=True); the prompt above is already what it should send. The crude
    parsing here exists only so both branches of the gate are testable without a key.
    """
    text = prompt.strip()
    lowered = text.lower()

    disruption = None
    if "denied boarding" in lowered or "bumped" in lowered:
        disruption = "denied_boarding"
    elif "cancel" in lowered:
        disruption = "cancelled"
    elif "delay" in lowered:
        disruption = "delayed"

    flight = re.search(r"\b([A-Z]{2})\s?(\d{2,4})\b", text)
    route = re.search(r"\b([A-Z]{3})\s*(?:->|→|to)\s*([A-Z]{3})\b", text)

    request = {
        "airline": flight.group(1) if flight else None,
        "flight_number": flight.group(0).replace(" ", "") if flight else None,
        "origin": route.group(1) if route else None,
        "destination": route.group(2) if route else None,
        "disruption": disruption,
        "stranded_at": route.group(1) if route else None,
        "party_size": 1,
        "arrive_by": None,
        "needs": ["flight", "stay", "rights"],
        "local_now": datetime.now().isoformat(timespec="seconds"),
    }
    request["missing"] = [f for f in REQUIRED_FIELDS if not request.get(f)]

    step = make_step(MODULE, REFINE_SYSTEM_PROMPT, prompt, request)
    return request, step


def _compose(request: dict, results: dict, failures: list[str], history: list[dict]) -> tuple[str, dict]:
    """Results -> the passenger's plan, as text. Returns (text, step).

    STUB. Person A replaces the body with a single lib.llm.call(COMPOSE_SYSTEM_PROMPT, ...).
    The deterministic assembly below keeps the endpoint useful in the meantime.
    """
    lines = []

    flight = _recommended(results.get("flight") or {})
    if flight:
        lines.append(
            f"Onward flight: {flight['airline']} {flight['flight_number']}, "
            f"{flight['origin']} to {flight['destination']}, departs {flight['depart']}."
        )

    stay = _recommended(results.get("stay") or {})
    if stay:
        meals = "meals included" if stay.get("meals_included") else "no meals"
        lines.append(
            f"Somewhere to sleep: {stay['name']}, {stay['area']}, "
            f"{stay['check_in']} to {stay['check_out']} ({stay['nights']} night(s), {meals})."
        )

    rights = results.get("rights")
    if rights:
        lines.append(f"What you are owed (under {rights.get('regulation')}):")
        for item in rights.get("entitlements", []):
            lines.append(f"  - {item['kind']}: {item['summary']} [{item['source']}]")
        for action in rights.get("next_actions", []):
            lines.append(f"  Next: {action}")

    for failure in failures:
        lines.append(f"Could not complete: {failure}")

    lines.append("")
    lines.append("Ask me to compare any of these, or about the terms of one in particular.")

    text = "\n".join(lines)
    return text, make_step(MODULE, COMPOSE_SYSTEM_PROMPT, str(results), {"text": text})


def run(prompt: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Returns (response_text, steps) — see `docs/PROJECT_PLAN.md` §1."""
    history = _trim(history)
    steps: list[dict] = []

    request, refine_step = _extract_request(prompt, history)
    steps.append(refine_step)

    # The gate: never dispatch a crew against a request that is missing what it needs.
    if request["missing"]:
        asked = list(dict.fromkeys(QUESTIONS[f] for f in request["missing"]))
        text = "Before I can help, I need a couple of details:\n" + "\n".join(
            f"  - {q}" for q in asked
        )
        return text, steps

    needs = request.get("needs") or []
    results: dict = {}
    failures: list[str] = []

    # One agent failing must not cost the passenger the rest of the plan.
    def dispatch(label: str, key: str, fn, *args):
        try:
            payload, agent_steps = fn(*args, history)
            results[key] = payload
            steps.extend(agent_steps)
        except Exception as exc:
            failures.append(f"{label} ({exc})")

    if "flight" in needs:
        dispatch("onward flights", "flight", flight_agent.run, request)

    # Date sync: the stay follows the flight that was actually found, never a guess.
    if "stay" in needs:
        stay_window = _stay_window(request, results.get("flight"))
        if stay_window:
            dispatch(
                "somewhere to sleep",
                "stay",
                accommodation_agent.run,
                request,
                stay_window,
            )

    if "rights" in needs:
        dispatch("your entitlements", "rights", documentation_agent.run, request)

    text, compose_step = _compose(request, results, failures, history)
    steps.append(compose_step)
    return text, steps
