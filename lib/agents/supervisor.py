"""Supervisor — reads the passenger's message, works out what they need, runs the crew.

Pattern: question refinement. A passenger standing at a gate writes a short, panicked,
incomplete message, so the Supervisor asks for what is missing *before* dispatching
anyone. That gate is also the cheapest thing in the system: an underspecified turn
costs one LLM call instead of seven.

It also owns the date sync — AccommodationAgent is never asked to guess which nights
the passenger is stranded, it is told, from the departure time of the flight
FlightAgent actually found.

Follow-up turns are narrowed the same way and for the same reason: "anything earlier
than the 04:25?" is a question only FlightAgent can answer, so only FlightAgent runs.
Re-dispatching the whole crew would cost ~7 LLM calls and 2 AeroDataBox units for one
line of conversation.
"""

from datetime import datetime

from lib import llm
from lib.agents import accommodation_agent, documentation_agent, flight_agent

MODULE = "Supervisor"

# How many prior turns reach a prompt. History is re-sent on every call of every turn,
# so an uncapped conversation costs O(n^2) tokens against a $13 project budget
# (`docs/PROJECT_PLAN.md` §7). Six turns is three exchanges of context.
HISTORY_TURNS = 6

REQUIRED_FIELDS = ("flight_number", "origin", "destination", "disruption", "stranded_at")

NEEDS = ("flight", "stay", "rights")
DISRUPTIONS = ("delayed", "cancelled", "denied_boarding")

# Runaway guards, not targets: both outputs are small, but an uncapped completion on a
# reasoning model is an uncapped charge against the $13 (`docs/PROJECT_PLAN.md` §5).
REFINE_MAX_COMPLETION_TOKENS = 4_000
COMPOSE_MAX_COMPLETION_TOKENS = 4_000

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

"stranded_at" is an airport, not a place inside one — give its IATA code where you can work it
out. If they were stopped before leaving, repeat the same code you put in "origin"; never write
the word "origin" itself.

On a first message "needs" is ["flight", "stay", "rights"]. Leave one out only when the
passenger rules it out in words — "I don't need a hotel", "I just want to know what I'm owed".
Never drop one because they did not think to ask: a passenger who does not know they are owed a
bed and a payout is the reason this exists, and a flight that leaves tomorrow means a bed
tonight whether or not they said so. Ask for as little as possible — only "flight_number",
"origin", "destination", "disruption" and "stranded_at" block the crew from starting.

If earlier turns are shown, this is a follow-up. Carry forward every field the passenger
already gave — they will not repeat themselves — and set "needs" to only what this message
actually asks for: a question about times, seats or other departures is ["flight"], one about
where they sleep is ["stay"], one about baggage, meals, money or what the airline owes is
["rights"]. Return an empty "needs" when the question can be answered from what is already on
the table. Re-running the whole crew for a one-line question costs the passenger time and
costs the project money.
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


def _history_block(history: list[dict]) -> list[str]:
    if not history:
        return []
    lines = ["Earlier in this conversation:"]
    for turn in history:
        lines.append(f"  passenger: {turn['prompt']}")
        lines.append(f"  you: {turn['response']}")
    return lines


def _refine_prompt(prompt: str, history: list[dict]) -> str:
    lines = _history_block(history)
    if lines:
        lines.append("")
    lines.append(f"Passenger's message: {prompt.strip()}")
    return "\n".join(lines)


def _text(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _party_size(value) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return 1
    return size if size > 0 else 1


def _request_from(parsed: dict, follow_up: bool) -> dict:
    """The model's JSON forced into the locked request shape (`docs/PROJECT_PLAN.md` §1).

    Everything the rest of the system indexes into is set here whatever came back: a
    missing "local_now" crashes the date sync, and an invented "needs" entry would
    dispatch an agent that does not exist.
    """
    request = {
        field: _text(parsed.get(field))
        for field in (
            "airline", "flight_number", "origin", "destination", "stranded_at", "arrive_by"
        )
    }
    disruption = _text(parsed.get("disruption"))
    request["disruption"] = disruption if disruption in DISRUPTIONS else None
    request["party_size"] = _party_size(parsed.get("party_size"))

    asked = parsed.get("needs")
    needs = [n for n in NEEDS if n in asked] if isinstance(asked, list) else []
    # A first turn is a whole disruption to sort out, so an unreadable "needs" falls back
    # to all three. A follow-up that needs nobody is answered from the conversation —
    # the cheapest turn in the system, and the common case once a plan exists.
    request["needs"] = needs if (needs or follow_up) else list(NEEDS)

    # The model has no clock, and the date sync runs on this.
    request["local_now"] = datetime.now().isoformat(timespec="seconds")
    request["missing"] = [f for f in REQUIRED_FIELDS if not request.get(f)]
    return request


def _extract_request(prompt: str, history: list[dict]) -> tuple[dict, dict]:
    """Question refinement: message -> (request, step).

    History goes into the call so a follow-up inherits what the passenger already
    said, and so `needs` can narrow to the one agent the follow-up actually needs.
    """
    parsed, step = llm.call(
        MODULE,
        REFINE_SYSTEM_PROMPT,
        _refine_prompt(prompt, history),
        expect_json=True,
        max_completion_tokens=REFINE_MAX_COMPLETION_TOKENS,
    )
    if not isinstance(parsed, dict):
        raise llm.LLMError(
            f"{MODULE}: the refinement pass returned {type(parsed).__name__}, not an object",
            steps=[step],
        )
    return _request_from(parsed, bool(history)), step


def _line(label: str, *parts: str) -> str:
    """One digest line, built only from the fields that actually came back."""
    return f"{label}: " + ", ".join(part for part in parts if part) + "."


def _digest(results: dict, failures: list[str]) -> str:
    """What the crew came back with, as flat lines.

    Both the composing call's input and — if that call is the one that fails — the
    plan the passenger gets instead. Nothing here is indexed with []: the agents'
    validation guarantees an id and a parseable date, not a label, and a model that
    left "area" out must not cost the passenger a plan that six calls already paid for.
    """
    lines = []

    flight = _recommended(results.get("flight") or {})
    if flight:
        lines.append(_line(
            "Onward flight",
            f"{flight.get('airline') or ''} {flight.get('flight_number') or ''}".strip(),
            " to ".join(p for p in (flight.get("origin"), flight.get("destination")) if p),
            f"departs {flight['depart']}" if flight.get("depart") else "",
        ))

    stay = _recommended(results.get("stay") or {})
    if stay:
        lines.append(_line(
            "Somewhere to sleep",
            stay.get("name") or "",
            stay.get("area") or "",
            " to ".join(p for p in (stay.get("check_in"), stay.get("check_out")) if p),
            f"{stay['nights']} night(s)" if stay.get("nights") else "",
            "meals included" if stay.get("meals_included") else "no meals",
        ))

    rights = results.get("rights")
    if rights:
        lines.append(f"What you are owed (under {rights.get('regulation')}):")
        for item in rights.get("entitlements") or []:
            source = item.get("source")
            lines.append(
                f"  - {item.get('kind')}: {item.get('summary')}"
                + (f" [{source}]" if source else "")
            )
        for action in rights.get("next_actions") or []:
            lines.append(f"  Next: {action}")

    for failure in failures:
        lines.append(f"Could not complete: {failure}")

    lines.append("")
    lines.append("Ask me to compare any of these, or about the terms of one in particular.")

    return "\n".join(lines)


def _compose_prompt(request: dict, digest: str, history: list[dict]) -> str:
    lines = [
        f"Flight: {request.get('airline')} {request.get('flight_number')}",
        f"Route: {request.get('origin')} -> {request.get('destination')}",
        f"What happened: {request.get('disruption')}",
        f"Party size: {request.get('party_size')}",
        f"Local time now: {request.get('local_now')}",
        "",
        f"They just said: {request.get('_passenger_prompt')}",
    ]
    block = _history_block(history)
    if block:
        lines += [""] + block
    lines += ["", "What the crew came back with:", digest]
    return "\n".join(lines)


def _compose(
    request: dict, results: dict, failures: list[str], history: list[dict]
) -> tuple[str, list[dict]]:
    """Results -> the passenger's plan, as text. Returns (text, steps)."""
    digest = _digest(results, failures)
    try:
        text, step = llm.call(
            MODULE,
            COMPOSE_SYSTEM_PROMPT,
            _compose_prompt(request, digest, history),
            max_completion_tokens=COMPOSE_MAX_COMPLETION_TOKENS,
        )
    except llm.LLMError as exc:
        # Up to six calls and two API quotas are already spent by the time we compose.
        # Losing all of it to the last call is worse than handing over the flat version.
        return digest, exc.steps
    return (text or "").strip() or digest, [step]


def run(prompt: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Returns (response_text, steps) — see `docs/PROJECT_PLAN.md` §1."""
    history = _trim(history)
    steps: list[dict] = []

    request, refine_step = _extract_request(prompt, history)
    # Keep the current utterance available to sub-agents. It is intentionally an
    # internal field: the locked structured-request schema can stay unchanged, while
    # follow-up questions do not disappear during extraction.
    request["_passenger_prompt"] = prompt.strip()
    steps.append(refine_step)

    needs = request["needs"]

    # The gate: never dispatch a crew against a request that is missing what it needs.
    # Nothing to dispatch means nothing is blocked, so a follow-up answered from the
    # conversation is never interrogated for details it already gave.
    if request["missing"] and needs:
        asked = list(dict.fromkeys(QUESTIONS[f] for f in request["missing"]))
        text = "Before I can help, I need a couple of details:\n" + "\n".join(
            f"  - {q}" for q in asked
        )
        return text, steps

    results: dict = {}
    failures: list[str] = []

    # One agent failing must not cost the passenger the rest of the plan.
    def dispatch(label: str, key: str, fn, *args):
        try:
            payload, agent_steps = fn(*args, history)
            results[key] = payload
            steps.extend(agent_steps)
        except Exception as exc:
            # A call that was made and then failed is still a call, and the spec wants
            # every one of them in `steps`. lib.llm.LLMError carries the failed call's
            # step; an agent that got partway through several calls attaches the ones
            # that already succeeded (see documentation_agent).
            steps.extend(getattr(exc, "steps", []))
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

    text, compose_steps = _compose(request, results, failures, history)
    steps.extend(compose_steps)
    return text, steps
