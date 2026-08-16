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

import logging
from datetime import datetime, timedelta

from lib import llm
from lib.agents import accommodation_agent, documentation_agent, flight_agent
from lib.tools import airports, flights

MODULE = "Supervisor"

logger = logging.getLogger(__name__)

# What the passenger reads when an agent could not finish and wrote nothing better.
FAILURE_MESSAGES = {
    "flight": "I could not get onward flight options just now.",
    "stay": "I could not find somewhere to stay just now.",
    "rights": "I could not work out what you are owed just now.",
}

# How many prior turns reach a prompt. History is re-sent on every call of every turn,
# so an uncapped conversation costs O(n^2) tokens against a $13 project budget
# (`docs/PROJECT_PLAN.md` §7). Six turns is three exchanges of context.
HISTORY_TURNS = 6

# Three each is what the agents already return at most, and the digest is re-sent to the
# composing call on every turn — this is the cap that keeps that honest.
MAX_DIGEST_OPTIONS = 3

MEALS_TEXT = {
    "included": "meals included",
    "not_included": "no meals",
    "unknown": "meals not confirmed",
}

# The search agents' caveat protocol (payload spec P4). Ordered: what stops the passenger
# acting comes before what is asked of them, which comes before what is merely worth
# knowing. They are written for whoever is coordinating the plan, so they are translated
# here rather than printed.
CAVEAT_BLOCKS = (
    ("CONFIRM:", "BEFORE YOU ACT ON THIS"),
    ("ASK:", "THINGS I NEED FROM YOU"),
    ("NOTE:", "WORTH KNOWING"),
)

REQUIRED_FIELDS = ("flight_number", "origin", "destination", "disruption", "stranded_at")

# What each need actually blocks on. AccommodationAgent reads exactly one field from
# the request - stranded_at - so "find us a hotel near TLV" was being interrogated for
# a flight number, a route and a disruption type that no hotel search ever touches.
# Anything not listed here keeps the full set.
REQUIRED_BY_NEED = {"stay": ("stranded_at",)}

# Generous on purpose: airlines cancel a fortnight ahead, and flagging that would bounce
# a real passenger. It still catches "today, 11th of November" read in August.
INCIDENT_FUTURE_LIMIT_DAYS = 30
# Not about whether the claim is valid — it is, for months — but about whether a plan
# that finds a bed tonight makes any sense for it.
INCIDENT_PAST_LIMIT_DAYS = 14

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

# A route that cannot exist or an airport nobody can look up leaves the crew nothing to
# search against, so it is worth a round trip. incident_time used to block here too, but
# nothing downstream needs it to be exactly right — the flight window comes from
# local_now (`_stay_window`), and incident_time's only consumer anywhere is one
# informational line in documentation_agent.py. A disruption reported weeks late is the
# product's own core case, not a mistake to interrogate: the date is usually right.
BLOCKING_CONFLICTS = {"route", "stranded_at"}

# Not worth asking about, but a field that cannot be true (arrive_by) or merely reads as
# unusual (incident_time, now that it no longer blocks) must not reach an agent's prompt
# unexplained. Anything listed here is cleared, and the assumption is stated in the plan.
CLEARED_ON_CONFLICT = {"arrive_by", "incident_time"}

CONFLICT_QUESTIONS = {
    "stranded_at": "I could not find an airport with the code {stated}. Which airport are you at?",
}

REFINE_SYSTEM_PROMPT = """You read a message from an air passenger whose flight has just been disrupted and turn it into a structured request.

The passenger is stressed and in a hurry, so the message is usually missing things. Extract what is
there. Do not invent anything — if a field is not stated and cannot be safely inferred, leave it null
and list it in "missing".

Return a JSON object only, no prose:
{"airline", "flight_number", "origin", "destination",
 "disruption": "delayed" | "cancelled" | "denied_boarding" | null,
 "stranded_at", "party_size", "arrive_by", "incident_time",
 "needs": ["flight", "stay", "rights"],
 "conflicts": [{"field", "stated", "reason"}],
 "missing": [field names]}

"stranded_at" is an airport, not a place inside one — give its IATA code where you can work it
out. If they were stopped before leaving, repeat the same code you put in "origin"; never write
the word "origin" itself.

"incident_time" is when the disrupted flight was scheduled to leave, in ISO 8601. Give it whenever
the passenger says or implies it, resolving relative words against the current date and time you are
given. Leave it null if they did not say - it is useful, not required.

"conflicts" is for anything the passenger stated that cannot be true. You are given the current
local date and time below. If they write a relative word - "today", "this morning", "tonight" -
next to a date that is not that day, that is a conflict on "incident_time". So is an airline that
does not match the flight number they gave, or an airport that contradicts the route they described.
Give {"field": which one, "stated": what they wrote, "reason": why it cannot be right}. Report only
what you are sure of, and leave it empty otherwise; date arithmetic is re-checked separately.

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
["rights"]. Return an empty "needs" when the question can be answered from what is already on the table -
including any options listed under "Options already on the table", which were already found and
paid for. Re-running a search for something already found costs the passenger time and costs the
project money.
"""

COMPOSE_SYSTEM_PROMPT = """You are Wingman, writing directly to one passenger whose flight has just been disrupted.

You are a single assistant. Never mention how you work: no tools, no searches, no internal steps, and
never any suggestion that the work was divided up or handed to anyone. The passenger is talking to
you, not to a system.

Speak to one tired person, not to a user. Short sentences, active voice, no jargon and no
regulation-speak they would have to decode.

If "Earlier in this conversation" is not shown below, this is the first message: write the full plan.
Lead with the flight, then where they sleep, then what they are owed and what to do about it - that
is the order they need it in.

If "Earlier in this conversation" IS shown below, this is a follow-up in a conversation already under
way. The passenger has already read everything down there, including your own earlier replies word
for word - do not write them a fresh plan and do not restate facts, figures, or entitlements you
already gave them there. Answer only what they just asked, directly and briefly. Bring back something
from earlier only if this question is actually about it. If you asked them something before and they
still have not answered it, remind them of that one open question in a single line instead of
re-listing everything you asked last time.

Stay on the one thing they asked about. If they ask about their sleep options, answer about sleep
only - do not bring their flight or what they are owed into it, even though you can see those too
below. The same the other way round. When they ask you to compare, compare in real depth using every
field you have for each option there - price, distance, stars, meals, whatever applies - not just
whichever single fact is easiest to restate.

State amounts and entitlements only where the findings support them, and say which document each came
from. Where something could not be finished, say plainly what is missing rather than papering over it.

Anything under "Before you act on this" goes ahead of your recommendation, not after it - and only if
you have not already told them. If there is a "Things I need from you" block, close on it as a direct
question instead of the general invitation.

State each assumption you are given, in your own words, so the passenger can correct it - but only
the ones you have not already told them earlier in this conversation.

End by inviting a follow-up: the passenger can compare options or ask about the terms of any one of
them. Skip that invitation if you already gave it and nothing about the plan has changed since.
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


def _tonight_window(request: dict, flight_payload: dict | None) -> dict:
    """Tonight until tomorrow — for a passenger who asked for a bed outright.

    `_stay_window` decides the nights of the *default plan*, and returns None when the
    onward flight leaves today. It was also deciding whether to search at all, so
    someone who asked in so many words for a hotel got told to go ask the airline desk
    and never saw a single property. An explicit request outranks our inference; the
    plan can still say their flight leaves today and they may not need the room.
    """
    now = datetime.fromisoformat(request["local_now"])
    option = _recommended(flight_payload or {}) or {}
    return {
        "check_in": now.date().isoformat(),
        "check_out": (now.date() + timedelta(days=1)).isoformat(),
        "nights": 1,
        "guests": request.get("party_size"),
        "departs": option.get("depart"),
    }


def _history_block(history: list[dict]) -> list[str]:
    if not history:
        return []
    lines = ["Earlier in this conversation:"]
    for turn in history:
        lines.append(f"  passenger: {turn['prompt']}")
        lines.append(f"  you: {turn['response']}")
    return lines


def _prior_results(history: list[dict]) -> dict:
    """Every agent payload from earlier turns, merged, most recent turn winning.

    The same merge `_prior_results_block` and `_prior_options_digest` each do inline;
    this is the one the dispatch path needs, so a narrowed follow-up can reuse a
    flight already found instead of the stay being silently skipped.
    """
    prior: dict = {}
    for turn in history:                       # oldest first, so later turns win per key
        found = turn.get("results") if isinstance(turn, dict) else None
        if isinstance(found, dict):
            prior.update(found)
    return prior


def _prior_results_block(history: list[dict]) -> list[str]:
    """The options already on the table, identities only.

    Merged per key, most recent turn wins: a flight-only follow-up must not shadow the
    stay found two turns ago, or a third turn re-dispatches AccommodationAgent for a
    room already found and paid for. Only identities, not whole payloads: history is
    re-sent on every call of every turn, so full payloads would cost O(n^2) tokens
    against a $13 project budget (`docs/PROJECT_PLAN.md` §7).
    """
    prior: dict = {}
    for turn in history:                       # oldest first, so later turns win per key
        found = turn.get("results") if isinstance(turn, dict) else None
        if isinstance(found, dict):
            prior.update(found)
    if not prior:
        return []

    lines = ["Options already on the table from earlier in this conversation:"]
    onward = _options(prior.get("flight"))
    if onward:
        lines.append("  flights: " + "; ".join(
            " ".join(p for p in (o.get("id"), o.get("flight_number"),
                                 f"departs {o['depart']}" if o.get("depart") else "") if p)
            for o in onward))
    stays = _options(prior.get("stay"))
    if stays:
        lines.append("  stays: " + "; ".join(
            " ".join(p for p in (o.get("id"), o.get("name"),
                                 f"{o['distance_km']} km" if o.get("distance_km") is not None else "")
                     if p)
            for o in stays))
    return lines if len(lines) > 1 else []


def _prior_options_digest(history: list[dict]) -> list[str]:
    """Every previously-found option, in full — for a follow-up that compares or asks
    for detail on something already found rather than searched for again.

    `_prior_results_block` above stays identity-only because the refinement call only
    needs to recognise what is already on the table, not cite it. The composing call is
    different: "compare the three sleep options" cannot be answered in any depth from an
    id, a name and a distance. Reuses `_flight_line`/`_stay_line` so a prior option reads
    exactly as rich as one freshly found. Still merged per key, most recent turn wins,
    and still capped at `MAX_DIGEST_OPTIONS` — this is bounded by how many options a
    search agent ever returns, not by how long the conversation has run, so it does not
    reintroduce the O(n^2) growth `_prior_results_block` was written to avoid.

    Kept in clearly separate, labelled sections so a request about one category never
    hands the composing call an in-context reason to start comparing another.
    """
    prior: dict = {}
    for turn in history:                       # oldest first, so later turns win per key
        found = turn.get("results") if isinstance(turn, dict) else None
        if isinstance(found, dict):
            prior.update(found)
    if not prior:
        return []

    lines = ["Options already found earlier in this conversation "
             "(answer from these; do not search again for them):"]
    onward = _options(prior.get("flight"))
    if onward:
        lines.append("  Onward flight options:")
        lines += [f"    {_flight_line(o)}" for o in onward]
    stays = _options(prior.get("stay"))
    if stays:
        lines.append("  Sleep options:")
        lines += [f"    {_stay_line(o)}" for o in stays]
    return lines if len(lines) > 1 else []


def _refine_prompt(prompt: str, history: list[dict], local_now: str) -> str:
    lines = _history_block(history)
    lines += _prior_results_block(history)
    if lines:
        lines.append("")
    lines.append(f"Current local date and time: {local_now}")
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


def _when(value) -> datetime | None:
    """A stated time as a naive datetime, or None if it is absent or unparseable."""
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _conflicts(request: dict) -> list[dict]:
    """What the passenger stated that cannot be true, from arithmetic alone.

    The model reports the contradictions only it can see — a relative word next to a
    date that is not that day. This reports the ones it gets quietly wrong. Both lists
    are merged in `_request_from`, with these winning on any shared field.
    """
    now = _when(request.get("local_now"))
    found: list[dict] = []

    incident = _when(request.get("incident_time"))
    if now and incident:
        days = (incident.date() - now.date()).days
        if days > INCIDENT_FUTURE_LIMIT_DAYS:
            found.append({"field": "incident_time", "stated": request["incident_time"],
                          "reason": f"that is {days} days from now"})
        elif -days > INCIDENT_PAST_LIMIT_DAYS:
            found.append({"field": "incident_time", "stated": request["incident_time"],
                          "reason": f"that was {-days} days ago"})

    origin, destination = request.get("origin"), request.get("destination")
    # Only when both are present: a half-stated route is already a missing field, and
    # asking about it twice in two different wordings helps nobody.
    if origin and destination:
        problem = flights.route_problem(origin, destination)
        if problem:
            found.append({"field": "route", "stated": f"{origin} to {destination}",
                          "reason": problem})

    stranded = request.get("stranded_at")
    if stranded and not airports.lookup(stranded):
        found.append({"field": "stranded_at", "stated": stranded,
                      "reason": "that is not an airport code I can look up"})

    arrive_by = _when(request.get("arrive_by"))
    if now and arrive_by and arrive_by < now:
        found.append({"field": "arrive_by", "stated": request["arrive_by"],
                      "reason": "that deadline has already passed"})

    return found


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _conflict_question(conflict: dict, now: str) -> str:
    template = CONFLICT_QUESTIONS.get(conflict["field"])
    if template:
        return template.format(stated=conflict["stated"], now=now)
    if conflict["field"] == "route":
        # route_problem already writes a passenger-ready sentence. It starts lower-case
        # because today it is interpolated after "FlightAgent: ".
        return _sentence(conflict["reason"])
    return f"You said {conflict['stated']} — {conflict['reason']}. Which is right?"


def _request_from(parsed: dict, follow_up: bool, local_now: str) -> dict:
    """The model's JSON forced into the locked request shape (`docs/PROJECT_PLAN.md` §1).

    Everything the rest of the system indexes into is set here whatever came back: a
    missing "local_now" crashes the date sync, and an invented "needs" entry would
    dispatch an agent that does not exist.
    """
    request = {
        field: _text(parsed.get(field))
        for field in (
            "airline", "flight_number", "origin", "destination", "stranded_at",
            "arrive_by", "incident_time",
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

    # The model has no clock, and the date sync runs on this. It is the passenger's
    # wall clock when their browser sent one, because the server's is UTC on Vercel.
    request["local_now"] = local_now

    reported = parsed.get("conflicts")
    from_model = [
        {"field": _text(c.get("field")) or "", "stated": _text(c.get("stated")) or "",
         "reason": _text(c.get("reason")) or ""}
        for c in (reported if isinstance(reported, list) else [])
        if isinstance(c, dict) and _text(c.get("field")) and _text(c.get("reason"))
    ]
    checked = _conflicts(request)
    claimed = {c["field"] for c in checked}
    request["conflicts"] = checked + [c for c in from_model if c["field"] not in claimed]

    request["assumptions"] = []
    for conflict in request["conflicts"]:
        if conflict["field"] in BLOCKING_CONFLICTS:
            continue
        note = f"They said {conflict['stated']}, but {conflict['reason']}."
        if conflict["field"] in CLEARED_ON_CONFLICT:
            request[conflict["field"]] = None
            note += " Working without it."
        request["assumptions"].append(note)

    wanted: set[str] = set()
    for need in request["needs"]:
        wanted.update(REQUIRED_BY_NEED.get(need, REQUIRED_FIELDS))
    # Nothing dispatched blocks on nothing; the follow-up path relies on that.
    request["missing"] = [f for f in REQUIRED_FIELDS
                          if f in wanted and not request.get(f)]
    return request


def _extract_request(prompt: str, history: list[dict], local_now: str) -> tuple[dict, dict]:
    """Question refinement: message -> (request, step).

    History goes into the call so a follow-up inherits what the passenger already
    said, and so `needs` can narrow to the one agent the follow-up actually needs.
    """
    parsed, step = llm.call(
        MODULE,
        REFINE_SYSTEM_PROMPT,
        _refine_prompt(prompt, history, local_now),
        expect_json=True,
        max_completion_tokens=REFINE_MAX_COMPLETION_TOKENS,
    )
    if not isinstance(parsed, dict):
        raise llm.LLMError(
            f"{MODULE}: the refinement pass returned {type(parsed).__name__}, not an object",
            steps=[step],
        )
    return _request_from(parsed, bool(history), local_now), step


def _options(payload: dict | None) -> list[dict]:
    """Every option a payload holds, recommended first, capped."""
    if not payload:
        return []
    options = [o for o in (payload.get("options") or []) if isinstance(o, dict)]
    wanted = payload.get("recommended_id")
    options.sort(key=lambda o: o.get("id") != wanted)
    return options[:MAX_DIGEST_OPTIONS]


def _flight_line(option: dict) -> str:
    """One flight, built only from the fields that actually came back.

    Nothing is indexed with []: the agent's validation guarantees an id and a real
    candidate behind it, not that every optional field survived.
    """
    parts = [
        " ".join(p for p in (option.get("flight_number"), option.get("airline")) if p),
        " to ".join(p for p in (option.get("origin"), option.get("destination")) if p),
        f"departs {option['depart']}" if option.get("depart") else "",
        f"terminal {option['terminal']}" if option.get("terminal") else "",
        f"arrives {option['arrive']}" if option.get("arrive") else "",
        f"{option['duration_minutes']} minutes in the air" if option.get("duration_minutes") else "",
        "arrives the day after it leaves" if option.get("arrives_next_day") else "",
        option.get("aircraft") or "",
        f"status {option['status']}" if option.get("status") else "",
    ]
    return ", ".join(p for p in parts if p) + "."


def _stay_line(option: dict) -> str:
    """One stay. `distance_km` and `meals` replace the two fields Task 6 deleted."""
    distance = option.get("distance_km")
    parts = [
        option.get("name") or "",
        f"a {option['kind']}" if option.get("kind") else "",
        f"{distance} km from the terminal" if distance is not None else "",
        option.get("city") or "",
        option.get("address") or "",
        f"phone {option['phone']}" if option.get("phone") else "",
        option.get("website") or "",
        f"{option['stars']} stars" if option.get("stars") else "",
        f"step-free access: {option['wheelchair']}" if option.get("wheelchair") else "",
        " to ".join(p for p in (option.get("check_in"), option.get("check_out")) if p),
        f"{option['nights']} night(s)" if option.get("nights") else "",
        MEALS_TEXT.get(option.get("meals"), MEALS_TEXT["unknown"]),
        option.get("price_estimate") or "",
    ]
    return ", ".join(p for p in parts if p) + "."


def _split_caveats(results: dict) -> dict[str, list[str]]:
    """Search-agent caveats, routed by prefix and stripped of it.

    An unprefixed one is a note: the model may add its own and the prefix is only asked
    for, while the ones that matter are generated in code and always carry it.
    """
    routed = {prefix: [] for prefix, _ in CAVEAT_BLOCKS}
    for key in ("flight", "stay"):
        for caveat in (results.get(key) or {}).get("caveats") or []:
            text = str(caveat).strip()
            prefix = next(
                (p for p, _ in CAVEAT_BLOCKS if text.upper().startswith(p)), None
            )
            body = text[len(prefix):].strip() if prefix else text
            if body:
                routed[prefix or "NOTE:"].append(body)
    return routed


def _digest(results: dict, failures: list[str]) -> str:
    """What was found, as flat lines.

    Both the composing call's input and — if that call is the one that fails — the plan
    the passenger gets instead, so the headings stay plain English. Nothing is indexed
    with []: a model that left a field out must not cost the passenger a plan that six
    calls already paid for.
    """
    lines: list[str] = []

    onward = _options(results.get("flight"))
    if onward:
        lines.append("ONWARD FLIGHT")
        lines.append(f"  Recommended: {_flight_line(onward[0])}")
        for extra in ("rebooking", "notes"):
            if onward[0].get(extra):
                lines.append(f"    {onward[0][extra]}")
        if len(onward) > 1:
            lines.append("  Also available:")
            lines += [f"    {_flight_line(o)}" for o in onward[1:]]
        lines.append("")

    stays = _options(results.get("stay"))
    if stays:
        lines.append("SOMEWHERE TO SLEEP")
        lines.append(f"  Recommended: {_stay_line(stays[0])}")
        if stays[0].get("notes"):
            lines.append(f"    {stays[0]['notes']}")
        if len(stays) > 1:
            lines.append("  Also available:")
            lines += [f"    {_stay_line(o)}" for o in stays[1:]]
        lines.append("")

    rights = results.get("rights")
    if rights:
        lines.append(f"WHAT YOU ARE OWED (under {rights.get('regulation')})")
        for item in rights.get("entitlements") or []:
            source, confidence = item.get("source"), item.get("confidence")
            lines.append(
                f"  - {item.get('kind')}: {item.get('summary')}"
                + (f" [{source}]" if source else "")
                + (f" ({confidence} confidence)" if confidence else "")
            )
        for action in rights.get("next_actions") or []:
            lines.append(f"  Next: {action}")
        # These carry no NOTE:/ASK:/CONFIRM: prefix and are not that protocol — they are
        # gaps in the evidence the reflection loop found, and belong in their own block.
        for caveat in rights.get("caveats") or []:
            lines.append(f"  Not established from the sources: {caveat}")
        lines.append("")

    routed = _split_caveats(results)
    for prefix, heading in CAVEAT_BLOCKS:
        if routed[prefix]:
            lines.append(heading)
            lines += [f"  - {item}" for item in routed[prefix]]
            lines.append("")

    if failures:
        lines.append("COULD NOT COMPLETE")
        lines += [f"  - {failure}" for failure in failures]
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
    prior = _prior_options_digest(history)
    if prior:
        lines += [""] + prior
    if request.get("assumptions"):
        lines += ["", "Assumptions behind this (see the system prompt for when to mention one):"]
        lines += [f"  - {a}" for a in request["assumptions"]]
    lines += ["", "Findings:", digest]
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


def run(prompt: str, history: list[dict], *,
        local_time: datetime | None = None) -> tuple[str, list[dict], dict]:
    """Returns (response_text, steps, results) — see `docs/PROJECT_PLAN.md` §1.

    `results` is what each agent came back with, so a follow-up can be answered from
    options already paid for instead of re-dispatching for them.
    """
    history = _trim(history)
    steps: list[dict] = []

    local_now = (local_time or datetime.now()).isoformat(timespec="seconds")
    request, refine_step = _extract_request(prompt, history, local_now)
    # Keep the current utterance available to sub-agents. It is intentionally an
    # internal field: the locked structured-request schema can stay unchanged, while
    # follow-up questions do not disappear during extraction.
    request["_passenger_prompt"] = prompt.strip()
    steps.append(refine_step)

    needs = request["needs"]

    blocking = [c for c in request["conflicts"] if c["field"] in BLOCKING_CONFLICTS]

    # The gate: never dispatch against a request that is missing what the crew needs, or
    # that states something that cannot be true. Nothing to dispatch means nothing is
    # blocked, so a follow-up answered from the conversation is never interrogated for
    # details it already gave.
    if (request["missing"] or blocking) and needs:
        asked = list(dict.fromkeys(
            [_conflict_question(c, request["local_now"]) for c in blocking]
            + [QUESTIONS[f] for f in request["missing"]]
        ))
        opener = ("Before I can help, I need to check a couple of things:" if blocking
                  else "Before I can help, I need a couple of details:")
        return opener + "\n" + "\n".join(f"  - {q}" for q in asked), steps, {}

    results: dict = {}
    failures: list[str] = []

    # One agent failing must not cost the passenger the rest of the plan.
    def dispatch(key: str, fn, *args):
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
            # The cause is ours, not the passenger's. `steps` keeps it for the trace and
            # the log keeps it for the failures that carry no step at all.
            logger.exception("%s could not be completed", key)
            reason = getattr(exc, "passenger_message", None) or FAILURE_MESSAGES[key]
            failures.append(_sentence(reason))

    if "flight" in needs:
        dispatch("flight", flight_agent.run, request)

    # Date sync: the stay follows the flight that was actually found, never a guess.
    # A stay-only follow-up ("where can I sleep?") narrows `needs` and so never
    # re-dispatches FlightAgent, leaving `results` empty this turn. Falling back to
    # the flight found earlier is what stops the passenger asking about a bed and
    # getting silence — and it is free, the flight was already paid for.
    if "stay" in needs:
        known_flight = results.get("flight") or _prior_results(history).get("flight")
        stay_window = _stay_window(request, known_flight)
        # `needs == ["stay"]` means the narrowing decided this message was *about* a
        # bed. Then the passenger asked, and being asked outranks our own inference
        # that a same-day flight makes a room unnecessary.
        if not stay_window and needs == ["stay"]:
            stay_window = _tonight_window(request, known_flight)
        if stay_window:
            dispatch("stay", accommodation_agent.run, request, stay_window)

    if "rights" in needs:
        dispatch("rights", documentation_agent.run, request)

    text, compose_steps = _compose(request, results, failures, history)
    steps.extend(compose_steps)
    return text, steps, results
