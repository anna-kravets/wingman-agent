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
import re
from datetime import datetime, timedelta

from lib import llm
from lib.citations import extract_citations
from lib.agents import accommodation_agent, documentation_agent, flight_agent
from lib.rag.routing import EU_EEA_AIRPORTS, ISRAEL_AIRPORTS, US_AIRPORTS
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

# The rights corpus holds three jurisdictions and no others (`lib/rag/routing.py`). An
# airport outside all three attaches no law, so DocumentationAgent answers that end of
# the route from whatever the model already believes - which is exactly the unsourced
# guessing the reflection loop exists to prevent. Say so instead of hiding it.
SUPPORTED_REGIONS = "the US, Israel, and the EU/EEA (Switzerland included)"
SUPPORTED_AIRPORTS = US_AIRPORTS | ISRAEL_AIRPORTS | EU_EEA_AIRPORTS

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
STAY_REQUEST_RE = re.compile(
    r"\b(hotels?|hotles?|accommodat(?:ion|ions|e)|rooms?|beds?|sleep|overnight)\b",
    re.IGNORECASE,
)

# Runaway guards, not output targets. Keep the shared 10k ceiling used by the
# DocumentationAgent: a critical answer must not be truncated because its plan contains
# several flights, hotels and legal regimes. Normal completions remain far shorter.
REFINE_MAX_COMPLETION_TOKENS = 10_000
COMPOSE_MAX_COMPLETION_TOKENS = 10_000
COMPOSE_REPAIR_MAX_COMPLETION_TOKENS = 10_000

QUESTIONS = {
    "flight_number": "Which airline and flight number was it?",
    "origin": "Which airport were you flying from, and where to?",
    "destination": "Which airport were you flying from, and where to?",
    "disruption": "What happened exactly — delayed, cancelled, or were you denied boarding?",
    "stranded_at": "Which airport are you at right now?",
}

# Three of those questions cover two fields at once, and asking the whole thing back when
# the passenger already gave half of it reads as not having listened. Each entry is the
# field that makes up the other half, plus the question to ask when that half is known.
HALF_QUESTIONS = {
    "flight_number": ("airline", "What was the flight number?"),
    "origin": ("destination", "Which airport were you flying from?"),
    "destination": ("origin", "Where were you flying to?"),
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

"origin", "destination" and "stranded_at" are airports, not places inside one — give the IATA
code whenever you can work it out, including from a city name they wrote instead ("Dublin" is
"DUB"). Leave the city as written only when it has several airports and they did not say which.
If they were stopped before leaving, repeat in "stranded_at" the same code you put in "origin";
never write the word "origin" itself.

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
regulation-speak they would have to decode. Naming the article or section a right comes
from is not regulation-speak - say it plainly, in the sentence that gives them the right
("Article 8 lets you choose a refund or a new flight"), so they can look it up and quote it.

"Party size" is the total number of travellers, including the passenger and any child mentioned in
their message. Never add the child a second time. If Party size is 2, every room or group reference
must say 2, never 3.

If "Earlier in this conversation" is not shown below, this is the first message: write the full plan.
Lead with the flight, then where they sleep, then what they are owed and what to do about it - that
is the order they need it in.

Completeness is part of correctness. Do not silently compress away a finding:
- For each flight shown, keep its flight number and departure time. These are schedule options, not
  confirmed seats or completed rebookings; say what the airline still needs to confirm. The option
  labelled "Recommended" in the findings is the recovery plan. Never call a different, near-departure
  flight the "earliest reasonable replacement", never tell the passenger to push for it first, and
  never undo the deterministic recommendation in your prose. It may be mentioned only as an urgent
  possibility that is unsafe to rely on until confirmed. Where a flight carries a note about how
  hard it is to rebook onto - the same airline being simpler than a new one - keep it: that is the
  difference between a queue they can join now and a booking they have to negotiate.
- For each sleep option shown, keep its name, distance, rough price range, property type, and what is
  or is not known about meals and availability. Every price is an estimate, never a quote - say so
  once. Where a property comes with a phone number, give them that number and tell them to ring it:
  nobody has held a room for them, and that call is the only way they can find out. A passenger
  asking for suggestions needs the prices in the first answer, not only after asking for a
  comparison. Check-in/check-out are requested stay dates, not proof of room availability: never
  label those dates "Availability" and always say that actual room availability was not checked or
  confirmed.
- Cover every entitlement listed under "What you are owed". Preserve exact payment deadlines,
  monetary amounts, important exceptions, the human-readable document/law named in its source,
  and its supplied section/article references. If the findings say a requested topic such as
  baggage was not established, say that too.
- Keep every step listed under "Next" as a step of its own. Do not merge two into one and do not
  drop the last ones because the list is getting long - the step that files the written compensation
  claim is the one that actually gets them paid, and it is always the easiest to lose.
- Put every line of any "Things I need from you" block to them as a direct question, and end on
  those questions. Each one is something only they can answer and nothing else in the plan settles.
  Never replace them with a general offer to help further.
- An evidence gap means "the supplied sources did not establish this", not "you have no right".
  Never turn a missing baggage passage into a categorical statement that baggage creates no remedy.
- If the passenger says checked bags are still with the airline, give one practical baggage action:
  ask the airline whether the bags will stay checked through to the replacement flight or be returned,
  and how they will be transferred if the replacement uses another carrier. Do not invent a legal
  baggage entitlement when the findings do not establish one.

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
you have not already told them.

If you are given a "Timing conflict" line, the flight you are recommending and the room you are
recommending cannot both be right. Say so in the same breath as the room, naming that flight: the
room only matters if they are not on it. Never hand them a flight leaving today and a bed for tonight
as though the two sit together. This is not a footnote and not an afterthought at the end.

State each assumption you are given, in your own words, so the passenger can correct it - but only
the ones you have not already told them earlier in this conversation.

If you are given a "Coverage limit" line, say it before anything you tell them about what they are
owed: name the regions whose rules you actually hold, name the airport that falls outside them, and
say that the entitlements at that end may be incomplete. Do not bury it at the end and do not soften
it away. Say it once, in your own words.

End by inviting a follow-up: the passenger can compare options or ask about the terms of any one of
them. Skip that invitation if you already gave it and nothing about the plan has changed since - and
skip it whenever there is a "Things I need from you" block, because those questions are the ending
instead. Never write both: a general "let me know if you want anything else" in place of the specific
thing you actually need from them is the one ending that wastes their next message.
"""

COMPOSE_REPAIR_SYSTEM_PROMPT = COMPOSE_SYSTEM_PROMPT + """

You are revising a draft that failed a deterministic completeness check. Rewrite the entire answer,
not just the missing sentences. Correct every listed omission using only the supplied findings. Keep
the answer calm and readable, but never trade away a concrete price, deadline, source document,
requested evidence gap, or option identity for brevity.
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
    return {
        "check_in": now.date().isoformat(),
        "check_out": (now.date() + timedelta(days=1)).isoformat(),
        "nights": 1,
        "guests": request.get("party_size"),
        # This fallback exists precisely because the selected flight does not imply
        # an overnight stay. Passing its same-day departure to AccommodationAgent
        # produced a one-night booking optimized for a flight leaving before check-in.
        "departs": None,
    }


def _explicit_stay_requested(request: dict) -> bool:
    """Whether the passenger asked for a place to sleep in this message."""
    return bool(STAY_REQUEST_RE.search(str(request.get("_passenger_prompt") or "")))


def _same_day_flight_assumption(request: dict, flight_payload: dict | None) -> str | None:
    """Explain the branch behind a fallback hotel search, if there is one."""
    option = _recommended(flight_payload or {})
    if not option or not option.get("depart"):
        return None
    now = datetime.fromisoformat(request["local_now"])
    depart = datetime.fromisoformat(option["depart"])
    if depart.date() != now.date():
        return None
    number = option.get("flight_number") or "the same-day flight"
    return (f"Hotel options assume you are not taking {number} today; confirm your onward "
            "flight before relying on a room for tonight.")


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
                      "reason": airports.unknown_reason(stranded)})

    arrive_by = _when(request.get("arrive_by"))
    if now and arrive_by and arrive_by < now:
        found.append({"field": "arrive_by", "stated": request["arrive_by"],
                      "reason": "that deadline has already passed"})

    return found


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _missing_question(field: str, request: dict) -> str:
    """The question for a missing field, narrowed to the half the passenger did not give."""
    half = HALF_QUESTIONS.get(field)
    if half and request.get(half[0]):
        return half[1]
    return QUESTIONS[field]


def _conflict_question(conflict: dict) -> str:
    if conflict["field"] in BLOCKING_CONFLICTS:
        # route_problem and airports.unknown_reason already write passenger-ready
        # sentences - which airport was meant, or that the code cannot be looked up.
        # They start lower-case because they are interpolated after "FlightAgent: ".
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
    # Passengers say "Dublin", not "DUB", and everything downstream - the flight
    # search, the hotel search, the rights routing - only speaks codes. Resolve here,
    # the one place the fields are set, so no consumer has to. A city with several
    # airports stays as written and becomes a question (`_conflicts`).
    for field in ("origin", "destination", "stranded_at"):
        request[field] = airports.resolve(request[field]) or request[field]

    # A departure that was delayed, cancelled or overbooked has not left, so the
    # passenger is standing at its origin. REFINE_SYSTEM_PROMPT asks the model to say
    # so and it obliges about half the time - measured, five identical replays of one
    # real message. The other half asked a passenger who had just given their route
    # where they were, and asking again only rolled the dice again. Only from an
    # origin that resolves: copying an ambiguous city would turn one unanswerable
    # question into two.
    if not request["stranded_at"] and airports.lookup(request["origin"] or ""):
        request["stranded_at"] = request["origin"]

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


def _pending_needs(history: list[dict]) -> list[str]:
    """Needs still outstanding from the immediately preceding turn, if it was gated
    (missing fields / a blocking conflict) before anything could be dispatched.

    Without this, answering a gate's clarifying question reads as a narrow follow-up
    about only that one field, and whatever else the original message asked for
    (never dispatched, since the gate fired first) silently falls off the plan.
    """
    if not history:
        return []
    pending = (history[-1].get("results") or {}).get("_pending_needs")
    return [n for n in pending if n in NEEDS] if isinstance(pending, list) else []


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
    pending = _pending_needs(history)
    if pending:
        carried = parsed.get("needs")
        parsed["needs"] = list(dict.fromkeys((carried if isinstance(carried, list) else []) + pending))
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


def _coverage_warning(request: dict) -> str | None:
    """Warn when either end of the route sits outside every jurisdiction we hold.

    Both halves matter. Neither end supported means nothing said about entitlements is
    grounded. One end supported is the quieter failure: the law of the covered end is
    retrieved and reads like a complete answer, while the other end's regime - UK261 on
    an LHR departure, say - is missing with nothing on the page to say so.
    """
    codes = [str(request.get(field) or "").strip().upper()
             for field in ("origin", "destination")]
    outside = list(dict.fromkeys(c for c in codes if c and c not in SUPPORTED_AIRPORTS))
    if not outside:
        return None
    subject = " and ".join(outside)
    verb = "is" if len(outside) == 1 else "are"
    return (f"Coverage limit: I only hold passenger-rights rules for {SUPPORTED_REGIONS}. "
            f"{subject} {verb} outside that, so anything I say about what you are owed "
            "at that end of the route may be incomplete or inaccurate.")


# The words a caveat uses for a property kind AccommodationAgent ranked but the digest
# cap then cut. Advice to prefer a hotel over the hostel is worse than useless once the
# hostel is not on the page: gpt-5.4-mini turned the dangling reference into "For 2
# guests, these are the three hotel-style options shown" - a comparison the passenger
# cannot see. Kinds come from `lib/tools/hotels.py`.
KIND_WORDS = {
    "hostel": ("hostel",),
    "guest_house": ("guest house", "guesthouse", "guest_house"),
    "apartment": ("apartment", "apartments"),
    "motel": ("motel",),
}


def _dangling(caveat: str, shown: list[dict], every: list[dict]) -> bool:
    """Whether a stay caveat only talks about options the cap removed."""
    lowered = caveat.lower()
    cut = {str(o.get("kind") or "") for o in every} - {str(o.get("kind") or "") for o in shown}
    return any(word in lowered
               for kind in cut for word in KIND_WORDS.get(kind, ()))


def _split_caveats(results: dict) -> dict[str, list[str]]:
    """Search-agent caveats, routed by prefix and stripped of it.

    An unprefixed one is a note: the model may add its own and the prefix is only asked
    for, while the ones that matter are generated in code and always carry it.
    """
    routed = {prefix: [] for prefix, _ in CAVEAT_BLOCKS}
    stay = results.get("stay") or {}
    shown, every = _options(stay), [o for o in (stay.get("options") or []) if isinstance(o, dict)]
    for key in ("flight", "stay"):
        for caveat in (results.get(key) or {}).get("caveats") or []:
            text = str(caveat).strip()
            if key == "stay" and _dangling(text, shown, every):
                continue
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
        actions = rights.get("next_actions") or []
        if actions:
            lines.append("  Next:")
            lines += [f"    - {action}" for action in actions]
        # These carry no NOTE:/ASK:/CONFIRM: prefix and are not that protocol — they are
        # gaps in the evidence the reflection loop found, and belong in their own block.
        caveats = rights.get("caveats") or []
        if caveats:
            lines.append("  Not established from the sources:")
            lines += [f"    - {caveat}" for caveat in caveats]
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


# Phrased as an instruction about the passenger's experience, not as a block of content.
# A bare delimited list reads to the model as text to emit, and gpt-5.4-mini duly
# pasted it into the answer, separators included.
REQUIRED_HEADING = ("Somewhere in your answer the passenger has to see each of these "
                    "exact words and figures:")

# Document names the guard insists on seeing spelled out.
SOURCE_DOCUMENTS = ("Aviation Services Law", "Conditions of Carriage",
                    "Contract of Carriage")

# Mirrors the provision pattern inside linkifyCitations (public/index.html). Keep the
# two in step: it decides which strings in the answer become clickable sources.
PROVISION_RE = re.compile(
    r"(?:s\.|section|Art\.|Article)\s*\d+[\w().-]*(?:\([^)]*\))*?(?:-\([^)]*\))?",
    re.IGNORECASE,
)


def _required_tokens(results: dict) -> list[str]:
    """The concrete strings a fluent rewrite tends to drop.

    A hint for the composing call, not a mirror of _composition_issues: the guard still
    catches whatever slips through, so the two lists are free to drift. Identities,
    figures, document names and the section/article a right comes from - not the
    guard's wording checks, because telling a model which adjectives to use makes it
    write like a form.

    The article references earn their place twice over. The GUI builds its source
    links out of exactly these strings (`linkifyCitations` in public/index.html), so an
    answer that never names Article 8 is also an answer with nothing to click.
    """
    tokens: list[str] = []
    for option in _options(results.get("flight")):
        if option.get("flight_number"):
            tokens.append(str(option["flight_number"]))
        # Departure times survive composition on their own; arrival times only did for
        # the recommended flight, leaving the backups as departures with no landing time
        # - the one figure that decides whether a backup is any use. Cheap to ask for:
        # this list costs nothing but prompt tokens, unlike a guard, which costs a call.
        if option.get("arrive"):
            tokens.append(str(option["arrive"])[11:16])

    for option in _options(results.get("stay")):
        # The number a passenger has to ring to confirm a room the search could not
        # confirm. AccommodationAgent's own ASK caveat tells them to call it.
        if option.get("phone"):
            tokens.append(str(option["phone"]))
        if option.get("name"):
            tokens.append(str(option["name"]))
        price = str(option.get("price_estimate") or "")
        tokens += [aliases[0] for _, aliases in _money_requirements(price)]
        tokens += [amount.replace(",", "") for amount in
                   re.findall(r"\d[\d,]*(?:\.\d+)?", price)]

    # The GUI links a citation's own reference, or a provision inside it, wherever that
    # exact string appears in the answer - and nothing else. "Article 8" on its own does
    # not link; "Art. 8(1)(a)" does, and so does "14 CFR Part 260 § 260.6" whole, which
    # holds no provision the pattern can find. Ask for the shortest form that works.
    linkable: list[str] = []
    for citation in results.get("citations") or []:
        for reference in citation.get("references") or []:
            reference = str(reference)
            linkable += PROVISION_RE.findall(reference) or [reference]
    tokens += linkable

    for item in (results.get("rights") or {}).get("entitlements") or []:
        summary = str(item.get("summary") or "")
        tokens += [aliases[0] for _, aliases in
                   _deadline_requirements(summary) + _money_requirements(summary)]
        source = str(item.get("source") or "")
        tokens += [name for name in SOURCE_DOCUMENTS if name in source]
        # An article no citation spells out still has to be named, or the guard reports
        # it missing. The spelled-out form is the one a passenger can read.
        for _, aliases in _citation_requirements(source):
            if not _has_any(" ".join(linkable), aliases):
                tokens.append(max(aliases, key=len))

    return list(dict.fromkeys(tokens))


def _compose_prompt(request: dict, digest: str, history: list[dict],
                    results: dict) -> str:
    lines = [
        f"Flight: {request.get('airline')} {request.get('flight_number')}",
        f"Route: {request.get('origin')} -> {request.get('destination')}",
        f"Stranded at: {request.get('stranded_at')}",
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
    if request.get("coverage_warning"):
        lines += ["", request["coverage_warning"]]
    if request.get("stay_conflict"):
        lines += ["", f"Timing conflict: {request['stay_conflict']}"]
    if request.get("assumptions"):
        lines += ["", "Assumptions behind this (see the system prompt for when to mention one):"]
        lines += [f"  - {a}" for a in request["assumptions"]]
    lines += ["", "Findings:", digest]
    # The guard downstream grades on these exact strings. Showing them beats making the
    # model guess, then paying to repair what it guessed wrong.
    required = _required_tokens(results)
    if required:
        lines += ["", REQUIRED_HEADING,
                  "  " + ", ".join(required),
                  "That is a checklist for you, not text for them. Never reproduce it - "
                  "not as a list, not as a run of words, not in bold. Every item belongs "
                  "inside an ordinary sentence, in the place it naturally goes."]
    return "\n".join(lines)


RIGHTS_TERMS = {
    "rebooking": ("rebook", "rerout", "replacement"),
    "refund": ("refund", "reimburse"),
    "hotel": ("hotel", "accommodation", "place to stay"),
    "meals": ("meal", "food", "refreshment", "beverage"),
    "cash_compensation": ("compensation",),
}

# (document fragment, article number) pairs that only establish who a regulation covers.
SCOPE_ONLY_ARTICLES = {("261/2004", "3")}

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def _compact(value) -> str:
    """Case/punctuation-insensitive text for deterministic presence checks."""
    return re.sub(r"[^a-z0-9€₪$]+", "", str(value or "").lower())


def _has_any(text: str, choices: tuple[str, ...]) -> bool:
    compact = _compact(text)
    return any(_compact(choice) in compact for choice in choices)


def _deadline_requirements(summary: str) -> list[tuple[str, tuple[str, ...]]]:
    """Concrete statutory deadlines that must survive prose composition."""
    found = re.findall(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b",
        summary or "", re.IGNORECASE,
    )
    requirements = []
    for value in found:
        lower = value.lower()
        digit = NUMBER_WORDS.get(lower, lower)
        aliases = (f"{lower} days", f"{digit} days")
        requirements.append((f"the {lower}-day deadline", aliases))
    return requirements


def _money_requirements(value: str) -> list[tuple[str, tuple[str, ...]]]:
    """Currency amounts, accepting either codes or familiar symbols."""
    requirements = []
    aliases = {"EUR": ("EUR", "€"), "NIS": ("NIS", "ILS", "₪"),
               "ILS": ("ILS", "NIS", "₪"), "USD": ("USD", "$")}
    for currency, amount in re.findall(
            r"\b(EUR|NIS|ILS|USD)\s*([\d,]+(?:\.\d+)?)", value or "", re.IGNORECASE):
        clean = amount.replace(",", "")
        ways = tuple(f"{mark} {clean}" for mark in aliases[currency.upper()])
        requirements.append((f"{currency.upper()} {amount}", ways))
    return requirements


def _citation_requirements(source: str) -> list[tuple[str, tuple[str, ...]]]:
    """Section/article references, with plain-language aliases.

    Keyed by the base article/section number, not the exact subsection. Two
    entitlements citing the same article's different subsections (Art. 8(1)(a)-(c)
    vs Art. 8(1)(a)) must not become two hard requirements no single plain-language
    sentence can satisfy at once - naming "Article 8" is what the compose prompt's
    own "no regulation-speak" instruction asks for, and still forces the right law
    to be named.

    Scope provisions are skipped. DocumentationAgent cites them next to the article
    that actually grants the entitlement, but they grant nothing themselves and no
    plain-language sentence about a refund names one - requiring it fails every honest
    draft and costs the passenger the whole composed answer.
    """
    requirements = []
    seen = set()
    pattern = r"\b(s\.|Art\.)\s*([0-9]+)"
    # Split first so a scope article is only skipped for the document that owns it.
    for prefix, base, segment in [
            (p, b, seg) for seg in re.split(r";", source or "")
            for p, b in re.findall(pattern, seg, re.IGNORECASE)]:
        if any(document in segment and base == article
               for document, article in SCOPE_ONLY_ARTICLES):
            continue
        key = (prefix.lower()[0], base)
        if key in seen:
            continue
        seen.add(key)
        if prefix.lower().startswith("s"):
            aliases = (f"s. {base}", f"s.{base}", f"section {base}")
            label = f"section {base} citation"
        else:
            aliases = (f"Art. {base}", f"Art.{base}", f"Article {base}")
            label = f"Article {base} citation"
        requirements.append((label, aliases))
    return requirements


def _composition_issues(text: str, request: dict, results: dict) -> list[str]:
    """Critical facts a fluent draft is not allowed to drop.

    This deliberately checks concrete identities, amounts, deadlines and source
    families rather than trying to judge natural-language legal reasoning. The latter
    remains DocumentationAgent's reflection job; this guard verifies delivery.
    """
    issues: list[str] = []

    flight_payload = results.get("flight") or {}
    for option in _options(flight_payload):
        number = option.get("flight_number")
        if number and not _has_any(text, (str(number),)):
            issues.append(f"flight option {number}")

    # There is deliberately no check that the recommended flight is *described* as the
    # recommendation. It was a window search for one of six literal phrases, and a draft
    # opening "LH 687 is the best same-day replacement." failed it - a pure false
    # positive on the only real sample we have. The dangerous case, pushing a
    # near-departure flight the passenger cannot actually board, is caught structurally
    # just below and is a misleading issue, not a wording one.
    flight_caveats = " ".join(str(c) for c in flight_payload.get("caveats") or [])
    if "urgent possibility" in flight_caveats.lower():
        lowered = text.lower()
        confirms_actionability = ("confirm" in lowered and any(
            word in lowered for word in ("seat", "rebook", "board", "gate")))
        if not confirms_actionability:
            issues.append("confirmation that the urgent flight has seats and is still boardable")
        urgent_numbers = re.findall(
            r"\b([A-Z0-9]{2}\s*\d{1,4})\s+leaves\b", flight_caveats,
            re.IGNORECASE,
        )
        for urgent in urgent_numbers:
            flight_pattern = r"\s*".join(map(re.escape, urgent.split()))
            unsafe = (
                rf"earliest\s+reasonable\s+replacement.{{0,180}}{flight_pattern}",
                rf"(?:push|recommend|choose|take|book).{{0,120}}{flight_pattern}.{{0,60}}\b(?:first|now)\b",
            )
            if any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in unsafe):
                issues.append(f"unsafe recommendation of urgent flight {urgent}")

    for option in _options(results.get("stay")):
        name = option.get("name")
        if name and not _has_any(text, (str(name),)):
            issues.append(f"stay option {name}")
        price = str(option.get("price_estimate") or "")
        for label, aliases in _money_requirements(price):
            if not _has_any(text, aliases):
                issues.append(f"{name or 'stay'} price {label}")
        # Ranges may contain the currency only once (EUR 90-150), so the second
        # bound is not returned by _money_requirements. Preserve every number.
        for amount in re.findall(r"\d[\d,]*(?:\.\d+)?", price):
            clean = amount.replace(",", "")
            if clean not in _compact(text):
                issues.append(f"{name or 'stay'} price bound {amount}")

    if _options(results.get("stay")):
        if re.search(r"\bavailability\s*:\s*20\d{2}-\d{2}-\d{2}", text,
                     re.IGNORECASE):
            issues.append("stay dates must not be labelled as room availability")
        lowered = text.lower()
        availability_honest = ("availability" in lowered and any(
            phrase in lowered for phrase in ("not confirmed", "not checked", "cannot confirm")))
        if not availability_honest:
            issues.append("room availability is not confirmed")

    rights = results.get("rights") or {}
    for item in rights.get("entitlements") or []:
        kind = str(item.get("kind") or "")
        summary = str(item.get("summary") or "")
        terms = RIGHTS_TERMS.get(kind)
        if terms and not _has_any(text, terms):
            issues.append(f"{kind} entitlement")
        if kind == "other":
            communication = tuple(word for word in
                                  ("communication", "telephone", "phone", "email", "message")
                                  if word in summary.lower())
            if communication and not _has_any(text, communication):
                issues.append("communication entitlement")
        for label, aliases in _deadline_requirements(summary) + _money_requirements(summary):
            if not _has_any(text, aliases):
                issues.append(label)

        source = str(item.get("source") or "")
        if "Aviation Services Law" in source and not _has_any(text, ("Aviation Services Law",)):
            issues.append("Aviation Services Law source")
        if re.search(r"\bEU\s*261\b", source, re.IGNORECASE) and not _has_any(
                text, ("EU 261", "EU261")):
            issues.append("EU 261 source")
        if "Conditions of Carriage" in source and not _has_any(
                text, ("Conditions of Carriage",)):
            issues.append("Conditions of Carriage source")
        if "Contract of Carriage" in source and not _has_any(
                text, ("Contract of Carriage",)):
            issues.append("Contract of Carriage source")
        for label, aliases in _citation_requirements(source):
            if not _has_any(text, aliases):
                issues.append(label)

    passenger = str(request.get("_passenger_prompt") or "").lower()
    if re.search(r"\bbags?|baggage\b", passenger):
        caveats = " ".join(str(c) for c in rights.get("caveats") or [])
        mentions_baggage = re.search(r"\b(?:bags?|baggage)\b", text, re.IGNORECASE)
        if not mentions_baggage:
            issues.append("a practical checked-baggage instruction")
        elif not re.search(
                r"\b(?:bags?|baggage)\b.{0,180}\b(?:return(?:ed)?|retrieve|collect|"
                r"transfer(?:red)?|remain checked|stay checked|handling|where)\b|"
                r"\b(?:return(?:ed)?|retrieve|collect|transfer(?:red)?|handling|where)\b"
                r".{0,180}\b(?:bags?|baggage)\b",
                text, re.IGNORECASE | re.DOTALL):
            issues.append("a practical checked-baggage instruction")
        if re.search(r"\bbags?|baggage\b", caveats, re.IGNORECASE) and not mentions_baggage:
            issues.append("the baggage evidence gap")
        if re.search(r"\b(?:bags?|baggage)\b.{0,100}\bdoes not (?:create|give|provide)\b",
                     text, re.IGNORECASE | re.DOTALL):
            issues.append("non-categorical wording for the baggage evidence gap")

    rights_caveats = " ".join(str(c) for c in rights.get("caveats") or [])
    if "applicab" in rights_caveats.lower():
        lowered = text.lower()
        # What has to survive is the concession, not the word "applicability". Requiring
        # an appl* word *and* a hedge rejected five of six plausible, correct renderings
        # ("It is not certain that European rules cover this flight"), and a free-floating
        # hedge passed on wording that conceded nothing - conv1's "if that does not clear,
        # LY 357" satisfied it by accident. Ask instead for doubt stated near the
        # regulation itself. This gates a paid repair now, so a false positive costs money.
        uncertainty = (
            "uncertain", "not certain", "unclear", "not clear", "depends", "may apply",
            "might apply", "may not", "might not", "not established", "cannot confirm",
            "can't confirm", "cannot say", "not confirmed", "unconfirmed", "probable",
            "likely", "not guaranteed", "assuming", "if these", "if those",
        )
        hedged = any(
            any(word in lowered[max(0, m.start() - 200):m.end() + 200] for word in uncertainty)
            for m in re.finditer(r"eu\s*261|european|\beu\b", lowered)
        )
        if not hedged:
            issues.append("the EU 261 applicability caveat")

    party_size = request.get("party_size")
    if isinstance(party_size, int) and party_size > 0:
        count_patterns = (
            r"\b(?:all\s+)?(\d+)\s+(?:of\s+you|guests|travell?ers|people|passengers)\b",
            r"\b(?:room|room\s+setup|setup)\s+for\s+(\d+)\b",
        )
        mentioned = [int(count) for pattern in count_patterns
                     for count in re.findall(pattern, text, re.IGNORECASE)]
        wrong = sorted({count for count in mentioned if count != party_size})
        if wrong:
            issues.append(
                f"party size is {party_size}, not {', '.join(map(str, wrong))}"
            )

    # Recommending a flight that leaves today and a room for tonight in the same answer
    # is a contradiction, and the passenger is the one who has to notice it. Checking for
    # three literal phrasings ("not taking", "if you do not take", "if you don't take")
    # rejected five of six correct renderings. What actually has to be true is structural:
    # the flight and the bed have to be reconciled in one breath, in whatever words.
    if request.get("stay_conflict"):
        match = re.search(r"\b[A-Z0-9]{2}\s*\d{1,4}\b", str(request["stay_conflict"]))
        pattern = r"\s*".join(map(re.escape, match.group(0).split())) if match else None
        # One sentence, not a proximity window. A window of 200 characters passed conv1's
        # own broken answer twice: on "stay checked through to the replacement flight"
        # (baggage, and "stay" is a verb there) and on two adjacent numbered actions,
        # "confirm LH 687 immediately" followed by "ask for hotel tonight". Sitting near
        # each other in a list is not reconciling them.
        linked = pattern and any(
            re.search(pattern, sentence, re.IGNORECASE)
            and re.search(r"\b(?:hotel|room|bed|sleep|overnight|night|accommodation)",
                          sentence, re.IGNORECASE)
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text))
        if not linked:
            issues.append("the same-day-flight assumption behind the hotel search")

    # Deliberately lenient: this asks for the uncovered airport plus any wording that
    # concedes a limit, not one fixed sentence. A guard strict enough to reject a
    # perfectly honest draft costs the passenger the whole composed answer.
    if request.get("coverage_warning"):
        uncovered = [code for code in (str(request.get("origin") or "").strip().upper(),
                                       str(request.get("destination") or "").strip().upper())
                     if code and code not in SUPPORTED_AIRPORTS]
        conceded = _has_any(text, ("outside", "do not cover", "don't cover", "not covered",
                                   "only hold", "only have", "limited to", "incomplete"))
        if not conceded or not any(_has_any(text, (code,)) for code in uncovered):
            issues.append(COVERAGE_ISSUE)

    return list(dict.fromkeys(issues))


COVERAGE_ISSUE = "the coverage limit on this route"

# Leaving a fact out and stating something untrue are not the same failure. Only these
# can send the passenger to act on a wrong belief - a room that was never held, a flight
# with no seat, a right they are told they do not have. These are the only issues whose
# survival is worth throwing away readable prose for.
MISLEADING_ISSUES = (
    "unsafe recommendation of urgent flight",
    "confirmation that the urgent flight has seats and is still boardable",
    "stay dates must not be labelled as room availability",
    "non-categorical wording for the baggage evidence gap",
    "party size is",
    # Silence about either of these reads as a complete answer, which is the wrong
    # belief exactly: a room the passenger thinks is held, and entitlements at an
    # airport whose rules we do not hold. `_coverage_warning` exists for that reason.
    "room availability is not confirmed",
    COVERAGE_ISSUE,
)


def _provenance(issue: str) -> bool:
    """Where a right came from, as opposed to what the passenger is owed."""
    return issue.endswith(" citation") or issue.endswith(" source")


def _buys_repair(issues: list[str]) -> list[str]:
    """Worth one more composing call. Any real omission qualifies.

    Provenance is not one: "section 6 citation" is a note to ourselves, and the
    required-token hint already keeps the reference in the prose where the GUI can
    link it.
    """
    return [issue for issue in issues if not _provenance(issue)]


def _must_fall_back(issues: list[str]) -> list[str]:
    """Worth throwing away the repaired prose for. Only the misleading ones.

    Split out of a single `_misleading` predicate that decided both this and whether
    to buy a repair at all. Conflating them meant a guard reading for the wrong words
    could dump a passenger to the flat digest over prose that was already correct, so
    every omission bought a footnote instead - and the footnote was the bug. Now any
    omission buys the repair, and only a surviving falsehood costs the prose.
    """
    return [issue for issue in issues if issue.startswith(MISLEADING_ISSUES)]


def _repair_prompt(original_prompt: str, draft: str, issues: list[str]) -> str:
    return "\n".join([
        "Original task and verified findings:",
        original_prompt,
        "",
        "Draft that must be revised:",
        draft,
        "",
        "Required facts missing from that draft:",
        *[f"- {issue}" for issue in issues],
    ])


def _reuse_prior_stay(request: dict, history: list[dict]) -> dict | None:
    """Reuse already-paid hotel options for a detail/price follow-up.

    Asking for more, different, closer or cheaper properties is a new search. Asking
    to see, compare or price the options already found is composition only.
    """
    prior = _prior_results(history).get("stay")
    if not prior:
        return None
    message = str(request.get("_passenger_prompt") or "")
    if re.search(r"\b(other|more|different|new|another|closer|cheaper)\b", message,
                 re.IGNORECASE):
        return None
    if re.search(r"\b(what about|suggest|show|compare|which|price|cost|details?|options?)\b",
                 message, re.IGNORECASE):
        return prior
    return None


def _compose(
    request: dict, results: dict, failures: list[str], history: list[dict]
) -> tuple[str, list[dict]]:
    """Results -> the passenger's plan, as text. Returns (text, steps)."""
    digest = _digest(results, failures)
    assumptions = request.get("assumptions") or []
    safe_digest = digest
    if assumptions:
        safe_digest = ("ASSUMPTIONS BEHIND THIS PLAN\n"
                       + "\n".join(f"  - {item}" for item in assumptions)
                       + "\n\n" + safe_digest)
    # Above everything, assumptions included: a passenger reading the flat fallback
    # must hit the limits of the answer before the answer itself.
    if request.get("stay_conflict"):
        safe_digest = request["stay_conflict"] + "\n\n" + safe_digest
    if request.get("coverage_warning"):
        safe_digest = request["coverage_warning"] + "\n\n" + safe_digest
    if re.search(r"\b(?:bags?|baggage)\b",
                 str(request.get("_passenger_prompt") or ""), re.IGNORECASE):
        safe_digest += (
            "\n\nCHECKED BAGS\n"
            "  - Ask the airline whether your bags will stay checked through to the replacement "
            "flight or be returned, and how they will be transferred if you change carriers."
        )
    prompt = _compose_prompt(request, digest, history, results)
    try:
        text, step = llm.call(
            MODULE,
            COMPOSE_SYSTEM_PROMPT,
            prompt,
            max_completion_tokens=COMPOSE_MAX_COMPLETION_TOKENS,
        )
    except llm.LLMError as exc:
        # Up to six calls and two API quotas are already spent by the time we compose.
        # Losing all of it to the last call is worse than handing over the flat version.
        return safe_digest, exc.steps

    draft = (text or "").strip()
    if not draft:
        return safe_digest, [step]
    issues = _buys_repair(_composition_issues(draft, request, results))
    if not issues:
        return draft, [step]

    # Anything the draft actually dropped buys one more composing call. This used to
    # buy a bulleted footnote of the guard's own internal labels instead - "the EU 261
    # applicability caveat", "clear recommendation of LH 687" - stapled under a rule and
    # a heading. The facts behind those labels are real and belong in the passenger's
    # answer in plain language, which is what the repair is for. Bounded at one extra
    # call, not an open-ended reflection loop.
    try:
        repaired, repair_step = llm.call(
            MODULE,
            COMPOSE_REPAIR_SYSTEM_PROMPT,
            _repair_prompt(prompt, draft, issues),
            max_completion_tokens=COMPOSE_REPAIR_MAX_COMPLETION_TOKENS,
        )
    except llm.LLMError as exc:
        return safe_digest, [step] + exc.steps

    repaired = (repaired or "").strip()
    if repaired:
        # Only a surviving falsehood is worth the prose. An omission that outlived the
        # repair usually means the guard is reading for words the composer had no reason
        # to choose, and trading a readable answer for a flat digest over that costs the
        # passenger far more than the missing line does.
        if not _must_fall_back(_composition_issues(repaired, request, results)):
            return repaired, [step, repair_step]
    # Two calls in and it still says something untrue. The flat digest is far less
    # elegant, but it is grounded, and a misleading answer is worse than an ugly one.
    return safe_digest, [step, repair_step]


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
            [_conflict_question(c) for c in blocking]
            + [_missing_question(f, request) for f in request["missing"]]
        ))
        one = len(asked) == 1
        opener = ("Before I can help, I need to check one thing:" if blocking and one
                  else "Before I can help, I need to check a couple of things:" if blocking
                  else "Before I can help, I need one more detail:" if one
                  else "Before I can help, I need a couple of details:")
        return opener + "\n" + "\n".join(f"  - {q}" for q in asked), steps, {"_pending_needs": needs}

    results: dict = {}
    failures: list[str] = []

    # A price/detail follow-up should use the properties already stored on the turn,
    # not spend another AccommodationAgent call to rediscover the same OSM rows.
    if "stay" in needs:
        prior_stay = _reuse_prior_stay(request, history)
        if prior_stay:
            results["stay"] = prior_stay
            needs = [need for need in needs if need != "stay"]

    # One agent failing must not cost the passenger the rest of the plan.
    def dispatch(key: str, fn, *args):
        try:
            payload, agent_steps = fn(*args, history)
            results[key] = payload
            if key == "rights":
                citations = extract_citations(payload, agent_steps)
                if citations:
                    results["citations"] = citations
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
        # An explicit request outranks the calendar-date inference on both first
        # turns and narrowed follow-ups. A follow-up typo can still be understood by
        # refinement as needs=["stay"], so keep that semantic signal too.
        explicit_stay = _explicit_stay_requested(request) or needs == ["stay"]
        if not stay_window and explicit_stay:
            stay_window = _tonight_window(request, known_flight)
            # Its own field, not one more line in `assumptions`. A generic assumption gets
            # one soft mention in the system prompt and the composer duly dropped this one;
            # `coverage_warning`, the same class of must-say fact, gets a named prompt line,
            # a dedicated paragraph and a place above the fallback digest, and survives.
            request["stay_conflict"] = _same_day_flight_assumption(request, known_flight)
        if stay_window:
            dispatch("stay", accommodation_agent.run, request, stay_window)

    if "rights" in needs:
        # Only when entitlements are on the table: a flight-only follow-up does not
        # need a legal disclaimer attached to it.
        warning = _coverage_warning(request)
        if warning:
            request["coverage_warning"] = warning
        dispatch("rights", documentation_agent.run, request)

    text, compose_steps = _compose(request, results, failures, history)
    steps.extend(compose_steps)
    return text, steps, results
