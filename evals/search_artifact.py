"""Mechanical checks over a saved live-run artifact.

Every check reads only the artifact, so it runs offline with no keys. The
candidates the agent was given are inside each step's user_prompt, which is
what makes grounding verifiable after the fact.

Checks assert *properties*, never specific flight numbers or counts: real
schedules change daily and a test pinned to today's timetable is a test that
fails tomorrow for no reason.
"""

from __future__ import annotations

import json
import re

MODULES = {"Supervisor", "FlightAgent", "AccommodationAgent", "DocumentationAgent"}

FLIGHT_MARKER = "Candidates (real, verified departures"
HOTEL_MARKER = "Hotels (real, near the airport"

BOOKING_SITES = (
    "booking.com", "expedia", "kayak", "skyscanner", "agoda", "hotels.com",
    "trivago", "momondo", "priceline", "orbitz", "travelocity", "airbnb",
    "tripadvisor", "kiwi.com", "opodo", "edreams",
)

HEDGES = ("estimate", "approx", "around", "roughly", "~", "about", "typically")
MONEY = re.compile(r"(?:eur|usd|gbp|ils|nis|\$|€|£|₪)\s?\d|\d+\s?(?:eur|usd|gbp|ils|nis)", re.I)
BAGGAGE_ASSERTION = re.compile(r"\d+\s?kg|\ballowance is\b|\byou may (?:bring|carry)\b", re.I)
CONTRACT_POINTER = re.compile(r"contract of carriage|conditions of carriage|entitlement", re.I)


def candidates_from_prompt(user_prompt: str) -> list[dict]:
    """The candidate list the agent was actually given, or [] in degraded mode."""
    for marker in (FLIGHT_MARKER, HOTEL_MARKER):
        index = user_prompt.find(marker)
        if index == -1:
            continue
        start = user_prompt.find("[", index)
        if start == -1:
            continue
        depth, in_string, escaped = 0, False, False
        for position in range(start, len(user_prompt)):
            character = user_prompt[position]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
            elif character == '"':
                in_string = not in_string
            elif not in_string and character == "[":
                depth += 1
            elif not in_string and character == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(user_prompt[start:position + 1])
                    except json.JSONDecodeError:
                        return []
    return []


def _step(artifact: dict, module: str) -> dict | None:
    return next((s for s in artifact.get("steps", []) if s["module"] == module), None)


def _options(step: dict | None) -> list[dict]:
    if not step or not isinstance(step.get("response"), dict):
        return []
    return step["response"].get("options") or []


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).upper()


def _result(check: str, ok: bool, detail: str, skip: bool = False) -> dict:
    return {"check": check, "status": "skip" if skip else ("pass" if ok else "fail"),
            "detail": detail}


def _check_trace_shape(artifact, case):
    steps = artifact.get("steps", [])
    problems = []
    for step in steps:
        if step["module"] not in MODULES:
            problems.append(f"unknown module {step['module']!r}")
        if set(step) != {"module", "prompt", "response"}:
            problems.append(f"{step['module']} has keys {sorted(step)}")
        if set(step.get("prompt", {})) != {"system_prompt", "user_prompt"}:
            problems.append(f"{step['module']} prompt keys {sorted(step.get('prompt', {}))}")
    for module in ("FlightAgent", "AccommodationAgent"):
        count = sum(1 for s in steps if s["module"] == module)
        if count > 1:
            problems.append(f"{module} produced {count} steps, expected at most 1")
    return _result("trace_shape", not problems,
                   "; ".join(problems) or f"{len(steps)} steps, all well-formed")


def _grounding(artifact, module, marker_key, option_key):
    step = _step(artifact, module)
    if not step:
        return _result(f"grounding_{marker_key}", True, f"{module} was not dispatched", skip=True)
    candidates = candidates_from_prompt(step["prompt"]["user_prompt"])
    if not candidates:
        return _result(f"grounding_{marker_key}", True,
                       "degraded run, nothing to ground against", skip=True)

    allowed = {_normalise(c.get("flight") or c.get("name")) for c in candidates}
    invented = [o.get(option_key) for o in _options(step)
                if _normalise(o.get(option_key)) not in allowed]
    return _result(
        f"grounding_{marker_key}", not invented,
        f"invented: {invented}" if invented
        else f"all {len(_options(step))} from {len(candidates)} candidates",
    )


def _check_grounding_flights(artifact, case):
    return _grounding(artifact, "FlightAgent", "flights", "flight_number")


def _check_grounding_hotels(artifact, case):
    return _grounding(artifact, "AccommodationAgent", "hotels", "name")


def _check_date_sync(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    if not step:
        return _result("date_sync", True, "no stay was needed", skip=True)
    prompt = step["prompt"]["user_prompt"]

    def field(label):
        match = re.search(rf"^{label}:\s*(.+)$", prompt, re.M)
        return match.group(1).strip() if match else None

    wanted_in, wanted_out, wanted_nights = field("Check in"), field("Check out"), field("Nights")
    problems = []
    for option in _options(step):
        if option.get("check_in") != wanted_in or option.get("check_out") != wanted_out:
            problems.append(
                f"{option.get('id')}: {option.get('check_in')}..{option.get('check_out')}")
        if str(option.get("nights")) != str(wanted_nights):
            problems.append(f"{option.get('id')}: nights {option.get('nights')} != {wanted_nights}")
    return _result("date_sync", not problems,
                   "; ".join(problems) or f"all options on {wanted_in}..{wanted_out} ({wanted_nights}n)")


def _check_no_accommodation(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    return _result("no_accommodation", step is None,
                   "AccommodationAgent ran when no stay was needed" if step else "correctly skipped")


def _check_no_booking_site(artifact, case):
    haystack = json.dumps(artifact.get("steps", []), ensure_ascii=False).lower()
    haystack += str(artifact.get("response", "")).lower()
    named = [site for site in BOOKING_SITES if site in haystack]
    return _result("no_booking_site", not named, f"named: {named}" if named else "none named")


def _check_price_honesty(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    if not step:
        return _result("price_honesty", True, "no stay was proposed", skip=True)
    blunt = [o.get("price_estimate") for o in _options(step)
             if not any(h in str(o.get("price_estimate", "")).lower() for h in HEDGES)]
    return _result("price_honesty", not blunt,
                   f"unhedged: {blunt}" if blunt else "every price is hedged")


def _check_no_asserted_fare(artifact, case):
    step = _step(artifact, "FlightAgent")
    if not step:
        return _result("no_asserted_fare", True, "FlightAgent was not dispatched", skip=True)
    # FlightAgent has no fare data at all, so any money figure it prints is invented.
    quoting = [f"{o.get('id')}: {o.get('fare_conditions')}" for o in _options(step)
               if MONEY.search(str(o.get("fare_conditions", "")) + str(o.get("notes", "")))]
    return _result("no_asserted_fare", not quoting,
                   f"quoted money: {quoting}" if quoting else "no fares asserted")


def _check_meals_honesty(artifact, case):
    step = _step(artifact, "AccommodationAgent")
    if not step:
        return _result("meals_honesty", True, "no stay was proposed", skip=True)
    candidates = {_normalise(c.get("name")): c
                  for c in candidates_from_prompt(step["prompt"]["user_prompt"])}
    problems = []
    for option in _options(step):
        candidate = candidates.get(_normalise(option.get("name")))
        if option.get("meals_included"):
            if not (candidate and candidate.get("breakfast")):
                problems.append(f"{option.get('id')} claims meals with no supporting tag")
        elif "confirm" not in str(option.get("notes", "")).lower():
            problems.append(f"{option.get('id')} does not say meals were unconfirmed")
    return _result("meals_honesty", not problems, "; ".join(problems) or "meals stated honestly")


def _check_degraded_labelling(artifact, case):
    problems = []
    checked = 0
    for module in ("FlightAgent", "AccommodationAgent"):
        step = _step(artifact, module)
        if not step or candidates_from_prompt(step["prompt"]["user_prompt"]):
            continue
        for option in _options(step):
            checked += 1
            if "illustrative" not in str(option.get("notes", "")).lower():
                problems.append(f"{module}/{option.get('id')} is unlabelled")
    if not checked:
        return _result("degraded_labelling", True, "nothing ran degraded", skip=True)
    return _result("degraded_labelling", not problems,
                   "; ".join(problems) or f"all {checked} options labelled illustrative")


def _check_deferral(artifact, case):
    step = _step(artifact, "FlightAgent")
    if not step:
        return _result("deferral", True, "FlightAgent was not dispatched", skip=True)
    text = json.dumps(_options(step), ensure_ascii=False)
    asserted = BAGGAGE_ASSERTION.search(text)
    if asserted:
        return _result("deferral", False, f"asserted a carriage rule: {asserted.group(0)!r}")
    points = bool(CONTRACT_POINTER.search(text))
    return _result("deferral", points,
                   "points at the Contract of Carriage" if points
                   else "neither asserted nor deferred - no pointer to the contract")


def _check_history_reached_agents(artifact, case):
    if not case.get("history"):
        return _result("history_reached_agents", True, "case has no history", skip=True)
    missing = [s["module"] for s in artifact.get("steps", [])
               if s["module"] in ("FlightAgent", "AccommodationAgent")
               and "Earlier in this conversation" not in s["prompt"]["user_prompt"]]
    return _result("history_reached_agents", not missing,
                   f"history missing from: {missing}" if missing
                   else "history reached every agent")


CHECKS = {
    "trace_shape": _check_trace_shape,
    "grounding_flights": _check_grounding_flights,
    "grounding_hotels": _check_grounding_hotels,
    "date_sync": _check_date_sync,
    "no_accommodation": _check_no_accommodation,
    "no_booking_site": _check_no_booking_site,
    "price_honesty": _check_price_honesty,
    "no_asserted_fare": _check_no_asserted_fare,
    "meals_honesty": _check_meals_honesty,
    "degraded_labelling": _check_degraded_labelling,
    "deferral": _check_deferral,
    "history_reached_agents": _check_history_reached_agents,
}


def evaluate(artifact: dict, case: dict) -> list[dict]:
    """Run the checks this case declares. Returns one result per check."""
    return [CHECKS[name](artifact, case) for name in case.get("checks", [])]
