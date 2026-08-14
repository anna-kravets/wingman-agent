"""A stand-in for LLMod so agent and Supervisor tests run with no API key.

Deliberately not autouse: tests/test_llm.py exercises lib.llm.call itself and
must see the real thing. Person C will want this too when DocumentationAgent
stops being a stub.
"""

import re
from datetime import datetime, timedelta

import pytest

from lib import llm
from lib.steps import make_step


def _flight_response(user_prompt: str) -> dict:
    depart = (datetime.now() + timedelta(days=1)).replace(
        hour=9, minute=40, second=0, microsecond=0)
    return {
        "options": [{
            "id": "F1", "airline": "Lufthansa", "flight_number": "LH 687",
            "origin": "TLV", "destination": "FRA",
            "depart": depart.isoformat(),
            "arrive": (depart + timedelta(hours=3, minutes=25)).isoformat(),
            "stops": 0,
            "fare_conditions": "Rebooking terms come from your Contract of Carriage.",
            "notes": "Earliest nonstop.",
        }],
        "recommended_id": "F1",
    }


def _accommodation_response(user_prompt: str) -> dict:
    fields = {}
    for line in user_prompt.splitlines():
        if ":" in line:
            label, _, value = line.partition(":")
            fields[label.strip()] = value.strip()
    return {
        "options": [{
            "id": "H1", "name": "Airport Plaza", "area": "8 minutes from the terminal",
            "check_in": fields.get("Check in", "2026-08-09"),
            "check_out": fields.get("Check out", "2026-08-10"),
            "nights": int(fields.get("Nights", "1") or 1),
            "price_estimate": "EUR 120 total (estimate)",
            "meals_included": False,
            "notes": "Meals not confirmed — check at the desk.",
        }],
        "recommended_id": "H1",
    }


# The phrase the Supervisor's refinement gate opens with, used here to spot a message
# that is answering it.
GATE_MARKER = "I need a couple of details"

# What a follow-up question is about, as far as the fake is concerned. The real model
# is told to work this out from the message (supervisor.REFINE_SYSTEM_PROMPT); the fake
# keyword-matches so the narrowing is testable without a key.
FOLLOW_UP_WORDS = {
    "flight": ("flight", "earlier", "later", "depart", "seat", "connection"),
    "stay": ("hotel", "sleep", "room", "bed"),
    "rights": ("owed", "compensation", "rights", "claim", "baggage", "meal", "refund"),
}


def _disruption_in(text: str) -> str | None:
    if "denied boarding" in text or "bumped" in text:
        return "denied_boarding"
    if "cancel" in text:
        return "cancelled"
    if "delay" in text:
        return "delayed"
    return None


def _refine_response(user_prompt: str) -> dict:
    """Stand-in for the Supervisor's question-refinement call.

    Extracts from the whole prompt, earlier turns included, so a follow-up inherits
    the details the passenger gave once — which is what the real prompt asks for.
    """
    earlier, _, message = user_prompt.rpartition("Passenger's message:")
    message = message.strip().lower()
    disruption = _disruption_in(user_prompt.lower())
    # Someone answering the refinement gate is completing their first request, not
    # asking a narrower question — the real prompt carries the same exception.
    answering_the_gate = GATE_MARKER in earlier.rsplit("  you:", 1)[-1]
    # A message that reports a disruption is a fresh request even mid-conversation;
    # only a question about a plan already on the table narrows the crew.
    follow_up = (
        "Earlier in this conversation:" in user_prompt
        and not _disruption_in(message)
        and not answering_the_gate
    )

    flight = re.search(r"\b([A-Z]{2})\s?(\d{2,4})\b", user_prompt)
    route = re.search(r"\b([A-Z]{3})\s*(?:->|→|to)\s*([A-Z]{3})\b", user_prompt)

    needs = list(FOLLOW_UP_WORDS)
    if follow_up:
        needs = [
            need for need, words in FOLLOW_UP_WORDS.items()
            if any(word in message for word in words)
        ]

    return {
        "airline": flight.group(1) if flight else None,
        "flight_number": flight.group(0).replace(" ", "") if flight else None,
        "origin": route.group(1) if route else None,
        "destination": route.group(2) if route else None,
        "disruption": disruption,
        "stranded_at": route.group(1) if route else None,
        "party_size": 1,
        "arrive_by": None,
        "needs": needs,
    }


def _compose_response(user_prompt: str) -> str:
    """The fake composer hands the crew's results straight back. The real model rewrites
    them as prose for one tired person; the tests assert on the facts, not the prose."""
    return user_prompt.rpartition("What the crew came back with:")[2].strip()


def _supervisor_response(user_prompt: str):
    """Both Supervisor calls arrive here: the composing one is the one carrying results."""
    if "What the crew came back with:" in user_prompt:
        return _compose_response(user_prompt)
    return _refine_response(user_prompt)


RESPONSES = {
    "Supervisor": _supervisor_response,
    "FlightAgent": _flight_response,
    "AccommodationAgent": _accommodation_response,
}


@pytest.fixture
def fake_llm(monkeypatch):
    def call(module, system_prompt, user_prompt, *, expect_json=False,
             max_completion_tokens=None):
        payload = RESPONSES[module](user_prompt)
        # Mirror lib.llm.call: a text call's step wraps the string, a JSON one does not.
        response = payload if expect_json else {"text": payload}
        return payload, make_step(module, system_prompt, user_prompt, response)

    monkeypatch.setattr(llm, "call", call)
    return call
