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
from lib.tools import flights, hotels


def _flight_response(user_prompt: str) -> dict:
    """Only the model's half: the agent fills the facts from the candidate.

    The flight numbers must match fake_search_data's candidates, which is why this
    fake got shorter rather than longer when the payload grew.
    """
    return {
        "options": [
            {"id": "F1", "flight_number": "LH 687",
             "rebooking": "Rebooking terms come from your Contract of Carriage.",
             "notes": "Earliest nonstop."},
            {"id": "F2", "flight_number": "LY 357",
             "rebooking": "A different airline from the one that cancelled.",
             "notes": "Leaves earlier, lands earlier."},
        ],
        "recommended_id": "F1",
        "caveats": [],
    }


def _accommodation_response(user_prompt: str) -> dict:
    return {
        "options": [{
            "id": "H1", "name": "Airport Plaza",
            "price_estimate": "EUR 100-140 total (estimate)",
            "notes": "Meals not confirmed — check at the desk.",
        }],
        "recommended_id": "H1",
        "caveats": [],
    }


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
    message = user_prompt.rpartition("Passenger's message:")[2].strip().lower()
    disruption = _disruption_in(user_prompt.lower())
    # A message that reports a disruption is a fresh request even mid-conversation;
    # only a question about a plan already on the table narrows the crew.
    follow_up = "Earlier in this conversation:" in user_prompt and not _disruption_in(message)

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
        "incident_time": None,
        "conflicts": [],
        "needs": needs,
    }


def _compose_response(user_prompt: str) -> str:
    """The fake composer hands the findings straight back. The real model rewrites them
    as prose for one tired person; the tests assert on the facts, not the prose."""
    return user_prompt.rpartition("Findings:")[2].strip()


def _supervisor_response(user_prompt: str):
    """Both Supervisor calls arrive here: the composing one is the one carrying results."""
    if "Findings:" in user_prompt:
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


@pytest.fixture
def fake_search_data(monkeypatch):
    """Stand-in candidates so orchestration tests reach the agents at all.

    Both search agents refuse outright when their tool returns nothing (the
    10/8/2026 reversal: no verified data means no invented options, and no LLM
    call). `conftest.py` forces WINGMAN_LIVE_DATA=0, so without this the crew
    would never get past the refusal and Supervisor tests would test nothing.
    """
    depart = (datetime.now() + timedelta(days=1)).replace(
        hour=9, minute=40, second=0, microsecond=0)

    monkeypatch.setattr(flights, "search", lambda *a, **k: [{
        "flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH",
        "origin": "TLV", "destination": "FRA",
        "depart": depart.isoformat(),
        "arrive": (depart + timedelta(hours=3, minutes=25)).isoformat(),
        "status": "Expected", "aircraft": "Airbus A320", "terminal": "3",
    }, {
        "flight": "LY 357", "airline": "El Al", "airline_iata": "LY",
        "origin": "TLV", "destination": "FRA",
        "depart": (depart - timedelta(hours=3)).isoformat(),
        "arrive": (depart + timedelta(minutes=25)).isoformat(),
        "status": "Expected", "aircraft": "Boeing 737-900", "terminal": "3",
    }])
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [{
        "name": "Airport Plaza", "distance_km": 2.4, "stars": "4",
        "breakfast": None, "area": "Lod", "phone": "+972 3 000 0000",
        "address": "12 HaNasi", "wheelchair": "yes",
    }])
