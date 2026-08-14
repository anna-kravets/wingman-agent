"""A stand-in for LLMod so agent and Supervisor tests run with no API key.

Deliberately not autouse: tests/test_llm.py exercises lib.llm.call itself and
must see the real thing. Person C will want this too when DocumentationAgent
stops being a stub.
"""

from datetime import datetime, timedelta

import pytest

from lib import llm
from lib.steps import make_step
from lib.tools import flights, hotels


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


RESPONSES = {
    "FlightAgent": _flight_response,
    "AccommodationAgent": _accommodation_response,
}


@pytest.fixture
def fake_llm(monkeypatch):
    def call(module, system_prompt, user_prompt, *, expect_json=False):
        payload = RESPONSES[module](user_prompt)
        return payload, make_step(module, system_prompt, user_prompt, payload)

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
    }])
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [{
        "name": "Airport Plaza", "distance_km": 2.4, "stars": "4",
        "breakfast": None, "area": "Lod",
    }])
