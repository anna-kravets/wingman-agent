"""A stand-in for LLMod so agent and Supervisor tests run with no API key.

Deliberately not autouse: tests/test_llm.py exercises lib.llm.call itself and
must see the real thing. Person C will want this too when DocumentationAgent
stops being a stub.
"""

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
