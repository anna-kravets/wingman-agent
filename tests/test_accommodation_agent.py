"""AccommodationAgent: grounding, the fixed stay window, and honest meals."""

import pytest

from lib import llm
from lib.agents import accommodation_agent
from lib.steps import make_step
from lib.tools import hotels

REQUEST = {"stranded_at": "TLV", "party_size": 2, "local_now": "2026-08-09T22:15:00"}
WINDOW = {"check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1,
          "guests": 2, "departs": "2026-08-10T09:40:00"}
CANDIDATE = {"name": "Airport Plaza", "distance_km": 2.4, "stars": "4",
             "breakfast": None, "area": "Lod"}


def fake_call(payload):
    def call(module, system_prompt, user_prompt, *, expect_json=False):
        return payload, make_step(module, system_prompt, user_prompt, payload)
    return call


def good_payload(check_in="2026-08-09", check_out="2026-08-10", recommended="H1"):
    return {
        "options": [{
            "id": "H1", "name": "Airport Plaza", "area": "2.4 km from the terminal",
            "check_in": check_in, "check_out": check_out, "nights": 1,
            "price_estimate": "EUR 120 total (estimate)", "meals_included": False,
            "notes": "Meals not confirmed - check at the desk.",
        }],
        "recommended_id": recommended,
    }


def test_real_hotels_reach_the_prompt(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])

    assert "Airport Plaza" in steps[0]["prompt"]["user_prompt"]


def test_the_stay_window_is_stated_as_non_negotiable(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])
    user_prompt = steps[0]["prompt"]["user_prompt"]

    assert "Check in: 2026-08-09" in user_prompt
    assert "Check out: 2026-08-10" in user_prompt
    assert "Nights: 1" in user_prompt


def test_one_llm_call_produces_exactly_one_step(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])

    assert len(steps) == 1
    assert steps[0]["module"] == "AccommodationAgent"


def test_degraded_mode_labels_its_output(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])

    assert "Illustrative" in steps[0]["prompt"]["user_prompt"]


def test_options_on_the_wrong_dates_are_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({
        "id": "H2", "name": "Wrong Nights Inn", "area": "town",
        "check_in": "2026-08-12", "check_out": "2026-08-13", "nights": 1,
        "price_estimate": "EUR 60", "meals_included": False, "notes": "",
    })
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    result, _ = accommodation_agent.run(REQUEST, WINDOW, [])

    # The nights come from the flight that was found; they are not negotiable.
    assert [o["id"] for o in result["options"]] == ["H1"]


def test_unparseable_dates_are_dropped(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(check_in="tonight")))

    with pytest.raises(llm.LLMError) as caught:
        accommodation_agent.run(REQUEST, WINDOW, [])

    assert len(caught.value.steps) == 1


def test_a_dangling_recommended_id_falls_back(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(recommended="H9")))

    result, _ = accommodation_agent.run(REQUEST, WINDOW, [])

    assert result["recommended_id"] == "H1"


def test_the_stub_flag_is_gone():
    assert not hasattr(accommodation_agent, "IS_STUB")
