"""FlightAgent: grounding, validation and the trace it leaves behind."""

import pytest

from lib import llm
from lib.agents import flight_agent
from lib.steps import make_step
from lib.tools import flights

REQUEST = {
    "airline": "LH", "flight_number": "LH318", "origin": "TLV", "destination": "FRA",
    "disruption": "cancelled", "stranded_at": "TLV", "party_size": 2,
    "arrive_by": None, "needs": ["flight"],
    "local_now": "2026-08-09T22:15:00",
}

CANDIDATE = {
    "flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH",
    "origin": "TLV", "destination": "FRA",
    "depart": "2026-08-10T16:30+03:00", "arrive": "2026-08-10T20:10+02:00",
    "status": "Expected", "aircraft": "Airbus A320", "terminal": "3",
}


def fake_call(payload):
    def call(module, system_prompt, user_prompt, *, expect_json=False):
        return payload, make_step(module, system_prompt, user_prompt, payload)
    return call


def good_payload(depart="2026-08-10T16:30+03:00", arrive="2026-08-10T20:10+02:00",
                 recommended="F1"):
    return {
        "options": [{
            "id": "F1", "airline": "Lufthansa", "flight_number": "LH 687",
            "origin": "TLV", "destination": "FRA", "depart": depart, "arrive": arrive,
            "stops": 0, "fare_conditions": "See your Contract of Carriage.",
            "notes": "Nonstop.",
        }],
        "recommended_id": recommended,
    }


def test_real_candidates_reach_the_prompt(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [])

    user_prompt = steps[0]["prompt"]["user_prompt"]
    assert "LH 687" in user_prompt
    assert "2026-08-10T16:30+03:00" in user_prompt


def test_one_llm_call_produces_exactly_one_step(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [])

    assert len(steps) == 1
    assert steps[0]["module"] == "FlightAgent"


def test_degraded_mode_tells_the_model_to_label_its_output(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [])

    assert "Illustrative" in steps[0]["prompt"]["user_prompt"]


def test_options_with_unparseable_times_are_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({
        "id": "F2", "airline": "El Al", "flight_number": "LY 357",
        "origin": "TLV", "destination": "FRA",
        "depart": "tomorrow morning", "arrive": "later", "stops": 0,
        "fare_conditions": "", "notes": "",
    })
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    result, _ = flight_agent.run(REQUEST, [])

    # The Supervisor calls fromisoformat on depart; a bad one costs the hotel too.
    assert [o["id"] for o in result["options"]] == ["F1"]


def test_a_dangling_recommended_id_falls_back_to_the_first_option(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(recommended="F9")))

    result, _ = flight_agent.run(REQUEST, [])

    assert result["recommended_id"] == "F1"


def test_no_usable_option_raises_but_keeps_the_step(monkeypatch):
    payload = {"options": [{"id": "F1", "depart": "soon", "arrive": "later"}],
               "recommended_id": "F1"}
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    with pytest.raises(llm.LLMError) as caught:
        flight_agent.run(REQUEST, [])

    # The spec wants every call that happened in the trace, failures included.
    assert len(caught.value.steps) == 1
    assert caught.value.steps[0]["module"] == "FlightAgent"


def test_history_reaches_the_prompt(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [{"prompt": "what about earlier?",
                                           "response": "checking"}])

    assert "what about earlier?" in steps[0]["prompt"]["user_prompt"]


def test_the_stub_flag_is_gone():
    assert not hasattr(flight_agent, "IS_STUB")
