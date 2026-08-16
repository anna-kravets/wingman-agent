"""FlightAgent: grounding, enrichment from the candidates, caveats, and the trace."""

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
    def call(module, system_prompt, user_prompt, *, expect_json=False, **kwargs):
        return payload, make_step(module, system_prompt, user_prompt, payload)
    return call


def good_payload(flight_number="LH 687", recommended="F1"):
    """What the model now returns: a choice and prose, not a transcription."""
    return {
        "options": [{
            "id": "F1",
            "flight_number": flight_number,
            "rebooking": "Same airline as the cancelled flight.",
            "notes": "Earliest arrival.",
        }],
        "recommended_id": recommended,
    }


def run_with(monkeypatch, candidates, payload):
    monkeypatch.setattr(flights, "search", lambda *a, **k: candidates)
    monkeypatch.setattr(llm, "call", fake_call(payload))
    return flight_agent.run(REQUEST, [])


# --- grounding and the trace ----------------------------------------------------


def test_real_candidates_reach_the_prompt(monkeypatch):
    _, steps = run_with(monkeypatch, [CANDIDATE], good_payload())

    user_prompt = steps[0]["prompt"]["user_prompt"]
    assert "LH 687" in user_prompt
    assert "2026-08-10T16:30+03:00" in user_prompt


def test_one_llm_call_produces_exactly_one_step(monkeypatch):
    _, steps = run_with(monkeypatch, [CANDIDATE], good_payload())

    assert len(steps) == 1
    assert steps[0]["module"] == "FlightAgent"


def test_flight_completion_guard_uses_the_project_10k_ceiling(monkeypatch):
    seen = []
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])

    def capture(module, system_prompt, user_prompt, **kwargs):
        seen.append(kwargs.get("max_completion_tokens"))
        payload = good_payload()
        return payload, make_step(module, system_prompt, user_prompt, payload)

    monkeypatch.setattr(llm, "call", capture)
    flight_agent.run(REQUEST, [])

    assert seen == [10_000]


def test_history_reaches_the_prompt(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [{"prompt": "what about earlier?",
                                           "response": "checking"}])

    assert "what about earlier?" in steps[0]["prompt"]["user_prompt"]


# --- facts come from the candidate, not the model -------------------------------


def test_facts_are_taken_from_the_candidate_not_the_model(monkeypatch):
    lying = {"options": [{
        "id": "F1", "flight_number": "LH 687",
        "depart": "2026-01-01T00:00:00", "arrive": "2026-01-01T01:00:00",
        "airline": "Wrong Airways", "origin": "XXX", "destination": "YYY",
        "rebooking": "", "notes": "",
    }], "recommended_id": "F1"}

    option = run_with(monkeypatch, [CANDIDATE], lying)[0]["options"][0]

    # The model cannot corrupt a fact it is no longer asked to copy.
    assert option["depart"] == CANDIDATE["depart"]
    assert option["airline"] == "Lufthansa"
    assert option["origin"] == "TLV" and option["destination"] == "FRA"


def test_fields_the_tool_fetched_are_surfaced(monkeypatch):
    option = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["options"][0]

    assert option["terminal"] == "3"
    assert option["aircraft"] == "Airbus A320"
    assert option["status"] == "Expected"
    assert option["airline_iata"] == "LH"


def test_duration_and_next_day_are_computed(monkeypatch):
    option = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["options"][0]

    # 16:30+03:00 -> 20:10+02:00 is 4h40m of actual travel, same calendar day.
    assert option["duration_minutes"] == 280
    assert option["arrives_next_day"] is False


def test_an_overnight_flight_is_flagged(monkeypatch):
    overnight = dict(CANDIDATE, flight="LH 999",
                     depart="2026-08-10T23:30+03:00", arrive="2026-08-11T03:10+02:00")

    option = run_with(monkeypatch, [overnight],
                      good_payload(flight_number="LH 999"))[0]["options"][0]

    assert option["arrives_next_day"] is True


def test_an_option_matching_no_candidate_is_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({"id": "F2", "flight_number": "XX 999",
                               "rebooking": "", "notes": ""})

    result, _ = run_with(monkeypatch, [CANDIDATE], payload)

    # Grounding is now structural: an invented flight matches nothing and cannot survive.
    assert [o["id"] for o in result["options"]] == ["F1"]


def test_flight_numbers_match_regardless_of_spacing(monkeypatch):
    result, _ = run_with(monkeypatch, [CANDIDATE], good_payload(flight_number="lh687"))

    assert result["options"][0]["flight_number"] == "LH 687"


def test_no_usable_option_raises_but_keeps_the_step(monkeypatch):
    invented = {"options": [{"id": "F1", "flight_number": "XX 999"}], "recommended_id": "F1"}

    with pytest.raises(llm.LLMError) as caught:
        run_with(monkeypatch, [CANDIDATE], invented)

    # The spec wants every call that happened in the trace, failures included.
    assert len(caught.value.steps) == 1
    assert caught.value.steps[0]["module"] == "FlightAgent"


def test_a_dangling_recommended_id_falls_back_to_the_first_option(monkeypatch):
    result, _ = run_with(monkeypatch, [CANDIDATE], good_payload(recommended="F9"))

    assert result["recommended_id"] == "F1"


def test_stops_and_fare_conditions_are_gone(monkeypatch):
    option = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["options"][0]

    assert "stops" not in option          # always 0 since direct-only is a decision
    assert "fare_conditions" not in option
    assert option["rebooking"] == "Same airline as the cancelled flight."


# --- caveats --------------------------------------------------------------------


def test_a_thin_result_is_flagged(monkeypatch):
    caveats = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["caveats"]

    assert any(c.startswith("NOTE:") and "1 flight" in c for c in caveats)


def test_a_different_carrier_is_flagged(monkeypatch):
    # REQUEST's disrupted flight is LH and the candidate is LH, so nothing to say.
    caveats = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["caveats"]
    assert not any("different airline" in c for c in caveats)

    other = dict(CANDIDATE, airline_iata="LY", airline="El Al", flight="LY 1")
    caveats = run_with(monkeypatch, [other],
                       good_payload(flight_number="LY 1"))[0]["caveats"]
    assert any("different airline" in c for c in caveats)


def test_airline_name_is_matched_to_the_cancelled_flights_iata_code(monkeypatch):
    """The live intake returns "Lufthansa", while candidates carry "LH".

    Comparing those strings directly produced the false claim that every option was
    on a different airline even when LH 687 was one of them.
    """
    request = dict(REQUEST, airline="Lufthansa")
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    caveats = flight_agent.run(request, [])[0]["caveats"]

    assert not any("different airline" in c for c in caveats)


def test_a_departure_the_passenger_may_miss_asks_for_confirmation(monkeypatch):
    # local_now is 22:15; a 23:00 departure leaves 45 minutes.
    soon = dict(CANDIDATE, flight="LH 1",
                depart="2026-08-09T23:00:00+03:00", arrive="2026-08-10T02:00:00+02:00")

    caveats = run_with(monkeypatch, [soon],
                       good_payload(flight_number="LH 1"))[0]["caveats"]

    assert any(c.startswith("CONFIRM:") and "minutes" in c for c in caveats)


def test_a_near_departure_is_not_recommended_over_a_realistic_later_option(monkeypatch):
    """A schedule in 30 minutes is a possibility, not a safe recovery plan."""
    soon = dict(CANDIDATE, flight="LH 1",
                depart="2026-08-09T22:45:00+03:00",
                arrive="2026-08-10T01:45:00+02:00")
    later = dict(CANDIDATE, flight="LH 2",
                 depart="2026-08-10T06:00:00+03:00",
                 arrive="2026-08-10T09:00:00+02:00")
    payload = {
        "options": [
            {"id": "F1", "flight_number": "LH 1", "rebooking": "", "notes": "Earliest."},
            {"id": "F2", "flight_number": "LH 2", "rebooking": "", "notes": "Later."},
        ],
        "recommended_id": "F1",
        "caveats": [],
    }

    result, _ = run_with(monkeypatch, [soon, later], payload)

    assert result["recommended_id"] == "F2"
    assert any("urgent possibility" in c.lower() for c in result["caveats"])


def test_a_delayed_option_is_flagged(monkeypatch):
    delayed = dict(CANDIDATE, status="Delayed")

    caveats = run_with(monkeypatch, [delayed], good_payload())[0]["caveats"]

    assert any("delayed" in c.lower() for c in caveats)


def test_the_model_may_add_its_own_caveats(monkeypatch):
    payload = dict(good_payload(), caveats=["NOTE: the model noticed something."])

    caveats = run_with(monkeypatch, [CANDIDATE], payload)[0]["caveats"]

    assert "NOTE: the model noticed something." in caveats
    assert len(caveats) > 1          # code-generated ones come first and are kept


def test_every_caveat_declares_its_intent(monkeypatch):
    caveats = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["caveats"]

    for caveat in caveats:
        assert caveat.startswith(("NOTE:", "ASK:", "CONFIRM:")), caveat


# --- refusals, unchanged --------------------------------------------------------


def test_no_candidates_refuses_without_calling_the_model(monkeypatch):
    called = []
    monkeypatch.setattr(flights, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "call", lambda *a, **k: called.append(1))

    with pytest.raises(llm.LLMError) as caught:
        flight_agent.run(REQUEST, [])

    assert not called
    assert caught.value.steps == []
    assert "Nothing was invented" in str(caught.value)


def test_an_impossible_route_is_refused_before_any_search(monkeypatch):
    searched, called = [], []
    monkeypatch.setattr(flights, "search", lambda *a, **k: searched.append(1) or [])
    monkeypatch.setattr(llm, "call", lambda *a, **k: called.append(1))

    with pytest.raises(llm.LLMError) as caught:
        flight_agent.run(dict(REQUEST, destination="TLV"), [])

    assert not searched and not called
    assert "no flight to look for" in str(caught.value)


def test_an_unknown_airport_says_so_rather_than_blaming_the_data(monkeypatch):
    monkeypatch.setattr(llm, "call", lambda *a, **k: pytest.fail("should not be called"))

    with pytest.raises(llm.LLMError) as caught:
        flight_agent.run(dict(REQUEST, origin="VDA"), [])

    message = str(caught.value)
    assert "VDA" in message and "not an airport I can look up" in message
    assert "schedules were not available" not in message


def test_the_stub_flag_is_gone():
    assert not hasattr(flight_agent, "IS_STUB")
