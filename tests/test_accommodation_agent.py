"""AccommodationAgent: enrichment from OpenStreetMap, the fixed stay window, caveats."""

import pytest

from lib import llm
from lib.agents import accommodation_agent
from lib.steps import make_step
from lib.tools import hotels

REQUEST = {"stranded_at": "TLV", "party_size": 2, "local_now": "2026-08-09T22:15:00"}
WINDOW = {"check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1,
          "guests": 2, "departs": "2026-08-10T09:40:00"}

CANDIDATE = {"name": "Airport Plaza", "distance_km": 2.4, "area": "Lod", "stars": "4",
             "phone": "+972 3 000 0000", "website": "https://example.test",
             "address": "12 HaNasi", "wheelchair": "yes"}


def fake_call(payload):
    def call(module, system_prompt, user_prompt, *, expect_json=False, **kwargs):
        return payload, make_step(module, system_prompt, user_prompt, payload)
    return call


def good_payload(name="Airport Plaza", recommended="H1"):
    return {
        "options": [{
            "id": "H1", "name": name,
            "price_estimate": "Roughly EUR 110-140 (estimate - not a quoted price)",
            "notes": "Closest to the terminal.",
        }],
        "recommended_id": recommended,
    }


def run_with(monkeypatch, candidates, payload):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: candidates)
    monkeypatch.setattr(llm, "call", fake_call(payload))
    return accommodation_agent.run(REQUEST, WINDOW, [])


# --- grounding and the trace ----------------------------------------------------


def test_real_hotels_reach_the_prompt(monkeypatch):
    _, steps = run_with(monkeypatch, [CANDIDATE], good_payload())

    assert "Airport Plaza" in steps[0]["prompt"]["user_prompt"]


def test_one_llm_call_produces_exactly_one_step(monkeypatch):
    _, steps = run_with(monkeypatch, [CANDIDATE], good_payload())

    assert len(steps) == 1
    assert steps[0]["module"] == "AccommodationAgent"


def test_the_stay_window_is_stated_as_non_negotiable(monkeypatch):
    _, steps = run_with(monkeypatch, [CANDIDATE], good_payload())
    user_prompt = steps[0]["prompt"]["user_prompt"]

    assert "Check in: 2026-08-09" in user_prompt
    assert "Check out: 2026-08-10" in user_prompt
    assert "Nights: 1" in user_prompt


# --- facts come from the candidate ----------------------------------------------


def test_facts_come_from_the_candidate(monkeypatch):
    option = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["options"][0]

    assert option["distance_km"] == 2.4
    assert option["phone"] == "+972 3 000 0000"
    assert option["website"] == "https://example.test"
    assert option["address"] == "12 HaNasi"
    assert option["wheelchair"] == "yes"
    assert option["city"] == "Lod"


def test_the_stay_window_is_written_not_asked_for(monkeypatch):
    option = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["options"][0]

    # The nights come from the flight that was found; the model cannot disagree.
    assert option["check_in"] == WINDOW["check_in"]
    assert option["check_out"] == WINDOW["check_out"]
    assert option["nights"] == WINDOW["nights"]


def test_an_invented_property_is_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({"id": "H2", "name": "Imaginary Suites",
                               "price_estimate": "", "notes": ""})

    result, _ = run_with(monkeypatch, [CANDIDATE], payload)

    assert [o["id"] for o in result["options"]] == ["H1"]


def test_no_usable_option_raises_but_keeps_the_step(monkeypatch):
    invented = {"options": [{"id": "H1", "name": "Imaginary Suites"}],
                "recommended_id": "H1"}

    with pytest.raises(llm.LLMError) as caught:
        run_with(monkeypatch, [CANDIDATE], invented)

    assert len(caught.value.steps) == 1


def test_a_dangling_recommended_id_falls_back(monkeypatch):
    result, _ = run_with(monkeypatch, [CANDIDATE], good_payload(recommended="H9"))

    assert result["recommended_id"] == "H1"


def test_meals_is_unknown_unless_the_data_says_otherwise(monkeypatch):
    option = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["options"][0]

    # "meals_included: false" was a lie dressed as data - we almost never know.
    assert option["meals"] == "unknown"


def test_meals_follows_the_breakfast_tag(monkeypatch):
    for tag, expected in (("yes", "included"), ("no", "not_included")):
        option = run_with(monkeypatch, [dict(CANDIDATE, breakfast=tag)],
                          good_payload())[0]["options"][0]
        assert option["meals"] == expected


def test_kind_appears_only_when_it_is_not_a_hotel(monkeypatch):
    option = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["options"][0]
    assert "kind" not in option

    option = run_with(monkeypatch, [dict(CANDIDATE, kind="hostel")],
                      good_payload())[0]["options"][0]
    assert option["kind"] == "hostel"


# --- caveats --------------------------------------------------------------------


def test_no_phone_anywhere_is_an_ask(monkeypatch):
    bare = {"name": "Airport Plaza", "distance_km": 2.4, "area": "Lod"}

    caveats = run_with(monkeypatch, [bare], good_payload())[0]["caveats"]

    assert any(c.startswith("ASK:") and "phone" in c for c in caveats)


def test_a_distant_option_asks_for_confirmation(monkeypatch):
    far = dict(CANDIDATE, distance_km=17.0)

    caveats = run_with(monkeypatch, [far], good_payload())[0]["caveats"]

    assert any(c.startswith("CONFIRM:") and "17" in c for c in caveats)


def test_only_non_hotels_is_flagged(monkeypatch):
    hostel = dict(CANDIDATE, kind="hostel")

    caveats = run_with(monkeypatch, [hostel], good_payload())[0]["caveats"]

    assert any("no ordinary hotels" in c.lower() for c in caveats)


def test_prices_are_always_flagged_as_estimates(monkeypatch):
    caveats = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["caveats"]

    assert any("estimate" in c.lower() for c in caveats)


def test_the_model_may_add_its_own_caveats(monkeypatch):
    payload = dict(good_payload(), caveats=["NOTE: the model noticed something."])

    caveats = run_with(monkeypatch, [CANDIDATE], payload)[0]["caveats"]

    assert "NOTE: the model noticed something." in caveats


def test_every_caveat_declares_its_intent(monkeypatch):
    caveats = run_with(monkeypatch, [CANDIDATE], good_payload())[0]["caveats"]

    for caveat in caveats:
        assert caveat.startswith(("NOTE:", "ASK:", "CONFIRM:")), caveat


# --- refusal, unchanged ---------------------------------------------------------


def test_no_candidates_refuses_without_calling_the_model(monkeypatch):
    called = []
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "call", lambda *a, **k: called.append(1))

    with pytest.raises(llm.LLMError) as caught:
        accommodation_agent.run(REQUEST, WINDOW, [])

    assert not called
    assert caught.value.steps == []
    assert "Nothing was invented" in str(caught.value)


def test_the_stub_flag_is_gone():
    assert not hasattr(accommodation_agent, "IS_STUB")
