"""Supervisor orchestration: the refinement gate, follow-up narrowing, the date sync,
the history cap, and the partial-failure policy.

Every LLM call in the path — the Supervisor's own two included — runs against the
`fake_llm` fixture in tests/conftest.py. No API key and no Supabase are needed.
"""

from datetime import date, datetime, timedelta

import pytest

from lib.agents import documentation_agent, supervisor
from lib.steps import make_step

pytestmark = pytest.mark.usefixtures("fake_llm", "fake_search_data")

COMPLETE = "LH318 TLV -> FRA was cancelled at the gate"

MODULES = {"Supervisor", "FlightAgent", "AccommodationAgent", "DocumentationAgent"}


@pytest.fixture(autouse=True)
def no_cost_documentation_agent(monkeypatch):
    """The harness tests exercise orchestration, not paid retrieval/model calls."""

    payload = {
        "regulation": "EU 261/2004",
        "entitlements": [],
        "next_actions": [],
        "caveats": [],
    }

    def fake_run(request, history):
        steps = [
            make_step("DocumentationAgent", phase, "test request", payload)
            for phase in ("draft", "critique", "refine")
        ]
        return payload, steps

    monkeypatch.setattr(documentation_agent, "run", fake_run)


def modules_of(steps):
    return [s["module"] for s in steps]


# --- the question-refinement gate ----------------------------------------------


def test_underspecified_message_asks_instead_of_dispatching():
    text, steps, _ = supervisor.run("my flight got cancelled help", [])

    assert "I need a couple of details" in text
    # The gate's whole point is cost: one call, not seven.
    assert modules_of(steps) == ["Supervisor"]


def test_gate_does_not_repeat_the_same_question_twice():
    # origin and destination are both missing and share a question.
    text, _, _ = supervisor.run("cancelled, I'm stuck", [])
    assert text.count("Which airport were you flying from") == 1


def test_complete_message_dispatches_the_crew():
    text, steps, _ = supervisor.run(COMPLETE, [])

    assert set(modules_of(steps)) == MODULES
    assert modules_of(steps)[0] == "Supervisor"    # refinement first
    assert modules_of(steps)[-1] == "Supervisor"   # compose last
    assert "ONWARD FLIGHT" in text


def test_every_step_module_is_on_the_architecture_diagram():
    _, steps, _ = supervisor.run(COMPLETE, [])
    assert set(modules_of(steps)) <= MODULES


def test_documentation_agent_emits_three_steps_for_its_reflection_loop():
    # The reason the interface contract is (payload, steps) rather than one step.
    _, steps, _ = supervisor.run(COMPLETE, [])
    assert modules_of(steps).count("DocumentationAgent") == 3


# --- follow-up turns ------------------------------------------------------------


def test_the_refinement_call_sees_the_earlier_turns():
    history = [{"prompt": COMPLETE, "response": "Onward flight: LH 687, TLV to FRA."}]
    _, steps, _ = supervisor.run("anything earlier?", history)

    assert "LH318" in steps[0]["prompt"]["user_prompt"]


def test_a_flight_follow_up_dispatches_only_the_flight_agent():
    history = [{"prompt": COMPLETE, "response": "Onward flight: LH 687, TLV to FRA."}]
    _, steps, _ = supervisor.run("anything earlier than the 04:25?", history)

    # Re-dispatching the crew here costs ~7 LLM calls and 2 AeroDataBox units for a
    # question only FlightAgent can answer.
    assert set(modules_of(steps)) == {"Supervisor", "FlightAgent"}


def test_a_follow_up_that_needs_nobody_is_answered_instead_of_interrogated():
    # Details are still missing, but nothing is being dispatched, so nothing is blocked.
    history = [{"prompt": "my flight got cancelled help", "response": "I need a couple of details"}]
    text, steps, _ = supervisor.run("never mind, thanks", history)

    assert modules_of(steps) == ["Supervisor", "Supervisor"]  # refinement, then compose
    assert "I need a couple of details" not in text


# --- the date sync --------------------------------------------------------------


def request_at(now: datetime) -> dict:
    return {"local_now": now.isoformat(), "party_size": 2, "stranded_at": "TLV"}


def flight_departing(when: datetime) -> dict:
    return {"options": [{"id": "F1", "depart": when.isoformat()}], "recommended_id": "F1"}


def test_stay_window_covers_tonight_until_the_flight_leaves():
    now = datetime(2026, 8, 8, 22, 15)
    window = supervisor._stay_window(
        request_at(now), flight_departing(datetime(2026, 8, 9, 9, 40))
    )

    assert window["check_in"] == "2026-08-08"
    assert window["check_out"] == "2026-08-09"
    assert window["nights"] == 1
    assert window["guests"] == 2


def test_stay_window_spans_multiple_nights():
    now = datetime(2026, 8, 8, 22, 15)
    window = supervisor._stay_window(
        request_at(now), flight_departing(datetime(2026, 8, 11, 6, 0))
    )
    assert window["nights"] == 3


def test_no_bed_needed_when_the_replacement_flight_leaves_today():
    now = datetime(2026, 8, 8, 6, 0)
    assert supervisor._stay_window(
        request_at(now), flight_departing(datetime(2026, 8, 8, 23, 55))
    ) is None


def test_no_stay_window_without_a_flight():
    now = datetime(2026, 8, 8, 22, 15)
    assert supervisor._stay_window(request_at(now), None) is None
    assert supervisor._stay_window(request_at(now), {"options": []}) is None


def test_accommodation_is_booked_for_the_nights_the_flight_implies():
    _, steps, _ = supervisor.run(COMPLETE, [])
    stay = next(s for s in steps if s["module"] == "AccommodationAgent")
    asked = stay["prompt"]["user_prompt"]

    # Assert on what the Supervisor *told* the agent, not on the agent echoing
    # it back: the date sync is the Supervisor's job, and this is where it shows.
    # The fake flight leaves 09:40 tomorrow, so: tonight, one night.
    assert f"Check in: {date.today().isoformat()}" in asked
    assert f"Check out: {(date.today() + timedelta(days=1)).isoformat()}" in asked
    assert "Nights: 1" in asked


# --- the history cap ------------------------------------------------------------


def test_history_is_capped():
    history = [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(20)]
    trimmed = supervisor._trim(history)

    assert len(trimmed) == supervisor.HISTORY_TURNS
    assert trimmed[-1]["prompt"] == "q19"  # keeps the most recent


def test_short_history_is_untouched():
    history = [{"prompt": "q", "response": "a"}]
    assert supervisor._trim(history) == history


def test_capped_history_is_what_reaches_the_agents():
    history = [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(20)]
    _, steps, _ = supervisor.run(COMPLETE, history)

    flight = next(s for s in steps if s["module"] == "FlightAgent")
    assert "q0" not in flight["prompt"]["user_prompt"]
    assert "q19" in flight["prompt"]["user_prompt"]


# --- the partial-failure policy -------------------------------------------------


def test_one_agent_failing_does_not_lose_the_rest_of_the_plan(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("Pinecone unreachable")

    monkeypatch.setattr(documentation_agent, "run", explode)
    text, steps, _ = supervisor.run(COMPLETE, [])

    assert "ONWARD FLIGHT" in text
    assert "SOMEWHERE TO SLEEP" in text
    assert supervisor.FAILURE_MESSAGES["rights"] in text
    # The cause is for us, not for a passenger standing at a gate.
    assert "Pinecone" not in text
    assert "DocumentationAgent" not in text


def test_an_agents_own_refusal_wording_survives(monkeypatch):
    from lib.agents import flight_agent
    from lib.llm import LLMError

    def refuse(*args, **kwargs):
        raise LLMError("FlightAgent: internal wording",
                       steps=[], passenger_message=flight_agent.NO_LIVE_DATA)

    monkeypatch.setattr(flight_agent, "run", refuse)
    text, _, _ = supervisor.run(COMPLETE, [])

    assert "check your airline's app or the departures board" in text
    assert "internal wording" not in text
    assert "FlightAgent" not in text


def test_an_internal_failure_falls_back_to_a_written_sentence(monkeypatch):
    from lib.agents import flight_agent
    from lib.llm import LLMError

    def refuse(*args, **kwargs):
        raise LLMError("FlightAgent: no option named a flight that was actually offered", steps=[])

    monkeypatch.setattr(flight_agent, "run", refuse)
    text, _, _ = supervisor.run(COMPLETE, [])

    assert supervisor.FAILURE_MESSAGES["flight"] in text
    assert "no option named a flight" not in text


def test_a_failed_call_still_appears_in_the_trace(monkeypatch):
    from lib.llm import LLMError
    from lib.steps import make_step

    failed_step = make_step("DocumentationAgent", "sys", "user", {"error": "timed out"})

    def fail(*args, **kwargs):
        raise LLMError("DocumentationAgent: timed out", steps=[failed_step])

    monkeypatch.setattr(documentation_agent, "run", fail)
    _, steps, _ = supervisor.run(COMPLETE, [])

    assert failed_step in steps
    assert modules_of(steps).count("DocumentationAgent") == 1


def test_calls_that_succeeded_before_a_failure_are_kept(monkeypatch):
    from lib.llm import LLMError
    from lib.steps import make_step

    drafted = make_step("DocumentationAgent", "draft", "user", {"ok": True})
    critiqued = make_step("DocumentationAgent", "critique", "user", {"error": "timed out"})

    def fail_halfway(*args, **kwargs):
        raise LLMError("DocumentationAgent: timed out", steps=[drafted, critiqued])

    monkeypatch.setattr(documentation_agent, "run", fail_halfway)
    _, steps, _ = supervisor.run(COMPLETE, [])

    # Both the successful draft and the failed critique survive, in order.
    assert steps.index(drafted) < steps.index(critiqued)
    assert modules_of(steps).count("DocumentationAgent") == 2


def test_a_failed_composing_call_still_hands_over_the_plan(monkeypatch):
    from lib import llm
    from lib.steps import make_step

    crew = llm.call  # the fake, already installed by the fixture
    failed_step = make_step("Supervisor", "compose", "user", {"error": "timed out"})

    def fail_on_compose(module, system_prompt, user_prompt, **kwargs):
        if "Findings:" in user_prompt:
            raise llm.LLMError("Supervisor: timed out", steps=[failed_step])
        return crew(module, system_prompt, user_prompt, **kwargs)

    monkeypatch.setattr(llm, "call", fail_on_compose)
    text, steps, _ = supervisor.run(COMPLETE, [])

    # Six calls and two external quotas are already spent by the time we compose.
    assert "ONWARD FLIGHT" in text
    assert "SOMEWHERE TO SLEEP" in text
    assert failed_step in steps          # and the call that failed is still in the trace


def test_a_half_labelled_option_still_produces_a_plan(monkeypatch):
    """Agent validation guarantees an id and a parseable date, not a full label."""
    from lib.agents import flight_agent

    bare = {"options": [{"id": "F1", "depart": (datetime.now() + timedelta(days=1)).isoformat()}],
            "recommended_id": "F1"}
    monkeypatch.setattr(flight_agent, "run", lambda *a, **k: (bare, []))

    text, _, _ = supervisor.run(COMPLETE, [])

    assert "ONWARD FLIGHT" in text
    assert supervisor.FAILURE_MESSAGES["flight"] not in text


def test_a_failing_flight_search_skips_the_stay_rather_than_guessing_nights(monkeypatch):
    from lib.agents import flight_agent

    monkeypatch.setattr(flight_agent, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
    text, steps, _ = supervisor.run(COMPLETE, [])

    # Booking the wrong nights is worse than saying a flight is needed first.
    assert "AccommodationAgent" not in modules_of(steps)
    assert supervisor.FAILURE_MESSAGES["flight"] in text
    assert "DocumentationAgent" in modules_of(steps)  # rights are independent


# --- the digest -------------------------------------------------------------


def digest_of(text_and_steps):
    return text_and_steps[0]


def test_every_option_reaches_the_plan_not_just_the_recommended_one():
    text, _, _ = supervisor.run(COMPLETE, [])

    # agent_info promises the passenger can compare. The composing call has to see
    # more than one option for that sentence to be true.
    assert "LH 687" in text
    assert "LY 357" in text


def test_the_recommended_option_comes_first():
    text, _, _ = supervisor.run(COMPLETE, [])

    assert text.index("LH 687") < text.index("LY 357")


def test_the_new_flight_facts_reach_the_plan():
    text, _, _ = supervisor.run(COMPLETE, [])

    assert "terminal 3" in text
    assert "Airbus A320" in text
    assert "Expected" in text


def test_the_hotels_phone_number_reaches_the_plan():
    text, _, _ = supervisor.run(COMPLETE, [])

    # The one job neither agent can do: confirming a room and a rate.
    assert "+972 3 000 0000" in text


def test_the_distance_reaches_the_plan_from_distance_km():
    text, _, _ = supervisor.run(COMPLETE, [])

    assert "2.4 km from the terminal" in text


def test_unknown_meals_are_not_asserted_as_absent():
    text, _, _ = supervisor.run(COMPLETE, [])

    # `meals_included: false` was a lie dressed as data - we almost never know.
    assert "meals not confirmed" in text
    assert "no meals" not in text


def test_the_deprecated_fields_are_read_nowhere():
    import inspect

    source = inspect.getsource(supervisor)
    assert "meals_included" not in source
    assert '"area"' not in source and "'area'" not in source


def test_the_options_shown_are_capped():
    payload = {"options": [{"id": f"F{n}"} for n in range(9)], "recommended_id": "F4"}

    shown = supervisor._options(payload)

    assert len(shown) == supervisor.MAX_DIGEST_OPTIONS
    assert shown[0]["id"] == "F4"


def test_no_payload_yields_no_options():
    assert supervisor._options(None) == []
    assert supervisor._options({"options": []}) == []


# --- the clock --------------------------------------------------------------


def test_the_passengers_own_clock_reaches_the_agents():
    when = datetime.now().replace(hour=3, minute=33, second=7, microsecond=0)
    _, steps, _ = supervisor.run(COMPLETE, [], local_time=when)

    flight = next(s for s in steps if s["module"] == "FlightAgent")
    assert "Local time now: " + when.isoformat(timespec="seconds") in flight["prompt"]["user_prompt"]


def test_without_a_clock_the_server_time_is_used():
    _, steps, _ = supervisor.run(COMPLETE, [])

    flight = next(s for s in steps if s["module"] == "FlightAgent")
    assert f"Local time now: {date.today().isoformat()}" in flight["prompt"]["user_prompt"]


def test_the_incident_time_survives_extraction():
    request = supervisor._request_from(
        {"incident_time": "2026-08-15T18:00"}, False, "2026-08-15T22:15:00")

    assert request["incident_time"] == "2026-08-15T18:00"


def test_an_absent_incident_time_is_none_not_missing():
    request = supervisor._request_from({}, False, "2026-08-15T22:15:00")

    # It is useful, not required: a passenger who cannot say when is still helped.
    assert request["incident_time"] is None
    assert "incident_time" not in request["missing"]


# --- sanity checks ---------------------------------------------------------------


def request_stating(**fields) -> dict:
    base = {"local_now": "2026-08-15T22:15:00", "origin": "TLV", "destination": "FRA",
            "stranded_at": "TLV", "incident_time": None, "arrive_by": None}
    return {**base, **fields}


def fields_of(conflicts):
    return [c["field"] for c in conflicts]


def test_a_date_months_away_is_a_conflict():
    # The motivating case: "today, 11th of November" read on 15 August.
    found = supervisor._conflicts(request_stating(incident_time="2026-11-11T09:00"))

    assert fields_of(found) == ["incident_time"]
    assert found[0]["stated"] == "2026-11-11T09:00"


def test_a_flight_cancelled_a_fortnight_ahead_is_not_a_conflict():
    # Airlines cancel in advance. Flagging that would bounce a real passenger.
    assert supervisor._conflicts(request_stating(incident_time="2026-08-29T09:00")) == []


def test_a_disruption_from_last_month_is_a_conflict():
    # Still a valid claim, but "a bed tonight" for it is nonsense.
    found = supervisor._conflicts(request_stating(incident_time="2026-07-04T09:00"))

    assert fields_of(found) == ["incident_time"]


def test_an_unparseable_incident_time_is_left_alone():
    # The model's job, not arithmetic's.
    assert supervisor._conflicts(request_stating(incident_time="last Tuesday")) == []


def test_a_route_that_cannot_exist_is_a_conflict():
    found = supervisor._conflicts(request_stating(origin="TLV", destination="TLV"))

    assert fields_of(found) == ["route"]
    assert "no flight to look for" in found[0]["reason"]


def test_a_half_stated_route_is_left_to_the_missing_fields_gate():
    # Otherwise the passenger is asked the same thing twice, differently worded.
    assert supervisor._conflicts(request_stating(origin="TLV", destination=None)) == []


def test_an_unknown_airport_is_a_conflict():
    found = supervisor._conflicts(request_stating(stranded_at="ZZZ"))

    assert fields_of(found) == ["stranded_at"]


def test_a_deadline_already_past_is_a_conflict():
    found = supervisor._conflicts(request_stating(arrive_by="2026-08-14T09:00"))

    assert fields_of(found) == ["arrive_by"]


def test_a_clean_request_has_no_conflicts():
    assert supervisor._conflicts(request_stating()) == []


def test_the_models_conflicts_are_kept_alongside_the_checked_ones():
    parsed = {"origin": "TLV", "destination": "TLV", "stranded_at": "TLV",
              "conflicts": [{"field": "airline", "stated": "Lufthansa",
                             "reason": "the flight number LY357 is El Al"}]}
    request = supervisor._request_from(parsed, False, "2026-08-15T22:15:00")

    assert fields_of(request["conflicts"]) == ["route", "airline"]


def test_the_checked_conflict_wins_when_both_name_the_same_field():
    parsed = {"origin": "TLV", "destination": "TLV", "stranded_at": "TLV",
              "conflicts": [{"field": "route", "stated": "TLV to TLV",
                             "reason": "made up by the model"}]}
    request = supervisor._request_from(parsed, False, "2026-08-15T22:15:00")

    assert fields_of(request["conflicts"]) == ["route"]
    assert "made up by the model" not in request["conflicts"][0]["reason"]


def test_junk_in_the_models_conflicts_is_dropped():
    parsed = {"origin": "TLV", "destination": "FRA", "stranded_at": "TLV",
              "conflicts": ["not an object", {"field": "airline"}, {"reason": ""}]}
    request = supervisor._request_from(parsed, False, "2026-08-15T22:15:00")

    assert request["conflicts"] == []


IMPOSSIBLE_ROUTE = "LH318 TLV -> TLV was cancelled at the gate"


def test_a_blocking_conflict_asks_instead_of_dispatching():
    text, steps, _ = supervisor.run(IMPOSSIBLE_ROUTE, [])

    assert "no flight to look for" in text
    # Same economics as the missing-fields gate: one call, not seven.
    assert modules_of(steps) == ["Supervisor"]


def test_the_conflict_question_reads_as_a_sentence():
    text, _, _ = supervisor.run(IMPOSSIBLE_ROUTE, [])

    # route_problem's reasons start lower-case because they are interpolated after
    # "FlightAgent: " today.
    assert "The origin and destination are both TLV" in text


def test_a_blocking_conflict_with_nothing_to_dispatch_does_not_interrogate():
    history = [{"prompt": IMPOSSIBLE_ROUTE, "response": "The origin and destination are both TLV"}]
    text, steps, _ = supervisor.run("never mind, thanks", history)

    assert modules_of(steps) == ["Supervisor", "Supervisor"]
    assert "Before I can help" not in text


def test_a_soft_conflict_proceeds_and_states_the_assumption():
    request = supervisor._request_from(
        {"origin": "TLV", "destination": "FRA", "stranded_at": "TLV",
         "arrive_by": "2026-08-14T09:00"},
        False, "2026-08-15T22:15:00")

    assert request["assumptions"] == [
        "They said 2026-08-14T09:00, but that deadline has already passed. Working without it."
    ]


def test_a_field_that_cannot_be_true_never_reaches_an_agent():
    request = supervisor._request_from(
        {"origin": "TLV", "destination": "FRA", "stranded_at": "TLV",
         "arrive_by": "2026-08-14T09:00"},
        False, "2026-08-15T22:15:00")

    # An impossible deadline handed to FlightAgent is worse than no deadline.
    assert request["arrive_by"] is None


def test_a_blocking_conflict_produces_no_assumption():
    request = supervisor._request_from(
        {"origin": "TLV", "destination": "TLV", "stranded_at": "TLV"},
        False, "2026-08-15T22:15:00")

    # It is being asked about, not assumed around.
    assert request["assumptions"] == []


def test_assumptions_reach_the_composing_call():
    request = {"assumptions": ["They said X, but Y."], "local_now": "2026-08-15T22:15:00"}
    prompt = supervisor._compose_prompt(request, "digest", [])

    assert "They said X, but Y." in prompt


def test_the_composing_prompt_never_names_the_inside_of_the_system():
    _, steps, _ = supervisor.run(COMPLETE, [])
    compose = [s for s in steps if s["module"] == "Supervisor"][-1]["prompt"]

    banned = ("crew", "FlightAgent", "AccommodationAgent", "DocumentationAgent", "agent")
    haystack = (compose["system_prompt"] + compose["user_prompt"]).lower()
    for word in banned:
        assert word.lower() not in haystack, f"{word!r} leaks into the composing call"


def test_the_composing_prompt_says_who_it_is():
    _, steps, _ = supervisor.run(COMPLETE, [])
    compose = [s for s in steps if s["module"] == "Supervisor"][-1]["prompt"]

    assert "Wingman" in compose["system_prompt"]


# --- caveats ---


CAVEATED = {
    "flight": {"options": [{"id": "F1"}], "recommended_id": "F1", "caveats": [
        "CONFIRM: LH 687 leaves in about 40 minutes - check they can reach the gate.",
        "NOTE: every option is on a different airline.",
    ]},
    "stay": {"options": [{"id": "H1"}], "recommended_id": "H1", "caveats": [
        "ASK: none of these has a phone number listed.",
        "every price here is an estimate.",
    ]},
}


def test_caveats_are_routed_to_their_own_blocks():
    text = supervisor._digest(CAVEATED, [])

    assert "BEFORE YOU ACT ON THIS" in text
    assert "THINGS I NEED FROM YOU" in text
    assert "WORTH KNOWING" in text


def test_the_prefix_is_stripped_before_the_passenger_sees_it():
    text = supervisor._digest(CAVEATED, [])

    assert "CONFIRM:" not in text
    assert "ASK:" not in text
    assert "NOTE:" not in text
    assert "LH 687 leaves in about 40 minutes" in text


def test_an_unprefixed_caveat_is_treated_as_a_note():
    routed = supervisor._split_caveats(CAVEATED)

    assert "every price here is an estimate." in routed["NOTE:"]


def test_the_rights_caveats_are_not_routed_as_the_search_protocol():
    results = {"rights": {"regulation": "EU 261/2004", "entitlements": [],
                          "next_actions": [], "caveats": ["No evidence on meals was retrieved."]}}
    text = supervisor._digest(results, [])

    # Same field name, different meaning: evidence gaps, not the NOTE:/ASK:/CONFIRM: protocol.
    assert "Not established from the sources: No evidence on meals was retrieved." in text
    assert "WORTH KNOWING" not in text


def test_no_caveats_means_no_empty_headings():
    text = supervisor._digest({"flight": {"options": [{"id": "F1"}], "recommended_id": "F1"}}, [])

    assert "BEFORE YOU ACT ON THIS" not in text
    assert "WORTH KNOWING" not in text


# --- results across turns ---


def test_the_results_come_back_with_the_plan():
    _, _, results = supervisor.run(COMPLETE, [])

    assert set(results) == {"flight", "stay", "rights"}
    assert results["flight"]["recommended_id"] == "F1"


def test_a_gated_turn_returns_no_results():
    _, _, results = supervisor.run("my flight got cancelled help", [])

    assert results == {}


def test_a_failed_agent_leaves_its_key_out(monkeypatch):
    monkeypatch.setattr(documentation_agent, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _, _, results = supervisor.run(COMPLETE, [])

    assert "rights" not in results
    assert "flight" in results


def test_earlier_options_reach_the_refinement_call():
    history = [{"prompt": COMPLETE, "response": "a plan",
                "results": {"flight": {"options": [{"id": "F1", "flight_number": "LH 687",
                                                    "depart": "2026-08-16T09:40"}],
                                       "recommended_id": "F1"}}}]
    _, steps, _ = supervisor.run("what else was there?", history)

    assert "LH 687" in steps[0]["prompt"]["user_prompt"]


def test_a_question_answered_from_earlier_options_dispatches_nobody():
    history = [{"prompt": COMPLETE, "response": "a plan",
                "results": {"flight": {"options": [{"id": "F1", "flight_number": "LH 687"}],
                                       "recommended_id": "F1"}}}]
    _, steps, _ = supervisor.run("never mind, thanks", history)

    # Re-running a search for something already found costs time and money.
    assert modules_of(steps) == ["Supervisor", "Supervisor"]


def test_a_turn_without_results_is_not_a_crash():
    history = [{"prompt": "hello", "response": "hi"}]

    assert supervisor._prior_results_block(history) == []


def test_only_the_most_recent_results_are_shown():
    history = [
        {"prompt": "a", "response": "a", "results": {"flight": {
            "options": [{"id": "F1", "flight_number": "OLD 111"}], "recommended_id": "F1"}}},
        {"prompt": "b", "response": "b", "results": {"flight": {
            "options": [{"id": "F1", "flight_number": "NEW 222"}], "recommended_id": "F1"}}},
    ]
    block = "\n".join(supervisor._prior_results_block(history))

    # History is re-sent on every call of every turn.
    assert "NEW 222" in block
    assert "OLD 111" not in block


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
