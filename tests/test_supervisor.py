"""Supervisor logic that does not depend on an LLM: the refinement gate, the date
sync, the history cap, and the partial-failure policy.

These run against the sub-agent stubs, so no API key and no Supabase are needed.
"""

from datetime import date, datetime, timedelta

import pytest

from lib.agents import documentation_agent, supervisor
from lib.steps import make_step

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
    text, steps = supervisor.run("my flight got cancelled help", [])

    assert "I need a couple of details" in text
    # The gate's whole point is cost: one call, not seven.
    assert modules_of(steps) == ["Supervisor"]


def test_gate_does_not_repeat_the_same_question_twice():
    # origin and destination are both missing and share a question.
    text, _ = supervisor.run("cancelled, I'm stuck", [])
    assert text.count("Which airport were you flying from") == 1


def test_complete_message_dispatches_the_crew():
    text, steps = supervisor.run(COMPLETE, [])

    assert set(modules_of(steps)) == MODULES
    assert modules_of(steps)[0] == "Supervisor"    # refinement first
    assert modules_of(steps)[-1] == "Supervisor"   # compose last
    assert "Onward flight" in text


def test_every_step_module_is_on_the_architecture_diagram():
    _, steps = supervisor.run(COMPLETE, [])
    assert set(modules_of(steps)) <= MODULES


def test_documentation_agent_emits_three_steps_for_its_reflection_loop():
    # The reason the interface contract is (payload, steps) rather than one step.
    _, steps = supervisor.run(COMPLETE, [])
    assert modules_of(steps).count("DocumentationAgent") == 3


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
    _, steps = supervisor.run(COMPLETE, [])
    stay = next(s for s in steps if s["module"] == "AccommodationAgent")
    option = stay["response"]["options"][0]

    # The stub flight leaves 09:40 tomorrow, so: tonight, one night.
    assert option["check_in"] == date.today().isoformat()
    assert option["check_out"] == (date.today() + timedelta(days=1)).isoformat()
    assert option["nights"] == 1


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
    _, steps = supervisor.run(COMPLETE, history)

    flight = next(s for s in steps if s["module"] == "FlightAgent")
    assert "q0" not in flight["prompt"]["user_prompt"]
    assert "q19" in flight["prompt"]["user_prompt"]


# --- the partial-failure policy -------------------------------------------------


def test_one_agent_failing_does_not_lose_the_rest_of_the_plan(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("Pinecone unreachable")

    monkeypatch.setattr(documentation_agent, "run", explode)
    text, steps = supervisor.run(COMPLETE, [])

    assert "Onward flight" in text                  # flight survived
    assert "Somewhere to sleep" in text             # stay survived
    assert "Could not complete: your entitlements" in text
    assert "Pinecone unreachable" in text           # the cause is not swallowed
    assert "DocumentationAgent" not in modules_of(steps)


def test_a_failed_call_still_appears_in_the_trace(monkeypatch):
    from lib.llm import LLMError
    from lib.steps import make_step

    failed_step = make_step("DocumentationAgent", "sys", "user", {"error": "timed out"})

    def fail(*args, **kwargs):
        raise LLMError("DocumentationAgent: timed out", steps=[failed_step])

    monkeypatch.setattr(documentation_agent, "run", fail)
    _, steps = supervisor.run(COMPLETE, [])

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
    _, steps = supervisor.run(COMPLETE, [])

    # Both the successful draft and the failed critique survive, in order.
    assert steps.index(drafted) < steps.index(critiqued)
    assert modules_of(steps).count("DocumentationAgent") == 2


def test_a_failing_flight_search_skips_the_stay_rather_than_guessing_nights(monkeypatch):
    from lib.agents import flight_agent

    monkeypatch.setattr(flight_agent, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
    text, steps = supervisor.run(COMPLETE, [])

    # Booking the wrong nights is worse than saying a flight is needed first.
    assert "AccommodationAgent" not in modules_of(steps)
    assert "Could not complete: onward flights" in text
    assert "DocumentationAgent" in modules_of(steps)  # rights are independent


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
