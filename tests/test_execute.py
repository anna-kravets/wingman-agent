"""The graded surface: the four endpoints and the exact shapes the spec locks down.

Runs with no API key and no Supabase — LLM calls go through the `fake_llm` fixture,
and conversation state is best-effort, so the endpoint answers as a single turn when
the env vars are absent.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from api.index import app
from lib import conversation
from lib.agents import documentation_agent, supervisor
from lib.steps import make_step

client = TestClient(app)

# The Supervisor's own refinement pass is a real LLM call now, so this file needs the
# fake as much as test_supervisor.py does: without it these tests either fail with no
# key configured, or spend real money with one.
pytestmark = pytest.mark.usefixtures("fake_llm")

COMPLETE = "LH318 TLV -> FRA was cancelled at the gate"

TOP_LEVEL = {"status", "error", "response", "steps"}
MODULES = {"Supervisor", "FlightAgent", "AccommodationAgent", "DocumentationAgent"}


@pytest.fixture(autouse=True)
def no_cost_documentation_agent(monkeypatch):
    payload = {
        "regulation": "EU 261/2004",
        "entitlements": [],
        "next_actions": [],
        "caveats": [],
    }

    def fake_run(request, history):
        return payload, [
            make_step("DocumentationAgent", phase, "test request", payload)
            for phase in ("draft", "critique", "refine")
        ]

    monkeypatch.setattr(documentation_agent, "run", fake_run)


# --- POST /api/execute ----------------------------------------------------------


def test_success_response_has_exactly_the_locked_fields():
    body = client.post("/api/execute", json={"prompt": COMPLETE}).json()

    assert set(body) == TOP_LEVEL
    assert body["status"] == "ok"
    assert body["error"] is None
    assert isinstance(body["response"], str) and body["response"]


def test_every_step_has_the_required_schema():
    steps = client.post("/api/execute", json={"prompt": COMPLETE}).json()["steps"]

    assert steps
    for step in steps:
        assert set(step) == {"module", "prompt", "response"}
        assert set(step["prompt"]) == {"system_prompt", "user_prompt"}
        assert step["module"] in MODULES
        assert step["prompt"]["system_prompt"]


@pytest.mark.parametrize(
    "body",
    [{}, {"prompt": None}, {"prompt": 42}, {"prompt": "   "}, ["prompt"]],
    ids=["missing", "null", "not-a-string", "blank", "not-an-object"],
)
def test_bad_input_is_rejected_with_a_readable_message(body):
    response = client.post("/api/execute", json=body).json()

    assert set(response) == TOP_LEVEL
    assert response["status"] == "error"
    assert response["response"] is None
    assert response["steps"] == []
    # The spec asks for a human-readable description, not a bare KeyError.
    assert "prompt" in response["error"] and len(response["error"].split()) > 3


def test_invalid_json_is_rejected_with_a_readable_message():
    response = client.post(
        "/api/execute", content=b"{not json", headers={"Content-Type": "application/json"}
    ).json()

    assert response["status"] == "error"
    assert "valid JSON" in response["error"]


def test_agent_failure_is_reported_in_the_error_shape(monkeypatch):
    monkeypatch.setattr(
        supervisor, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    response = client.post("/api/execute", json={"prompt": COMPLETE}).json()

    assert response["status"] == "error"
    assert response["response"] is None
    assert "boom" in response["error"]


# --- multi-turn -----------------------------------------------------------------


def test_prior_turns_reach_the_supervisor(monkeypatch):
    store = {"c1": [{"prompt": "earlier question", "response": "earlier answer"}]}
    owners = []

    def fake_load(owner_id, conversation_id):
        owners.append(owner_id)
        return list(store.get(conversation_id, []))

    def fake_save(owner_id, conversation_id, history, title):
        owners.append(owner_id)
        store[conversation_id] = history

    monkeypatch.setattr(conversation, "load_history", fake_load)
    monkeypatch.setattr(conversation, "save_history", fake_save)

    seen = {}
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(history=list(history)), ("ok", []))[1],
    )

    client.post("/api/execute", json={"prompt": "follow up", "conversation_id": "c1"})

    assert seen["history"] == [{"prompt": "earlier question", "response": "earlier answer"}]
    assert len(store["c1"]) == 2               # the new turn was persisted
    assert store["c1"][-1]["prompt"] == "follow up"
    assert len(set(owners)) == 1                # one cookie owner scopes both operations


def test_works_as_a_single_turn_when_supabase_is_not_configured(monkeypatch):
    # No env vars in CI or on a fresh clone: the GUI still sends a conversation_id,
    # and the agent must answer rather than error.
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    body = client.post(
        "/api/execute", json={"prompt": COMPLETE, "conversation_id": "unconfigured"}
    ).json()

    assert body["status"] == "ok"


def test_history_failure_does_not_block_an_agent_response(monkeypatch):
    monkeypatch.setattr(
        conversation, "load_history", lambda *args: (_ for _ in ()).throw(RuntimeError("db down"))
    )
    monkeypatch.setattr(conversation, "save_history", lambda *args: None)

    body = client.post(
        "/api/execute", json={"prompt": COMPLETE, "conversation_id": "resilient"}
    ).json()

    assert body["status"] == "ok"


def test_history_save_failure_does_not_discard_an_agent_response(monkeypatch):
    monkeypatch.setattr(conversation, "load_history", lambda *args: [])
    monkeypatch.setattr(
        conversation, "save_history", lambda *args: (_ for _ in ()).throw(RuntimeError("db down"))
    )

    body = client.post(
        "/api/execute", json={"prompt": COMPLETE, "conversation_id": "resilient"}
    ).json()

    assert body["status"] == "ok"


def test_anonymous_cookie_scopes_conversation_listing(monkeypatch):
    seen_owners = []
    monkeypatch.setattr(
        conversation,
        "list_conversations",
        lambda owner_id: seen_owners.append(owner_id) or [],
    )
    anonymous_client = TestClient(app)

    first = anonymous_client.get("/api/conversations")
    second = anonymous_client.get("/api/conversations")

    assert first.json() == {"status": "ok", "conversations": []}
    assert "wingman_device_id=" in first.headers["set-cookie"]
    assert "HttpOnly" in first.headers["set-cookie"]
    assert "SameSite=lax" in first.headers["set-cookie"]
    assert len(seen_owners) == 2 and seen_owners[0] == seen_owners[1]
    assert anonymous_client.cookies.get("wingman_device_id") != seen_owners[0]
    assert len(seen_owners[0]) == 64  # Supabase stores a hash, not the bearer cookie.


def test_delete_is_owner_scoped(monkeypatch):
    deleted = []
    monkeypatch.setattr(
        conversation,
        "delete_conversation",
        lambda owner_id, conversation_id: deleted.append((owner_id, conversation_id)),
    )
    anonymous_client = TestClient(app)

    response = anonymous_client.delete("/api/conversations/c1")

    assert response.json() == {"status": "ok"}
    assert len(deleted) == 1
    assert deleted[0][1] == "c1"


def test_the_browsers_local_time_reaches_the_supervisor(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(when=local_time), ("ok", []))[1],
    )

    client.post("/api/execute", json={"prompt": COMPLETE, "local_time": "2026-03-04T05:06:07"})

    assert seen["when"] == datetime(2026, 3, 4, 5, 6, 7)


def test_an_offset_is_reduced_to_the_wall_clock_reading(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(when=local_time), ("ok", []))[1],
    )

    client.post("/api/execute", json={"prompt": COMPLETE, "local_time": "2026-03-04T05:06:07+03:00"})

    # Naive throughout: flight_agent subtracts local_now from a tz-stripped departure.
    assert seen["when"] == datetime(2026, 3, 4, 5, 6, 7)


def test_an_unusable_local_time_falls_back_to_the_server_clock(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        supervisor, "run",
        lambda prompt, history, *, local_time=None: (seen.update(when=local_time), ("ok", []))[1],
    )

    client.post("/api/execute", json={"prompt": COMPLETE, "local_time": "yesterday evening"})

    assert seen["when"] is None


# --- the other three endpoints --------------------------------------------------


def test_team_info_has_the_required_fields():
    body = client.get("/api/team_info").json()

    assert set(body) == {"group_batch_order_number", "team_name", "students"}
    assert len(body["students"]) == 3
    for student in body["students"]:
        assert set(student) == {"name", "email"} and "@" in student["email"]
    assert "TODO" not in str(body)


def test_agent_info_has_the_required_fields():
    body = client.get("/api/agent_info").json()

    assert set(body) == {"description", "purpose", "prompt_template", "prompt_examples"}
    assert "template" in body["prompt_template"]
    for module in MODULES:
        # The spec requires module names to agree across the diagram, the trace and
        # any written description. This is the description.
        assert module in body["description"]


def test_model_architecture_returns_a_png():
    response = client.get("/api/model_architecture")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_gui_is_served_without_auth():
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="run"' in response.text        # the send button
    assert "<textarea" in response.text
    assert 'id="new-chat"' in response.text
    assert 'id="conversation-list"' in response.text
    assert 'id="delete-dialog"' in response.text
    assert 'id="confirm-delete"' in response.text
    assert 'method: "DELETE"' in response.text
    assert "localStorage" in response.text
    assert "No account required" not in response.text
    assert 'type="password"' not in response.text
    assert 'href="/login"' not in response.text
    assert "UI preview" not in response.text
    assert "demo-conversation" not in response.text
