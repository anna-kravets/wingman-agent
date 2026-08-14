"""The runner's wiring, exercised with fakes so it costs nothing."""

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "run_search_agents_live", REPO_ROOT / "scripts" / "run_search_agents_live.py")
runner = importlib.util.module_from_spec(spec)
sys.modules["run_search_agents_live"] = runner
spec.loader.exec_module(runner)


CASE = {
    "case_id": "t", "title": "t", "probes": ["x"],
    "prompt": "LH318 TLV -> FRA was cancelled at the gate",
    "history": [], "request_override": {"party_size": 4},
    "local_now": {"offset_days": 1, "time": "23:30"},
    "live_data": True, "checks": [], "expect": {},
}


def complete_request():
    return {"party_size": 1, "local_now": "ignored", "flight_number": "LH318",
            "origin": "TLV", "destination": "FRA", "disruption": "cancelled",
            "stranded_at": "TLV"}


def test_local_now_is_injected_from_the_case():
    def original(prompt, history):
        return complete_request(), {"module": "Supervisor"}

    patched = runner.build_request_patch(CASE, original)
    request, _ = patched("prompt", [])

    expected_date = date.today() + timedelta(days=1)
    assert request["local_now"].startswith(expected_date.isoformat())
    assert request["local_now"].endswith("23:30:00")


def test_request_override_wins():
    def original(prompt, history):
        return complete_request(), {"module": "Supervisor"}

    request, _ = runner.build_request_patch(CASE, original)("prompt", [])
    assert request["party_size"] == 4


def test_missing_is_recomputed_so_the_gate_still_works():
    def original(prompt, history):
        return {"party_size": 1, "local_now": "x", "flight_number": None,
                "origin": None, "destination": None, "disruption": None,
                "stranded_at": None}, {"module": "Supervisor"}

    case = dict(CASE, request_override={})
    request, _ = runner.build_request_patch(case, original)("prompt", [])
    assert "flight_number" in request["missing"]


def test_call_ceiling_raises_before_overspending():
    with pytest.raises(RuntimeError, match="call ceiling"):
        runner.guard_budget(used=30, ceiling=30)


def test_call_ceiling_allows_headroom():
    runner.guard_budget(used=10, ceiling=30)  # must not raise
