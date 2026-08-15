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


def complete_request(local_now="ignored"):
    return {"party_size": 1, "local_now": local_now, "flight_number": "LH318",
            "origin": "TLV", "destination": "FRA", "disruption": "cancelled",
            "stranded_at": "TLV"}


def test_local_now_is_injected_from_the_case():
    def original(prompt, history, local_now):
        # patched() no longer overwrites local_now itself — it now trusts
        # `original` to embed the clock it was handed, as the real
        # supervisor._extract_request does via _request_from.
        return complete_request(local_now), {"module": "Supervisor"}

    patched = runner.build_request_patch(CASE, original)
    request, _ = patched("prompt", [], "ignored")

    expected_date = date.today() + timedelta(days=1)
    assert request["local_now"].startswith(expected_date.isoformat())
    assert request["local_now"].endswith("23:30:00")


def test_request_override_wins():
    def original(prompt, history, local_now):
        return complete_request(), {"module": "Supervisor"}

    request, _ = runner.build_request_patch(CASE, original)("prompt", [], "ignored")
    assert request["party_size"] == 4


def test_missing_is_recomputed_so_the_gate_still_works():
    def original(prompt, history, local_now):
        return {"party_size": 1, "local_now": "x", "flight_number": None,
                "origin": None, "destination": None, "disruption": None,
                "stranded_at": None}, {"module": "Supervisor"}

    case = dict(CASE, request_override={})
    request, _ = runner.build_request_patch(case, original)("prompt", [], "ignored")
    assert "flight_number" in request["missing"]


def test_call_ceiling_raises_before_overspending():
    with pytest.raises(RuntimeError, match="call ceiling"):
        runner.guard_budget(used=30, ceiling=30)


def test_call_ceiling_allows_headroom():
    runner.guard_budget(used=10, ceiling=30)  # must not raise
