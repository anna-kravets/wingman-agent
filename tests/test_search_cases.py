"""The scenario file is data, so a typo in it should fail loudly here."""

import json

import pytest

from evals.search_cases import CHECK_NAMES, load_search_cases


def test_loads_all_twelve_scenarios():
    cases = load_search_cases()
    assert len(cases) == 12


def test_every_case_declares_what_it_probes():
    for case_id, case in load_search_cases().items():
        assert case["probes"], f"{case_id} declares no probes"
        assert case["title"], f"{case_id} has no title"


def test_every_declared_check_is_a_real_check():
    for case_id, case in load_search_cases().items():
        unknown = set(case["checks"]) - CHECK_NAMES
        assert not unknown, f"{case_id} names unknown checks: {unknown}"


def test_local_now_is_well_formed():
    for case_id, case in load_search_cases().items():
        local_now = case["local_now"]
        assert isinstance(local_now["offset_days"], int), case_id
        hour, _, minute = local_now["time"].partition(":")
        assert 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59, case_id


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "dupe.json"
    case = {
        "case_id": "same", "title": "t", "probes": ["x"], "prompt": "p",
        "history": [], "request_override": {}, "local_now": {"offset_days": 0, "time": "12:00"},
        "live_data": True, "checks": [], "expect": {},
    }
    path.write_text(json.dumps({"schema_version": 1, "cases": [case, dict(case)]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_search_cases(path)


def test_wrong_schema_version_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        load_search_cases(path)
