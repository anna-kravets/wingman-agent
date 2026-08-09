"""Flight search, replayed against a real captured AeroDataBox response.

The fixture is a genuine TLV departures window: 188 departures, 118KB, of
which exactly 3 go to FRA. That ratio is the reason filtering lives in this
module and not in a prompt.
"""

import json
import pathlib
from datetime import datetime

import httpx
import pytest

from lib.tools import flights

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "aerodatabox_tlv_departures.json")
    .read_text(encoding="utf-8")
)
AFTER = datetime.fromisoformat("2026-08-10T06:00")


@pytest.fixture(autouse=True)
def live(monkeypatch):
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "1")
    monkeypatch.setenv("AERODATABOX_API_KEY", "test-key")
    monkeypatch.setenv("AERODATABOX_API_HOST", "example.invalid")
    flights._cache.clear()


def respond_with(payload, status=200, calls=None):
    def get(url, **kwargs):
        if calls is not None:
            calls.append(url)
        return httpx.Response(status, json=payload, request=httpx.Request("GET", url))
    return get


def test_filters_the_whole_airport_down_to_the_route(monkeypatch):
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE))

    results = flights.search("TLV", "FRA", AFTER)

    assert len(results) == 3
    assert {r["destination"] for r in results} == {"FRA"}
    assert {r["airline_iata"] for r in results} == {"LY", "DE", "LH"}


def test_trimmed_results_are_small_enough_for_a_prompt(monkeypatch):
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE))

    results = flights.search("TLV", "FRA", AFTER)

    # The raw response is ~118KB. Anything near that in a prompt would eat
    # the project's $13 budget.
    assert len(json.dumps(results)) < 2000


def test_timestamps_are_normalised_to_iso_8601(monkeypatch):
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE))

    results = flights.search("TLV", "FRA", AFTER)

    for result in results:
        # AeroDataBox sends "2026-08-10 06:05+03:00"; the Supervisor calls
        # datetime.fromisoformat on this and loses the hotel too if it fails.
        assert "T" in result["depart"]
        datetime.fromisoformat(result["depart"])
        datetime.fromisoformat(result["arrive"])


def test_results_are_cached_so_a_repeat_search_is_free(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    flights.search("TLV", "FRA", AFTER)
    flights.search("TLV", "FRA", AFTER)

    assert len(calls) == 1


def test_kill_switch_makes_no_request_at_all(monkeypatch):
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "0")
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    assert flights.search("TLV", "FRA", AFTER) == []
    assert calls == []


def test_missing_key_returns_empty_rather_than_raising(monkeypatch):
    monkeypatch.delenv("AERODATABOX_API_KEY", raising=False)

    # The agent degrades to LLM-only; it must not lose the turn.
    assert flights.search("TLV", "FRA", AFTER) == []


def test_a_route_with_3_options_does_not_spend_a_second_call(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    # FRA has exactly 3 flights in this window — enough, so stop at one call.
    flights.search("TLV", "FRA", AFTER)

    assert len(calls) == 1


def test_a_thin_route_widens_the_window_once(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    # NCE has exactly 1 flight in this window — too thin, so it widens once.
    results = flights.search("TLV", "NCE", AFTER)

    assert len(calls) == 2   # never more: 2 calls == 4 of 600 monthly units
    assert len(results) == 2  # one from each window (the fixture replays)


def test_rate_limit_is_retried_once(monkeypatch):
    responses = [
        httpx.Response(429, json={"message": "rate limit"},
                       request=httpx.Request("GET", "https://example.invalid")),
        httpx.Response(200, json=FIXTURE,
                       request=httpx.Request("GET", "https://example.invalid")),
    ]
    monkeypatch.setattr(httpx, "get", lambda url, **kw: responses.pop(0))
    monkeypatch.setattr(flights.time, "sleep", lambda _: None)

    assert len(flights.search("TLV", "FRA", AFTER)) == 3


def test_transport_failure_degrades_to_empty(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "get", boom)

    assert flights.search("TLV", "FRA", AFTER) == []
