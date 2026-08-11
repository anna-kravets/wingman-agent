"""Hotel search, replayed against a real captured Overpass response.

The fixture is a genuine 12km query around TLV: 20 hotels, of which only one
carries a star rating. That sparseness is why price and meals must never be
asserted downstream.
"""

import json
import pathlib

import httpx
import pytest

from lib.tools import hotels

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "overpass_tlv_hotels.json")
    .read_text(encoding="utf-8")
)


@pytest.fixture(autouse=True)
def live(monkeypatch):
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "1")
    hotels._cache.clear()


def respond_with(payload, status=200, calls=None):
    def post(url, **kwargs):
        if calls is not None:
            calls.append(url)
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))
    return post


def test_returns_named_hotels_sorted_by_distance(monkeypatch):
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE))

    results = hotels.search("TLV")

    assert results
    assert all(r["name"] for r in results)
    assert results == sorted(results, key=lambda r: r["distance_km"])


def test_distance_is_computed_from_the_airport(monkeypatch):
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE))

    results = hotels.search("TLV")

    # Everything came from a 12km radius query around Ben Gurion.
    assert all(0 <= r["distance_km"] <= 13 for r in results)


def test_result_list_is_capped_for_prompt_size(monkeypatch):
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE))

    results = hotels.search("TLV")

    assert len(results) <= hotels.MAX_RESULTS
    assert len(json.dumps(results, ensure_ascii=False)) < 2000


def test_english_names_are_preferred(monkeypatch):
    payload = {"elements": [
        {"lat": 32.02, "lon": 34.89,
         "tags": {"tourism": "hotel", "name": "מלון שדה", "name:en": "Airport Hotel"}},
    ]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    assert hotels.search("TLV")[0]["name"] == "Airport Hotel"


def test_unnamed_entries_are_dropped(monkeypatch):
    payload = {"elements": [
        {"lat": 32.02, "lon": 34.89, "tags": {"tourism": "hotel"}},
        {"lat": 32.03, "lon": 34.90, "tags": {"tourism": "hotel", "name": "Real Hotel"}},
    ]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    results = hotels.search("TLV")

    assert [r["name"] for r in results] == ["Real Hotel"]


def test_ways_use_their_centre_coordinate(monkeypatch):
    payload = {"elements": [
        {"type": "way", "center": {"lat": 32.02, "lon": 34.89},
         "tags": {"tourism": "hotel", "name": "Way Hotel"}},
    ]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    assert hotels.search("TLV")[0]["name"] == "Way Hotel"


def test_results_are_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE, calls=calls))

    hotels.search("TLV")
    hotels.search("TLV")

    assert len(calls) == 1


def test_a_failing_mirror_falls_through_to_the_next(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return httpx.Response(504, text="gateway timeout",
                                  request=httpx.Request("POST", url))
        return httpx.Response(200, json=FIXTURE, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)

    # The public Overpass instance really does return 504 intermittently.
    assert hotels.search("TLV")
    assert len(calls) == 2


def test_all_mirrors_failing_degrades_to_empty(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "post", boom)

    assert hotels.search("TLV") == []


def test_unknown_airport_returns_empty(monkeypatch):
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE))

    assert hotels.search("ZZZZ") == []


def test_kill_switch_makes_no_request(monkeypatch):
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "0")
    calls = []
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE, calls=calls))

    assert hotels.search("TLV") == []
    assert calls == []
