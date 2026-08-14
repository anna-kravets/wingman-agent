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
    monkeypatch.setattr(hotels.time, "sleep", lambda _: None)  # no real waiting in tests
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


def test_a_throttled_mirror_is_retried_before_giving_up(monkeypatch):
    # Overpass throttles on slot contention and recovers in seconds. A live run
    # refused to propose any hotel because of a blip, moments after the same
    # query had returned eight - so one refusal per blip is not acceptable.
    attempts = []

    def post(url, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            return httpx.Response(429, text="too many requests",
                                  request=httpx.Request("POST", url))
        return httpx.Response(200, json=FIXTURE, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(hotels.time, "sleep", lambda _: None)

    assert hotels.search("TLV")
    assert attempts[0] == attempts[1]  # same mirror retried, not skipped


def test_retries_are_bounded(monkeypatch):
    attempts = []

    def post(url, **kwargs):
        attempts.append(url)
        return httpx.Response(504, text="gateway timeout", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(hotels.time, "sleep", lambda _: None)

    assert hotels.search("TLV", radius_km=12) == []
    assert len(attempts) == len(hotels.MIRRORS) * hotels.ATTEMPTS_PER_MIRROR


def test_all_mirrors_failing_degrades_to_empty(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "post", boom)

    assert hotels.search("TLV") == []


def test_unknown_airport_returns_empty(monkeypatch):
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE))

    assert hotels.search("ZZZZ") == []


# --- richer detail, all of it already in the response we were paying for --------


def hotel(name, **tags):
    return {"lat": 32.02, "lon": 34.89, "tags": {"tourism": "hotel", "name": name, **tags}}


def test_contact_details_are_passed_through(monkeypatch):
    # The single most useful field we have: we cannot check price or availability,
    # but a phone number lets the passenger do exactly that.
    payload = {"elements": [hotel("Airport Plaza", phone="+972 3 000 0000",
                                  website="https://example.test")]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    result = hotels.search("TLV")[0]

    assert result["phone"] == "+972 3 000 0000"
    assert result["website"] == "https://example.test"


def test_contact_prefixed_tags_are_understood(monkeypatch):
    payload = {"elements": [hotel("Airport Plaza", **{"contact:phone": "+972 9 999 9999"})]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    assert hotels.search("TLV")[0]["phone"] == "+972 9 999 9999"


def test_street_address_is_assembled(monkeypatch):
    payload = {"elements": [hotel("Airport Plaza", **{"addr:street": "HaNasi",
                                                     "addr:housenumber": "12"})]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    assert hotels.search("TLV")[0]["address"] == "12 HaNasi"


def test_accessibility_is_reported(monkeypatch):
    payload = {"elements": [hotel("Airport Plaza", wheelchair="yes")]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    assert hotels.search("TLV")[0]["wheelchair"] == "yes"


def test_absent_details_are_omitted_entirely(monkeypatch):
    # Keys with no value would be dead weight in a prompt on a $13 budget.
    payload = {"elements": [hotel("Bare Hotel")]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    result = hotels.search("TLV")[0]

    assert set(result) == {"name", "distance_km", "area"}


def test_other_accommodation_types_are_included(monkeypatch):
    payload = {"elements": [
        {"lat": 32.02, "lon": 34.89, "tags": {"tourism": "hostel", "name": "Airport Hostel"}},
        {"lat": 32.03, "lon": 34.90, "tags": {"tourism": "guest_house", "name": "Guest House"}},
    ]}
    monkeypatch.setattr(httpx, "post", respond_with(payload))

    assert {r["name"] for r in hotels.search("TLV")} == {"Airport Hostel", "Guest House"}


def test_the_query_asks_for_every_accommodation_type(monkeypatch):
    sent = {}

    def post(url, **kwargs):
        sent["query"] = kwargs["data"]["data"]
        return httpx.Response(200, json={"elements": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    hotels.search("TLV")

    for kind in hotels.ACCOMMODATION:
        assert f'"{kind}"' in sent["query"]


# --- adaptive radius ------------------------------------------------------------
# Ramon/Eilat returned zero hotels at 12km and was written up as a coverage gap.
# It is 19km from Eilat: at 40km OpenStreetMap has 141 of them. The limitation
# was the radius, not the data.


def test_radius_widens_when_the_first_search_is_thin(monkeypatch):
    radii = []

    def post(url, **kwargs):
        query = kwargs["data"]["data"]
        radii.append(int(query.split("around:")[1].split(",")[0]))
        body = {"elements": [hotel("Far Hotel")]} if len(radii) == 3 else {"elements": []}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    results = hotels.search("TLV")

    assert radii == [12000, 25000, 40000]
    assert [r["name"] for r in results] == ["Far Hotel"]


def test_a_well_served_airport_only_searches_once(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE, calls=calls))

    hotels.search("TLV")

    assert len(calls) == 1


def test_an_explicit_radius_is_respected_and_not_widened(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", respond_with({"elements": []}, calls=calls))

    assert hotels.search("TLV", radius_km=5) == []
    assert len(calls) == 1


def test_kill_switch_makes_no_request(monkeypatch):
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "0")
    calls = []
    monkeypatch.setattr(httpx, "post", respond_with(FIXTURE, calls=calls))

    assert hotels.search("TLV") == []
    assert calls == []
