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


def test_a_thin_route_widens_the_window_until_it_has_enough(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    # NCE has 1 flight per replayed window, so it takes three 12h windows to
    # reach MIN_OPTIONS. Widening to 48h is what rescues routes like this: at
    # 24h the live route found nothing at all and the agent refused.
    results = flights.search("TLV", "NCE", AFTER)

    assert len(calls) == 3
    assert len(results) == flights.MIN_OPTIONS


def test_the_window_never_widens_past_the_ceiling(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with({"departures": []}, calls=calls))

    # A real airport code, so route validation lets it through to the API.
    assert flights.search("TLV", "BCN", AFTER) == []
    # 4 windows == 48 hours == 8 of 600 monthly units, and never more.
    assert len(calls) == flights.MAX_WINDOWS == 4


def test_a_busy_route_still_costs_one_call(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    # The 48h ceiling must not make common routes more expensive.
    flights.search("TLV", "FRA", AFTER)

    assert len(calls) == 1


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


# --- route validation, before a single unit is spent ----------------------------
# TLV->TLV widened through all four windows and spent 8 units on a route that
# cannot exist, then blamed a data outage. Both are decidable offline.


def test_an_impossible_route_costs_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    assert flights.search("TLV", "TLV", AFTER) == []
    assert calls == []


def test_an_unknown_airport_costs_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    assert flights.search("VDA", "TLV", AFTER) == []
    assert calls == []


def test_route_problem_names_the_actual_problem():
    assert "no flight to look for" in flights.route_problem("TLV", "TLV")
    assert "VDA" in flights.route_problem("VDA", "TLV")
    assert flights.route_problem("TLV", "FRA") is None


def test_route_problem_is_case_and_space_insensitive():
    assert flights.route_problem(" tlv ", "TLV") is not None
    assert flights.route_problem(" tlv ", " fra ") is None


# --- cache economics ------------------------------------------------------------


def test_the_same_route_on_the_same_day_is_cached_across_hours(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    # A live run created three cache entries for TLV->FRA on one day - hours 05,
    # 22 and 23 - and paid three times for the same schedule.
    flights.search("TLV", "FRA", datetime.fromisoformat("2026-08-10T05:00"))
    flights.search("TLV", "FRA", datetime.fromisoformat("2026-08-10T22:00"))

    assert len(calls) == 1


def test_a_different_day_is_a_different_search(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))

    flights.search("TLV", "FRA", datetime.fromisoformat("2026-08-10T22:00"))
    flights.search("TLV", "FRA", datetime.fromisoformat("2026-08-11T22:00"))

    assert len(calls) == 2


def test_a_genuinely_unserved_route_is_only_searched_once(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", respond_with({"departures": []}, calls=calls))

    flights.search("TLV", "AMS", AFTER)
    flights.search("TLV", "AMS", AFTER)

    # Four windows the first time, nothing the second: an unserved route used to
    # cost the full 8 units every time it was asked about.
    assert len(calls) == flights.MAX_WINDOWS


def test_a_failed_fetch_is_not_cached_as_empty(monkeypatch):
    # Caching a transport failure would keep a warm instance broken.
    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "get", boom)
    assert flights.search("TLV", "FRA", AFTER) == []

    calls = []
    monkeypatch.setattr(httpx, "get", respond_with(FIXTURE, calls=calls))
    assert len(flights.search("TLV", "FRA", AFTER)) == 3
    assert calls


# --- prompt size ----------------------------------------------------------------


def test_the_candidate_list_is_capped(monkeypatch):
    many = {"departures": [departure(f"LH {n}", "Expected") for n in range(40)]}
    monkeypatch.setattr(httpx, "get", respond_with(many))

    results = flights.search("TLV", "FRA", AFTER)

    # hotels.search has always capped at 8; flights had no bound at all, so a
    # busy route could inflate the prompt without limit.
    assert len(results) == flights.MAX_CANDIDATES == 12


def test_the_cap_keeps_the_earliest_departures(monkeypatch):
    def at(hour, number):
        d = departure(number, "Expected")
        d["departure"]["scheduledTime"]["local"] = f"2026-08-10 {hour:02d}:00+03:00"
        return d

    many = {"departures": [at(23, "LH LATE")] + [at(h, f"LH {h}") for h in range(13)]}
    monkeypatch.setattr(httpx, "get", respond_with(many))

    results = flights.search("TLV", "FRA", AFTER)

    assert "LH LATE" not in [r["flight"] for r in results]


# --- status filtering -----------------------------------------------------------
# Live validation recommended a flight whose status was already "Departed": the model
# flagged it in one run and ignored it in the next. Whether an option is catchable at
# all is not a judgement call, so it is decided here rather than in a prompt.


def respond_once(payload):
    """Serve `payload` for the first window, nothing for the second.

    Fewer than MIN_OPTIONS results makes search widen the window; replaying the
    same payload would double every flight and mask what is being tested.
    """
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        body = payload if len(calls) == 1 else {"departures": []}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))
    return get


def departure(number, status):
    return {
        "number": number,
        "airline": {"name": "Lufthansa", "iata": "LH"},
        "aircraft": {"model": "Airbus A320"},
        "status": status,
        "departure": {"scheduledTime": {"local": "2026-08-10 09:40+03:00"}, "terminal": "3"},
        "arrival": {"airport": {"iata": "FRA"},
                    "scheduledTime": {"local": "2026-08-10 13:05+02:00"}},
    }


@pytest.mark.parametrize("status", ["Departed", "Canceled", "Cancelled", "Arrived",
                                    "EnRoute", "Diverted", "GateClosed", "departed"])
def test_unusable_statuses_are_dropped(monkeypatch, status):
    payload = {"departures": [departure("LH 1", status), departure("LH 2", "Expected")]}
    monkeypatch.setattr(httpx, "get", respond_once(payload))

    results = flights.search("TLV", "FRA", AFTER)

    assert [r["flight"] for r in results] == ["LH 2"]


def test_delayed_flights_are_still_offered(monkeypatch):
    # A delayed flight has not left; it may be exactly what the passenger needs.
    payload = {"departures": [departure("LH 1", "Delayed")]}
    monkeypatch.setattr(httpx, "get", respond_once(payload))

    assert [r["flight"] for r in flights.search("TLV", "FRA", AFTER)] == ["LH 1"]


def test_an_unrecognised_status_is_kept(monkeypatch):
    # Deny-list, not allow-list: a new status AeroDataBox invents must not silently
    # hide every option and trigger a false "nothing could be verified" refusal.
    payload = {"departures": [departure("LH 1", "SomeNewStatus")]}
    monkeypatch.setattr(httpx, "get", respond_once(payload))

    assert [r["flight"] for r in flights.search("TLV", "FRA", AFTER)] == ["LH 1"]


def test_all_candidates_unusable_returns_empty(monkeypatch):
    payload = {"departures": [departure("LH 1", "Departed"), departure("LH 2", "Canceled")]}
    monkeypatch.setattr(httpx, "get", respond_once(payload))

    # Which is correct: the agent then refuses rather than offering a flight that left.
    assert flights.search("TLV", "FRA", AFTER) == []
