# FlightAgent + AccommodationAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two stub agents with real ones that ground their answers in live flight schedules (AeroDataBox) and real hotels (OpenStreetMap), while staying inside a 600-unit/month API quota and a $13 LLM budget.

**Architecture:** Three new pure data-fetch modules under `lib/tools/` do all network work and return small, trimmed candidate lists. Each agent injects its candidates into an existing role+one-shot prompt, makes exactly one LLM call through `lib/llm.py`, validates the result, and returns `(payload, [step])`. No tool call produces a `steps[]` entry — only the LLM call does.

**Tech Stack:** Python 3.11+, `httpx` (already in `requirements.txt`), `pytest`. No new dependencies.

**Spec:** [`../specs/2026-08-09-flight-accommodation-agents-design.md`](../specs/2026-08-09-flight-accommodation-agents-design.md)

## Global Constraints

- Module names are locked and must appear verbatim: `Supervisor`, `FlightAgent`, `AccommodationAgent`, `DocumentationAgent`.
- Every LLM call produces exactly one `steps[]` entry: `{"module", "prompt": {"system_prompt", "user_prompt"}, "response"}`. Tool/HTTP calls produce none.
- Agent signatures and payload schemas are fixed by `docs/PROJECT_PLAN.md` §1 and must not change.
- `$13` total LLM budget for the whole project — keep prompts small; the candidate trimming is the main lever.
- AeroDataBox free plan: **600 units/month, 2 units per call**. Max **2 calls per flight search**.
- Never name a booking or flight-search website in any user-facing text.
- `/api/execute` must finish well under Vercel's 300s limit.
- Tests must pass with **no API keys of any kind** configured.
- Captured fixtures are in the session scratchpad — copy, do not re-fetch (re-fetching costs quota):
  - `<scratchpad>/aerodatabox_tlv_departures.json` (118,294 bytes)
  - `<scratchpad>/overpass_tlv_hotels.json` (7,519 bytes)

  Scratchpad root: `C:\Users\GILAIR~1\AppData\Local\Temp\claude\c--Users-GILAI-ROM-OneDrive---Technion------------------------------------wingman-agent\41eecf20-70d6-49a6-b12e-7a4cc3f77640\scratchpad`

## File Structure

| File | Responsibility |
|---|---|
| `lib/tools/__init__.py` | `live_data_enabled()` — the shared network kill-switch |
| `scripts/build_airports.py` | One-off generator for the airport table; not imported by the app |
| `lib/tools/airports_data.py` | Generated: `AIRPORTS` dict, 3267 entries |
| `lib/tools/airports.py` | `lookup(iata)` over the generated table |
| `lib/tools/flights.py` | AeroDataBox fetch, route filter, trim, ISO normalisation, cache, retry |
| `lib/tools/hotels.py` | Overpass fetch, `name:en`, distance, sort, cap, cache, mirror fallback |
| `lib/agents/flight_agent.py` | *Modify:* real `run()`, regrounded prompt, validation |
| `lib/agents/accommodation_agent.py` | *Modify:* real `run()`, regrounded prompt, validation |
| `tests/conftest.py` | `fake_llm` fixture so agent/Supervisor tests need no key |
| `conftest.py` | *Modify:* autouse `WINGMAN_LIVE_DATA=0` so tests never hit the network |
| `tests/fixtures/*.json` | Captured real API responses |
| `tests/test_tools_*.py`, `tests/test_flight_agent.py`, `tests/test_accommodation_agent.py` | New tests |
| `tests/test_supervisor.py` | *Modify:* use `fake_llm`; assert date sync via `user_prompt` |

---

### Task 1: Airport coordinate table

**Files:**
- Create: `lib/tools/__init__.py`, `scripts/build_airports.py`, `lib/tools/airports_data.py` (generated), `lib/tools/airports.py`
- Test: `tests/test_tools_airports.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `lib.tools.live_data_enabled() -> bool`; `lib.tools.airports.lookup(iata: str | None) -> dict | None` returning `{"iata","name","city","country","lat","lon"}` with `lat`/`lon` as floats.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tools_airports.py`:

```python
"""The airport table is static and offline, so these are real assertions, not mocks."""

from lib.tools import airports, live_data_enabled


def test_lookup_returns_coordinates_for_a_known_airport():
    result = airports.lookup("TLV")

    assert result["iata"] == "TLV"
    assert "Ben Gurion" in result["name"]
    assert result["lat"] == 32.0114
    assert result["lon"] == 34.8867


def test_lookup_is_case_and_whitespace_insensitive():
    assert airports.lookup(" tlv ")["iata"] == "TLV"


def test_lookup_returns_none_for_nonsense():
    assert airports.lookup("ZZZZ") is None
    assert airports.lookup("") is None
    assert airports.lookup(None) is None


def test_table_covers_the_major_hubs_the_demo_uses():
    for code in ("TLV", "FRA", "VIE", "LHR", "JFK"):
        assert airports.lookup(code), f"{code} missing from the table"


def test_live_data_is_disabled_in_tests():
    # conftest sets WINGMAN_LIVE_DATA=0 so no test ever touches the quota.
    assert live_data_enabled() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools_airports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.tools'`

- [ ] **Step 3: Create the kill-switch module**

Create `lib/tools/__init__.py`:

```python
"""Data-fetch tools. Nothing in this package makes an LLM call.

`live_data_enabled()` gates every outbound request. It is off in tests and in
routine local development so neither the AeroDataBox quota (600 units/month,
2 per call) nor Overpass's fair-use limits are spent on work that does not
need real data. Unset means enabled, so production needs no extra config.
"""

import os

_OFF = ("0", "false", "no", "")


def live_data_enabled() -> bool:
    return os.environ.get("WINGMAN_LIVE_DATA", "1").strip().lower() not in _OFF
```

- [ ] **Step 4: Add the autouse test kill-switch**

Modify `conftest.py` — append to the existing file (keep the `sys.path` block):

```python
import pytest


@pytest.fixture(autouse=True)
def _no_live_data(monkeypatch):
    """No test may spend API quota or hit a public endpoint."""
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "0")
```

- [ ] **Step 5: Write the generator script**

Create `scripts/build_airports.py`:

```python
"""Regenerate lib/tools/airports_data.py from OurAirports (public domain).

Run by hand; the output is committed. Shipped as a Python module rather than
a JSON file because Vercel's Python builder traces imports, not data files —
a .json would need a vercel.json includeFiles change to survive the bundle.

    python scripts/build_airports.py
"""

import csv
import io
import pathlib

import httpx

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OUT = pathlib.Path(__file__).resolve().parents[1] / "lib" / "tools" / "airports_data.py"

HEADER = '''"""Generated by scripts/build_airports.py — do not edit by hand.

Source: OurAirports (public domain). Large and medium airports that have an
IATA code and scheduled service. Tuple order: (name, city, country, lat, lon).
"""

AIRPORTS = {
'''


def main() -> None:
    response = httpx.get(URL, timeout=180, follow_redirects=True)
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))

    keep = [
        row for row in rows
        if row["iata_code"]
        and row["type"] in ("large_airport", "medium_airport")
        and row["scheduled_service"] == "yes"
    ]

    lines = [HEADER]
    for row in sorted(keep, key=lambda r: r["iata_code"]):
        entry = (
            row["name"],
            row["municipality"] or None,
            row["iso_country"],
            round(float(row["latitude_deg"]), 4),
            round(float(row["longitude_deg"]), 4),
        )
        lines.append(f"    {row['iata_code']!r}: {entry!r},\n")
    lines.append("}\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {OUT} with {len(keep)} airports")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the generator**

Run: `python scripts/build_airports.py`
Expected: `wrote .../lib/tools/airports_data.py with 3267 airports` (count may drift slightly as OurAirports updates; anything near 3200–3400 is fine).

- [ ] **Step 7: Write the lookup module**

Create `lib/tools/airports.py`:

```python
"""IATA -> coordinates. Static, offline, no quota.

Exists so hotel search can turn "the passenger is stranded at TLV" into a
point on the map without spending an API call on geocoding.
"""

from lib.tools.airports_data import AIRPORTS


def lookup(iata: str | None) -> dict | None:
    """The airport, or None if the code is unknown."""
    if not iata:
        return None
    code = iata.strip().upper()
    row = AIRPORTS.get(code)
    if not row:
        return None
    name, city, country, lat, lon = row
    return {"iata": code, "name": name, "city": city,
            "country": country, "lat": lat, "lon": lon}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools_airports.py -v`
Expected: 5 passed.

- [ ] **Step 9: Commit**

```bash
git add lib/tools/__init__.py lib/tools/airports.py lib/tools/airports_data.py scripts/build_airports.py tests/test_tools_airports.py conftest.py
git commit -m "Add an offline airport coordinate table

Hotel search needs a point on the map for an IATA code. Geocoding it would
cost an API call per turn against a 600-unit monthly quota, so the table is
generated once from OurAirports and committed.

It ships as a Python module rather than JSON because Vercel's Python builder
traces imports, not data files — a .json would silently miss the bundle."
```

---

### Task 2: Flight search tool

**Files:**
- Create: `lib/tools/flights.py`, `tests/fixtures/aerodatabox_tlv_departures.json`
- Test: `tests/test_tools_flights.py`

**Interfaces:**
- Consumes: `lib.tools.live_data_enabled()`.
- Produces: `lib.tools.flights.search(origin: str, destination: str, after: datetime) -> list[dict]`, each dict `{"flight","airline","airline_iata","origin","destination","depart","arrive","status","aircraft","terminal"}` with `depart`/`arrive` as ISO 8601 strings. Returns `[]` on any failure.

- [ ] **Step 1: Copy the captured fixture**

```bash
mkdir -p tests/fixtures
cp "$SCRATCHPAD/aerodatabox_tlv_departures.json" tests/fixtures/
```

Do not re-fetch — a fresh call costs 2 of the 600 monthly units.

- [ ] **Step 2: Write the failing test**

Create `tests/test_tools_flights.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_tools_flights.py -v`
Expected: FAIL with `ImportError: cannot import name 'flights'`

- [ ] **Step 4: Write the implementation**

Create `lib/tools/flights.py`:

```python
"""Onward flight candidates from AeroDataBox. No LLM calls in this module.

Filtering and trimming happen here on purpose. A raw airport window is about
118KB and 188 departures, of which a single route's flights trim to roughly
630 bytes. Sending the raw response to the model instead would dominate the
project's $13 budget for no gain.

Free plan economics: 600 units/month, 2 units per call. A search is capped at
two calls, and `live_data_enabled()` keeps development off the meter entirely.
"""

import os
import time
from datetime import datetime, timedelta

import httpx

from lib.tools import live_data_enabled

WINDOW_HOURS = 12          # AeroDataBox caps a departures range at 12 hours
MIN_OPTIONS = 3            # below this it is worth one more call
MAX_WINDOWS = 2            # hard ceiling: 2 calls == 4 units per search
TIMEOUT_SECONDS = 60
RETRY_PAUSE_SECONDS = 2.0  # the BASIC plan rate-limits per second

DEFAULT_HOST = "aerodatabox.p.rapidapi.com"

_cache: dict[tuple, list[dict]] = {}


def _iso(stamp: str | None) -> str | None:
    """AeroDataBox sends '2026-08-10 06:05+03:00'; ISO 8601 wants the T."""
    if not stamp:
        return None
    return stamp.replace(" ", "T", 1)


def _trim(departure: dict) -> dict:
    """One departure, reduced to what the model actually needs."""
    dep = departure.get("departure") or {}
    arr = departure.get("arrival") or {}
    airline = departure.get("airline") or {}
    return {
        "flight": departure.get("number"),
        "airline": airline.get("name"),
        "airline_iata": airline.get("iata"),
        "destination": (arr.get("airport") or {}).get("iata"),
        "depart": _iso((dep.get("scheduledTime") or {}).get("local")),
        "arrive": _iso((arr.get("scheduledTime") or {}).get("local")),
        "status": departure.get("status"),
        "aircraft": (departure.get("aircraft") or {}).get("model"),
        "terminal": dep.get("terminal"),
    }


def _window(host: str, key: str, iata: str, start: datetime, end: datetime) -> list[dict]:
    url = (f"https://{host}/flights/airports/iata/{iata}"
           f"/{start:%Y-%m-%dT%H:%M}/{end:%Y-%m-%dT%H:%M}")
    params = {"withLeg": "true", "direction": "Departure", "withCancelled": "true",
              "withCodeshared": "false", "withCargo": "false",
              "withPrivate": "false", "withLocation": "false"}
    headers = {"x-rapidapi-key": key, "x-rapidapi-host": host}

    response = httpx.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
    if response.status_code == 429:
        time.sleep(RETRY_PAUSE_SECONDS)
        response = httpx.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("departures") or []


def search(origin: str, destination: str, after: datetime) -> list[dict]:
    """Real flights origin -> destination departing after `after`.

    Returns [] on any failure or when live data is switched off: the agent
    degrades to reasoning unaided rather than losing the passenger's turn.
    """
    if not (origin and destination and live_data_enabled()):
        return []
    key = os.environ.get("AERODATABOX_API_KEY")
    if not key:
        return []
    host = os.environ.get("AERODATABOX_API_HOST") or DEFAULT_HOST

    origin, destination = origin.strip().upper(), destination.strip().upper()
    cache_key = (origin, destination, after.strftime("%Y-%m-%dT%H"))
    if cache_key in _cache:
        return _cache[cache_key]

    found: list[dict] = []
    start = after
    for _ in range(MAX_WINDOWS):
        end = start + timedelta(hours=WINDOW_HOURS)
        try:
            departures = _window(host, key, origin, start, end)
        except (httpx.HTTPError, ValueError, KeyError):
            break
        for departure in departures:
            option = _trim(departure)
            if option["destination"] == destination and option["depart"]:
                option["origin"] = origin
                found.append(option)
        if len(found) >= MIN_OPTIONS:
            break
        start = end

    if found:
        _cache[cache_key] = found
    return found
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools_flights.py -v`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add lib/tools/flights.py tests/test_tools_flights.py tests/fixtures/aerodatabox_tlv_departures.json
git commit -m "Fetch real onward flights from AeroDataBox

Filtering and trimming happen in the tool rather than the prompt: a raw
airport window is 118KB and 188 departures, of which the three that fly the
passenger's route come to 630 bytes.

Timestamps arrive space-separated and are normalised to ISO 8601 here,
because the Supervisor's date sync parses them and a bad one costs the
passenger the hotel as well as the flight."
```

---

### Task 3: Hotel search tool

**Files:**
- Create: `lib/tools/hotels.py`, `tests/fixtures/overpass_tlv_hotels.json`
- Test: `tests/test_tools_hotels.py`

**Interfaces:**
- Consumes: `lib.tools.live_data_enabled()`, `lib.tools.airports.lookup`.
- Produces: `lib.tools.hotels.search(iata: str, radius_km: float = 12.0) -> list[dict]`, each dict `{"name","distance_km","stars","breakfast","area"}`, sorted nearest-first, at most 8. Returns `[]` on any failure.

- [ ] **Step 1: Copy the captured fixture**

```bash
cp "$SCRATCHPAD/overpass_tlv_hotels.json" tests/fixtures/
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_tools_hotels.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_tools_hotels.py -v`
Expected: FAIL with `ImportError: cannot import name 'hotels'`

- [ ] **Step 4: Write the implementation**

Create `lib/tools/hotels.py`:

```python
"""Hotels near an airport, from OpenStreetMap via Overpass. No key, no LLM call.

OSM knows a hotel's name and where it is, and usually nothing else — a real
12km query around TLV returned 20 hotels of which one had a star rating and
almost none had an address or website. So this returns name, distance and
whatever tags happen to exist. Price, availability and meals are not knowable
here and must not be invented by the agent that consumes this.

Distance is the one thing we can state exactly, and it is also what matters
most to a passenger who has to make a morning departure.
"""

import math
import httpx

from lib.tools import airports, live_data_enabled

# The public instance intermittently returns 504; the mirror is the fallback.
MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT = "wingman-agent/0.1 (course project)"
MAX_RESULTS = 8
TIMEOUT_SECONDS = 60
EARTH_RADIUS_KM = 6371

_cache: dict[tuple, list[dict]] = {}


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return round(EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a)), 1)


def _query(lat: float, lon: float, radius_m: int) -> str:
    return (
        f'[out:json][timeout:30];'
        f'(node["tourism"="hotel"](around:{radius_m},{lat},{lon});'
        f' way["tourism"="hotel"](around:{radius_m},{lat},{lon}););'
        f'out center tags;'
    )


def _fetch(query: str) -> list[dict]:
    """Try each mirror in turn. Raises only if all of them fail."""
    last_error: Exception | None = None
    for url in MIRRORS:
        try:
            response = httpx.post(url, data={"data": query},
                                  headers={"User-Agent": USER_AGENT},
                                  timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json().get("elements") or []
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            last_error = exc
    raise last_error if last_error else RuntimeError("no Overpass mirror configured")


def search(iata: str, radius_km: float = 12.0) -> list[dict]:
    """Real hotels near the airport, nearest first. [] on any failure."""
    if not (iata and live_data_enabled()):
        return []
    airport = airports.lookup(iata)
    if not airport:
        return []

    cache_key = (airport["iata"], radius_km)
    if cache_key in _cache:
        return _cache[cache_key]

    try:
        elements = _fetch(_query(airport["lat"], airport["lon"], int(radius_km * 1000)))
    except Exception:
        return []

    found: list[dict] = []
    for element in elements:
        tags = element.get("tags") or {}
        centre = element.get("center") or {}
        name = tags.get("name:en") or tags.get("name")   # OSM names are often local-language
        lat = element.get("lat", centre.get("lat"))
        lon = element.get("lon", centre.get("lon"))
        if not (name and lat is not None and lon is not None):
            continue
        found.append({
            "name": name,
            "distance_km": _distance_km(airport["lat"], airport["lon"], lat, lon),
            "stars": tags.get("stars"),
            "breakfast": tags.get("breakfast"),
            "area": tags.get("addr:city") or airport["city"],
        })

    found.sort(key=lambda hotel: hotel["distance_km"])
    found = found[:MAX_RESULTS]
    if found:
        _cache[cache_key] = found
    return found
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools_hotels.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add lib/tools/hotels.py tests/test_tools_hotels.py tests/fixtures/overpass_tlv_hotels.json
git commit -m "Find real hotels near the airport from OpenStreetMap

Keyless and unmetered, so the hotel half can never be blocked by a quota.
OSM knows a hotel's name and position and little else — one of twenty near
TLV had a star rating — so this returns exactly that, and distance, which is
what a passenger making a morning departure actually needs.

The public Overpass instance returned a 504 during development, hence the
mirror fallback."
```

---

### Task 4: FlightAgent

**Files:**
- Modify: `lib/agents/flight_agent.py` (replace `SYSTEM_PROMPT`, `_user_prompt`, `run`; delete `IS_STUB`)
- Create: `tests/conftest.py`, `tests/test_flight_agent.py`
- Modify: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `lib.tools.flights.search`, `lib.llm.call`, `lib.llm.LLMError`.
- Produces: `flight_agent.run(request: dict, history: list[dict]) -> tuple[dict, list[dict]]` — unchanged signature, payload per `PROJECT_PLAN.md` §1, exactly one step.

- [ ] **Step 1: Write the shared fake-LLM fixture**

Create `tests/conftest.py`:

```python
"""A stand-in for LLMod so agent and Supervisor tests run with no API key.

Deliberately not autouse: tests/test_llm.py exercises lib.llm.call itself and
must see the real thing. Person C will want this too when DocumentationAgent
stops being a stub.
"""

from datetime import datetime, timedelta

import pytest

from lib import llm
from lib.steps import make_step


def _flight_response(user_prompt: str) -> dict:
    depart = (datetime.now() + timedelta(days=1)).replace(
        hour=9, minute=40, second=0, microsecond=0)
    return {
        "options": [{
            "id": "F1", "airline": "Lufthansa", "flight_number": "LH 687",
            "origin": "TLV", "destination": "FRA",
            "depart": depart.isoformat(),
            "arrive": (depart + timedelta(hours=3, minutes=25)).isoformat(),
            "stops": 0,
            "fare_conditions": "Rebooking terms come from your Contract of Carriage.",
            "notes": "Earliest nonstop.",
        }],
        "recommended_id": "F1",
    }


def _accommodation_response(user_prompt: str) -> dict:
    fields = {}
    for line in user_prompt.splitlines():
        if ":" in line:
            label, _, value = line.partition(":")
            fields[label.strip()] = value.strip()
    return {
        "options": [{
            "id": "H1", "name": "Airport Plaza", "area": "8 minutes from the terminal",
            "check_in": fields.get("Check in", "2026-08-09"),
            "check_out": fields.get("Check out", "2026-08-10"),
            "nights": int(fields.get("Nights", "1") or 1),
            "price_estimate": "EUR 120 total (estimate)",
            "meals_included": False,
            "notes": "Meals not confirmed — check at the desk.",
        }],
        "recommended_id": "H1",
    }


RESPONSES = {
    "FlightAgent": _flight_response,
    "AccommodationAgent": _accommodation_response,
}


@pytest.fixture
def fake_llm(monkeypatch):
    def call(module, system_prompt, user_prompt, *, expect_json=False):
        payload = RESPONSES[module](user_prompt)
        return payload, make_step(module, system_prompt, user_prompt, payload)

    monkeypatch.setattr(llm, "call", call)
    return call
```

- [ ] **Step 2: Write the failing FlightAgent test**

Create `tests/test_flight_agent.py`:

```python
"""FlightAgent: grounding, validation and the trace it leaves behind."""

from datetime import datetime, timedelta

import pytest

from lib import llm
from lib.agents import flight_agent
from lib.steps import make_step
from lib.tools import flights

REQUEST = {
    "airline": "LH", "flight_number": "LH318", "origin": "TLV", "destination": "FRA",
    "disruption": "cancelled", "stranded_at": "TLV", "party_size": 2,
    "arrive_by": None, "needs": ["flight"],
    "local_now": "2026-08-09T22:15:00",
}

CANDIDATE = {
    "flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH",
    "origin": "TLV", "destination": "FRA",
    "depart": "2026-08-10T16:30+03:00", "arrive": "2026-08-10T20:10+02:00",
    "status": "Expected", "aircraft": "Airbus A320", "terminal": "3",
}


def fake_call(payload):
    def call(module, system_prompt, user_prompt, *, expect_json=False):
        return payload, make_step(module, system_prompt, user_prompt, payload)
    return call


def good_payload(depart="2026-08-10T16:30+03:00", arrive="2026-08-10T20:10+02:00",
                 recommended="F1"):
    return {
        "options": [{
            "id": "F1", "airline": "Lufthansa", "flight_number": "LH 687",
            "origin": "TLV", "destination": "FRA", "depart": depart, "arrive": arrive,
            "stops": 0, "fare_conditions": "See your Contract of Carriage.",
            "notes": "Nonstop.",
        }],
        "recommended_id": recommended,
    }


def test_real_candidates_reach_the_prompt(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [])

    user_prompt = steps[0]["prompt"]["user_prompt"]
    assert "LH 687" in user_prompt
    assert "2026-08-10T16:30+03:00" in user_prompt


def test_one_llm_call_produces_exactly_one_step(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [])

    assert len(steps) == 1
    assert steps[0]["module"] == "FlightAgent"


def test_degraded_mode_tells_the_model_to_label_its_output(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [])

    assert "Illustrative" in steps[0]["prompt"]["user_prompt"]


def test_options_with_unparseable_times_are_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({
        "id": "F2", "airline": "El Al", "flight_number": "LY 357",
        "origin": "TLV", "destination": "FRA",
        "depart": "tomorrow morning", "arrive": "later", "stops": 0,
        "fare_conditions": "", "notes": "",
    })
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    result, _ = flight_agent.run(REQUEST, [])

    # The Supervisor calls fromisoformat on depart; a bad one costs the hotel too.
    assert [o["id"] for o in result["options"]] == ["F1"]


def test_a_dangling_recommended_id_falls_back_to_the_first_option(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(recommended="F9")))

    result, _ = flight_agent.run(REQUEST, [])

    assert result["recommended_id"] == "F1"


def test_no_usable_option_raises_but_keeps_the_step(monkeypatch):
    payload = {"options": [{"id": "F1", "depart": "soon", "arrive": "later"}],
               "recommended_id": "F1"}
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    with pytest.raises(llm.LLMError) as caught:
        flight_agent.run(REQUEST, [])

    # The spec wants every call that happened in the trace, failures included.
    assert len(caught.value.steps) == 1
    assert caught.value.steps[0]["module"] == "FlightAgent"


def test_history_reaches_the_prompt(monkeypatch):
    monkeypatch.setattr(flights, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = flight_agent.run(REQUEST, [{"prompt": "what about earlier?",
                                           "response": "checking"}])

    assert "what about earlier?" in steps[0]["prompt"]["user_prompt"]


def test_the_stub_flag_is_gone():
    assert not hasattr(flight_agent, "IS_STUB")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_flight_agent.py -v`
Expected: FAIL — `test_the_stub_flag_is_gone` fails and the grounding tests fail because `run()` never calls `flights.search`.

- [ ] **Step 4: Replace the prompt**

In `lib/agents/flight_agent.py`, replace the module docstring, `IS_STUB` and `SYSTEM_PROMPT` with:

```python
"""FlightAgent — onward options after a disruption, grounded in real schedules.

Pattern: role prompt + one-shot example (`docs/PROJECT_PLAN.md` §1). The example
now shows the model *choosing from* verified candidates rather than inventing
options, because `lib/tools/flights.py` supplies real departures.

Scope is deliberately narrow (design doc D5): schedule data cannot answer what
a ticket allows, so baggage, fare and compensation questions are deferred to
DocumentationAgent, which can cite the Contract of Carriage clause.
"""

import json
from datetime import datetime

from lib import llm
from lib.tools import flights

MODULE = "FlightAgent"

SYSTEM_PROMPT = """You choose onward flights for a passenger whose flight has just been disrupted.

You are given a list of REAL flights, already verified as departing from the passenger's airport to
their destination. Choose only from that list. Never invent a flight, and never alter a time, flight
number or airline.

You do not book, hold or pay for anything — you propose options the passenger acts on themselves.
Include flights on other airlines: the passenger's Contract of Carriage may entitle them to be
rebooked on a competitor, so do not silently drop them. Never name a booking or flight-search website.

Return at most three options. Prefer the earliest arrival that meets any deadline given; where they
differ meaningfully, include one that is gentler on conditions.

You have schedule data ONLY. You do not know fares, seat availability, baggage allowances, or what
the airline owes this passenger. Never state a price as fact, and never state a baggage or
compensation rule — those come from the passenger's Contract of Carriage and are handled elsewhere
in the plan. Use "fare_conditions" to point there, not to assert terms.

Only direct flights are available to you. If none suits, say so in "notes" rather than inventing a
connection.

Return a JSON object only, no prose:
{"options": [{"id", "airline", "flight_number", "origin", "destination", "depart", "arrive",
              "stops", "fare_conditions", "notes"}],
 "recommended_id": "<id>"}

"depart" and "arrive" must be copied exactly from the candidate you chose, in ISO 8601 local time.
They decide which nights the passenger is stranded, so an altered time books the wrong hotel.

Example
-------
Candidates:
[{"flight": "LY 357", "airline": "El Al", "airline_iata": "LY", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T06:05+03:00", "arrive": "2026-08-10T09:40+02:00", "status": "Expected", "aircraft": "Boeing 737-900", "terminal": "3"},
 {"flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T16:30+03:00", "arrive": "2026-08-10T20:10+02:00", "status": "Expected", "aircraft": "Airbus A320", "terminal": "3"}]
Request: LH318 TLV->FRA cancelled, 2 adults, must arrive by the evening of 2026-08-10, local time now 2026-08-09T22:15.
Response:
{"options": [{"id": "F1", "airline": "El Al", "flight_number": "LY 357", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T06:05+03:00", "arrive": "2026-08-10T09:40+02:00", "stops": 0, "fare_conditions": "A different airline from the one that cancelled. Whether your ticket can be moved across is set by your Contract of Carriage - see the entitlements section.", "notes": "Earliest arrival, and it clears your deadline with the day to spare."}, {"id": "F2", "airline": "Lufthansa", "flight_number": "LH 687", "origin": "TLV", "destination": "FRA", "depart": "2026-08-10T16:30+03:00", "arrive": "2026-08-10T20:10+02:00", "stops": 0, "fare_conditions": "Same airline as the cancelled flight, which is usually the simplest rebooking to arrange at the desk.", "notes": "Later, but stays with your original carrier and still lands inside your deadline."}],
 "recommended_id": "F1"}
"""

DEGRADED_NOTE = (
    "No live schedule data was available for this route. Propose plausible options from your own "
    "knowledge, and put 'Illustrative option - not live availability.' in the notes of every one."
)
```

- [ ] **Step 5: Replace `_user_prompt`, add validation, replace `run`**

Replace everything from `def _user_prompt` to the end of the file with:

```python
def _user_prompt(request: dict, history: list[dict], candidates: list[dict]) -> str:
    lines = [
        f"Flight: {request.get('airline')} {request.get('flight_number')}",
        f"Route: {request.get('origin')} -> {request.get('destination')}",
        f"What happened: {request.get('disruption')}",
        f"Passenger is at: {request.get('stranded_at')}",
        f"Party size: {request.get('party_size')}",
        f"Must arrive by: {request.get('arrive_by') or 'as soon as possible'}",
        f"Local time now: {request.get('local_now')}",
        "",
    ]
    if candidates:
        lines.append("Candidates (real, verified departures - choose only from these):")
        lines.append(json.dumps(candidates, ensure_ascii=False))
    else:
        lines.append(DEGRADED_NOTE)

    if history:
        lines.append("")
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"  passenger: {turn['prompt']}")
            lines.append(f"  you: {turn['response']}")
    return "\n".join(lines)


def _validate(payload: dict) -> dict:
    """Drop anything the Supervisor cannot use.

    `supervisor._stay_window` calls `datetime.fromisoformat` on "depart", so an
    unparseable time costs the passenger the hotel as well as the flight.
    """
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")

    usable = []
    for option in payload.get("options") or []:
        if not isinstance(option, dict) or not option.get("id"):
            continue
        try:
            datetime.fromisoformat(option.get("depart") or "")
            datetime.fromisoformat(option.get("arrive") or "")
        except (TypeError, ValueError):
            continue
        usable.append(option)

    if not usable:
        raise ValueError("no option came back with a usable departure time")

    payload["options"] = usable
    if payload.get("recommended_id") not in {o["id"] for o in usable}:
        payload["recommended_id"] = usable[0]["id"]
    return payload


def run(request: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). See `docs/PROJECT_PLAN.md` §1 for both shapes."""
    after = datetime.fromisoformat(request["local_now"])
    candidates = flights.search(request.get("origin"), request.get("destination"), after)

    payload, step = llm.call(
        MODULE, SYSTEM_PROMPT, _user_prompt(request, history, candidates), expect_json=True
    )
    try:
        return _validate(payload), [step]
    except ValueError as exc:
        # The call happened, so the trace keeps it even though it was unusable.
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
```

- [ ] **Step 6: Run FlightAgent tests to verify they pass**

Run: `python -m pytest tests/test_flight_agent.py -v`
Expected: 8 passed.

- [ ] **Step 7: Repair the Supervisor tests**

`tests/test_supervisor.py` currently exercises the stubs. Two changes:

Add below the imports:

```python
pytestmark = pytest.mark.usefixtures("fake_llm")
```

Replace `test_accommodation_is_booked_for_the_nights_the_flight_implies` with a version that
asserts the Supervisor's date sync directly, rather than an agent echoing it back:

```python
def test_accommodation_is_booked_for_the_nights_the_flight_implies():
    _, steps = supervisor.run(COMPLETE, [])
    stay = next(s for s in steps if s["module"] == "AccommodationAgent")
    asked = stay["prompt"]["user_prompt"]

    # The fake flight leaves 09:40 tomorrow, so: tonight, one night.
    assert f"Check in: {date.today().isoformat()}" in asked
    assert f"Check out: {(date.today() + timedelta(days=1)).isoformat()}" in asked
    assert "Nights: 1" in asked
```

- [ ] **Step 8: Run the whole suite**

Run: `python -m pytest -v`
Expected: all pass. `test_documentation_agent_emits_three_steps_for_its_reflection_loop` still passes — DocumentationAgent is Person C's stub and is untouched.

- [ ] **Step 9: Commit**

```bash
git add lib/agents/flight_agent.py tests/test_flight_agent.py tests/conftest.py tests/test_supervisor.py
git commit -m "Ground FlightAgent in real departures

The one-shot example now shows the model choosing from verified candidates
instead of inventing them, and the prompt refuses baggage and fare questions
outright: schedule data cannot answer them, and DocumentationAgent can, with
the clause cited.

Validation drops options whose times will not parse, because the Supervisor's
date sync reads depart and a bad one would cost the hotel as well.

Touches tests/test_supervisor.py, which asserted against the old stub. The new
fake_llm fixture keeps it running with no API key; Person C will want it too."
```

---

### Task 5: AccommodationAgent

**Files:**
- Modify: `lib/agents/accommodation_agent.py`
- Test: `tests/test_accommodation_agent.py`

**Interfaces:**
- Consumes: `lib.tools.hotels.search`, `lib.llm.call`, `lib.llm.LLMError`.
- Produces: `accommodation_agent.run(request: dict, stay_window: dict, history: list[dict]) -> tuple[dict, list[dict]]` — unchanged signature, exactly one step.

- [ ] **Step 1: Write the failing test**

Create `tests/test_accommodation_agent.py`:

```python
"""AccommodationAgent: grounding, the fixed stay window, and honest meals."""

import pytest

from lib import llm
from lib.agents import accommodation_agent
from lib.steps import make_step
from lib.tools import hotels

REQUEST = {"stranded_at": "TLV", "party_size": 2, "local_now": "2026-08-09T22:15:00"}
WINDOW = {"check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1,
          "guests": 2, "departs": "2026-08-10T09:40:00"}
CANDIDATE = {"name": "Airport Plaza", "distance_km": 2.4, "stars": "4",
             "breakfast": None, "area": "Lod"}


def fake_call(payload):
    def call(module, system_prompt, user_prompt, *, expect_json=False):
        return payload, make_step(module, system_prompt, user_prompt, payload)
    return call


def good_payload(check_in="2026-08-09", check_out="2026-08-10", recommended="H1"):
    return {
        "options": [{
            "id": "H1", "name": "Airport Plaza", "area": "2.4 km from the terminal",
            "check_in": check_in, "check_out": check_out, "nights": 1,
            "price_estimate": "EUR 120 total (estimate)", "meals_included": False,
            "notes": "Meals not confirmed - check at the desk.",
        }],
        "recommended_id": recommended,
    }


def test_real_hotels_reach_the_prompt(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])

    assert "Airport Plaza" in steps[0]["prompt"]["user_prompt"]


def test_the_stay_window_is_stated_as_non_negotiable(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])
    user_prompt = steps[0]["prompt"]["user_prompt"]

    assert "Check in: 2026-08-09" in user_prompt
    assert "Check out: 2026-08-10" in user_prompt
    assert "Nights: 1" in user_prompt


def test_one_llm_call_produces_exactly_one_step(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])

    assert len(steps) == 1
    assert steps[0]["module"] == "AccommodationAgent"


def test_degraded_mode_labels_its_output(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "call", fake_call(good_payload()))

    _, steps = accommodation_agent.run(REQUEST, WINDOW, [])

    assert "Illustrative" in steps[0]["prompt"]["user_prompt"]


def test_options_on_the_wrong_dates_are_dropped(monkeypatch):
    payload = good_payload()
    payload["options"].append({
        "id": "H2", "name": "Wrong Nights Inn", "area": "town",
        "check_in": "2026-08-12", "check_out": "2026-08-13", "nights": 1,
        "price_estimate": "EUR 60", "meals_included": False, "notes": "",
    })
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(payload))

    result, _ = accommodation_agent.run(REQUEST, WINDOW, [])

    # The nights come from the flight that was found; they are not negotiable.
    assert [o["id"] for o in result["options"]] == ["H1"]


def test_unparseable_dates_are_dropped(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(check_in="tonight")))

    with pytest.raises(llm.LLMError) as caught:
        accommodation_agent.run(REQUEST, WINDOW, [])

    assert len(caught.value.steps) == 1


def test_a_dangling_recommended_id_falls_back(monkeypatch):
    monkeypatch.setattr(hotels, "search", lambda *a, **k: [CANDIDATE])
    monkeypatch.setattr(llm, "call", fake_call(good_payload(recommended="H9")))

    result, _ = accommodation_agent.run(REQUEST, WINDOW, [])

    assert result["recommended_id"] == "H1"


def test_the_stub_flag_is_gone():
    assert not hasattr(accommodation_agent, "IS_STUB")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_accommodation_agent.py -v`
Expected: FAIL — `test_the_stub_flag_is_gone` fails, grounding tests fail.

- [ ] **Step 3: Replace the prompt**

In `lib/agents/accommodation_agent.py`, replace the docstring, `IS_STUB` and `SYSTEM_PROMPT` with:

```python
"""AccommodationAgent — stays for exactly the nights the new itinerary strands the passenger.

Pattern: role prompt + one-shot example (`docs/PROJECT_PLAN.md` §1), now choosing
from real hotels supplied by `lib/tools/hotels.py`.

`stay_window` is the point of the date sync: the Supervisor derives the nights
from the flight FlightAgent actually found, so this agent never guesses them.

OpenStreetMap knows a hotel's name and where it is and rarely anything else, so
price and meals are estimates and must be labelled as such (design doc D8).
"""

import json
from datetime import date

from lib import llm
from lib.tools import hotels

MODULE = "AccommodationAgent"

SYSTEM_PROMPT = """You find somewhere for a stranded air passenger to sleep, for a fixed set of nights.

You are given a list of REAL hotels near the airport, with the exact distance to the terminal.
Choose only from that list and never invent a property or change its name.

The nights are given to you and are NOT negotiable — they come from the replacement flight the
passenger is taking. Every option must use exactly the check-in and check-out dates you are given.

You do not book, hold or pay for anything. You propose options the passenger acts on themselves.
Never name a booking website.

Prioritise, in order: close enough to reach the airport for the departure, then anything the data
tells you about the property.

You do NOT know prices, availability, or whether meals are included — that information is not in
your source. "price_estimate" must read as an estimate. Set "meals_included" to true only if the
data explicitly says so; otherwise set it false and say in "notes" that meals were not confirmed
and to check at the desk. Never assert a fact about a real named business that you were not given.

Return a JSON object only, no prose:
{"options": [{"id", "name", "area", "check_in", "check_out", "nights",
              "price_estimate", "meals_included", "notes"}],
 "recommended_id": "<id>"}

Example
-------
Hotels: [{"name": "Airport Plaza", "distance_km": 2.4, "stars": "4", "breakfast": null, "area": "Lod"},
         {"name": "City Central Inn", "distance_km": 11.2, "stars": null, "breakfast": null, "area": "Tel Aviv"}]
Request: 2 guests stranded at TLV, check in 2026-08-09, check out 2026-08-10, 1 night, onward flight departs 2026-08-10T09:40.
Response:
{"options": [{"id": "H1", "name": "Airport Plaza", "area": "2.4 km from the terminal, Lod", "check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1, "price_estimate": "Roughly EUR 110-140 for the night (estimate - not a quoted price)", "meals_included": false, "notes": "Closest to the terminal, which matters for an 09:40 departure. Meals were not confirmed - ask at the desk, and keep the receipt if you are claiming care costs back."}, {"id": "H2", "name": "City Central Inn", "area": "11.2 km from the terminal, Tel Aviv", "check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1, "price_estimate": "Roughly EUR 80-100 for the night (estimate - not a quoted price)", "meals_included": false, "notes": "Cheaper but 11 km out, so leave a clear hour to get back for the flight. Meals not confirmed."}],
 "recommended_id": "H1"}
"""

DEGRADED_NOTE = (
    "No live hotel data was available for this airport. Propose plausible options from your own "
    "knowledge, and put 'Illustrative option - not live availability.' in the notes of every one."
)
```

- [ ] **Step 4: Replace `_user_prompt`, add validation, replace `run`**

Replace everything from `def _user_prompt` to the end of the file with:

```python
def _user_prompt(request: dict, stay_window: dict, history: list[dict],
                 candidates: list[dict]) -> str:
    lines = [
        f"Passenger is stranded at: {request.get('stranded_at')}",
        f"Guests: {stay_window.get('guests')}",
        f"Check in: {stay_window.get('check_in')}",
        f"Check out: {stay_window.get('check_out')}",
        f"Nights: {stay_window.get('nights')}",
        f"Onward flight departs: {stay_window.get('departs')}",
        "",
    ]
    if candidates:
        lines.append("Hotels (real, near the airport - choose only from these):")
        lines.append(json.dumps(candidates, ensure_ascii=False))
    else:
        lines.append(DEGRADED_NOTE)

    if history:
        lines.append("")
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"  passenger: {turn['prompt']}")
            lines.append(f"  you: {turn['response']}")
    return "\n".join(lines)


def _validate(payload: dict, stay_window: dict) -> dict:
    """Drop options that do not match the nights the Supervisor derived.

    Booking the wrong nights is the one failure that leaves the passenger worse
    off than no answer at all, so a mismatch is dropped rather than reported.
    """
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")

    wanted_in = stay_window.get("check_in")
    wanted_out = stay_window.get("check_out")

    usable = []
    for option in payload.get("options") or []:
        if not isinstance(option, dict) or not option.get("id"):
            continue
        try:
            date.fromisoformat(option.get("check_in") or "")
            date.fromisoformat(option.get("check_out") or "")
        except (TypeError, ValueError):
            continue
        if option["check_in"] != wanted_in or option["check_out"] != wanted_out:
            continue
        usable.append(option)

    if not usable:
        raise ValueError("no option came back on the nights the flight implies")

    payload["options"] = usable
    if payload.get("recommended_id") not in {o["id"] for o in usable}:
        payload["recommended_id"] = usable[0]["id"]
    return payload


def run(request: dict, stay_window: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). See `docs/PROJECT_PLAN.md` §1 for both shapes."""
    candidates = hotels.search(request.get("stranded_at"))

    payload, step = llm.call(
        MODULE, SYSTEM_PROMPT,
        _user_prompt(request, stay_window, history, candidates), expect_json=True
    )
    try:
        return _validate(payload, stay_window), [step]
    except ValueError as exc:
        raise llm.LLMError(f"{MODULE}: {exc}", steps=[step]) from exc
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_accommodation_agent.py -v`
Expected: 8 passed.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add lib/agents/accommodation_agent.py tests/test_accommodation_agent.py
git commit -m "Ground AccommodationAgent in real hotels near the airport

Chooses from OpenStreetMap properties with exact distances to the terminal,
which is what actually decides whether a passenger makes a morning departure.

OSM does not know prices or meals, so the prompt forbids asserting either and
meals_included stays false with the uncertainty spelled out in notes rather
than invented about a real named business.

Options on the wrong nights are dropped: the window comes from the flight that
was found, and booking the wrong dates is worse than no answer."
```

---

### Task 6: Documentation and configuration

**Files:**
- Modify: `.env.example`, `README.md`, `CLAUDE.md`, `docs/PROJECT_PLAN.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Add the new environment variables**

Replace `.env.example` with:

```
SUPABASE_URL=
SUPABASE_KEY=
LLMOD_API_KEY=
LLMOD_API_BASE=

# AeroDataBox via RapidAPI — real flight schedules for FlightAgent.
# Free plan: 600 units/month, 2 units per call.
AERODATABOX_API_KEY=
AERODATABOX_API_HOST=aerodatabox.p.rapidapi.com

# Set to 0 for local development so neither the AeroDataBox quota nor
# Overpass's fair-use limits are spent on work that does not need real data.
# Unset means enabled, so production needs no extra configuration.
WINGMAN_LIVE_DATA=1
```

- [ ] **Step 2: Document local development in the README**

Add a section to `README.md`:

```markdown
### Live data

`FlightAgent` uses AeroDataBox (free plan: **600 units/month, 2 units per call**) and
`AccommodationAgent` uses OpenStreetMap's Overpass API (keyless).

Set `WINGMAN_LIVE_DATA=0` in your local `.env` while developing. Both tools then make no
network requests at all and the agents degrade to reasoning unaided, clearly labelled. The
test suite forces this off, so `pytest` never spends quota.

Check remaining quota (costs nothing):

    curl -s -H "x-rapidapi-key: $AERODATABOX_API_KEY" \
         -H "x-rapidapi-host: aerodatabox.p.rapidapi.com" \
         -D - -o /dev/null \
         https://aerodatabox.p.rapidapi.com/subscriptions/balance | grep api-units
```

- [ ] **Step 3: Correct CLAUDE.md**

In `CLAUDE.md` §3, replace the FlightAgent bullet's "stays conversational for follow-ups
(compare options, check terms)" with:

```markdown
- **Flight agent** — pattern: **role prompt + one-shot example**, choosing from real departures
  supplied by `lib/tools/flights.py` (AeroDataBox). Finds alternative flights across airlines and
  stays conversational for follow-ups (compare options, ask about timings, request an earlier
  departure). It has **schedule data only**: baggage, fare and entitlement questions are deferred
  to `DocumentationAgent`, which can cite the Contract of Carriage clause.
```

In §4, add under the existing data sources:

```markdown
- **Flight schedules** — AeroDataBox (RapidAPI free plan, 600 units/month). Real departures,
  filtered to the passenger's route in `lib/tools/flights.py`. No free source of fares or seat
  availability exists as of 8/2026 (Amadeus Self-Service was decommissioned 17/7/2026), so prices
  are LLM estimates and must be labelled as such.
- **Hotels** — OpenStreetMap via Overpass (keyless). Real properties with exact distance from the
  airport. No prices, availability or meals.
```

In §6, replace the open question "Flights/hotels data source (mock vs. free-tier real API)" with:

```markdown
- **Resolved (9/8/2026):** flights from AeroDataBox, hotels from OpenStreetMap Overpass. Amadeus
  Self-Service — the obvious choice — was decommissioned 17/7/2026. Prices and seat availability
  have no free source and are LLM estimates, labelled as such. Full rationale and the measured
  API facts: `docs/superpowers/specs/2026-08-09-flight-accommodation-agents-design.md`.
```

- [ ] **Step 4: Update PROJECT_PLAN.md**

In §0, replace the unresolved data-source line with:

```markdown
- [x] **Flights/hotels data source:** AeroDataBox (flights) + OpenStreetMap Overpass (hotels),
  decided 9/8/2026. See §6. No booking site is named in any user-facing text.
```

In §2 Phase 2, mark both agents done:

```markdown
- [x] **FlightAgent:** real departures from AeroDataBox, filtered to the route and trimmed in
  `lib/tools/flights.py`; role+one-shot prompt now selects from verified candidates.
- [x] **AccommodationAgent:** real hotels from OpenStreetMap, for the nights the Supervisor derives.
```

In §2 Phase 3, fix the multi-turn example — a ski-bag question is a Contract of Carriage
question, which FlightAgent cannot answer:

```markdown
- [~] Multi-turn end-to-end: GUI sends prior `conversation_id`, the route loads history and passes
  it to `supervisor.run(prompt, history)`, agents see prior context on follow-ups (e.g. "is there
  anything earlier than the 16:30?" for `FlightAgent`; "can I take my ski bag on it?" is a
  Contract of Carriage question and belongs to `DocumentationAgent`).
```

In §2 Phase 3, extend the `agent_info` description note:

```markdown
  - [ ] The `description` must state that flight schedules and hotels are real, while prices and
        fare conditions are estimates — no free source of fares or availability exists (§6).
```

In §7, replace the flights/hotels risk bullet with:

```markdown
- ~~No real booking API is named in the spec~~ — **resolved 9/8/2026**: AeroDataBox + OpenStreetMap.
  The live risk is now **quota**: 600 units/month at 2 units per call, shared across the team.
  Mitigated by a per-instance cache and `WINGMAN_LIVE_DATA=0` in development. Contingency ladder
  (documented, not built) in the design doc §10.
```

Add to the §6 decisions log:

```markdown
| 9/8/2026 | Flights from **AeroDataBox**, hotels from **OpenStreetMap Overpass** | Amadeus Self-Service, the obvious choice, was decommissioned 17/7/2026 and its keys disabled; Kiwi Tequila went invite-only. AeroDataBox has the largest free quota still open to self-signup (600 units/month); Overpass is keyless, so the hotel half can never be blocked by a quota. |
| 9/8/2026 | Prices and seat availability are **LLM estimates, labelled** | No free source for either exists as of 8/2026. `/api/agent_info`'s description must say so. |
| 9/8/2026 | A tool/HTTP fetch produces **no `steps[]` entry** | The spec ties `steps[]` to LLM calls, and a step requires `prompt.system_prompt`/`user_prompt`, which an HTTP GET has neither of. The fetched data appears inside the agent's LLM `user_prompt` instead, so the trace still shows what drove the answer. Applies to Person C's Pinecone retrieval too. |
| 9/8/2026 | `FlightAgent` **defers baggage/fare/entitlement questions to `DocumentationAgent`** | Schedule data cannot answer what a ticket allows; the Contract of Carriage can, with a clause citation. Corrects §2 Phase 3's ski-bag example and `CLAUDE.md` §3's "check terms". |
| 9/8/2026 | Data-source failure **degrades to LLM-only, labelled in `notes`** | The demo must survive an exhausted quota or a flaky Overpass during grading. Mirrors `lib/conversation.py`'s posture on Supabase. |
| 9/8/2026 | Airport coordinates ship as a generated **Python module**, not JSON | Vercel's Python builder traces imports, not data files; a `.json` would need a `vercel.json` `includeFiles` change and would fail silently in production if the glob were wrong. |
```

- [ ] **Step 5: Verify nothing contradicts**

Run: `grep -rn "check terms\|ski bag\|mock vs\|data source" CLAUDE.md docs/PROJECT_PLAN.md`
Expected: every hit is one of the corrected passages above. CLAUDE.md §7 requires a reversed
decision to be fixed everywhere it is stated, not just in the log.

- [ ] **Step 6: Run the whole suite one last time**

Run: `python -m pytest -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add .env.example README.md CLAUDE.md docs/PROJECT_PLAN.md
git commit -m "Record the data-source decision everywhere it is stated

CLAUDE.md §7 requires a reversed decision to be corrected in every file that
states it, not only in the decisions log. Amadeus Self-Service closing forced
the choice, and narrowing FlightAgent's remit makes the ski-bag example in
Phase 3 and the 'check terms' line in CLAUDE.md §3 wrong as written."
```

---

## Verification before opening the PR

- [ ] `python -m pytest -v` — everything green with no API keys configured.
- [ ] With a real `LLMOD_API_KEY` and `WINGMAN_LIVE_DATA=1`, run `uvicorn api.index:app --reload`
      and POST a real disruption to `/api/execute`. Confirm: the `steps[]` trace has exactly one
      `FlightAgent` entry and one `AccommodationAgent` entry, the flight numbers in the response
      match the candidates in the `user_prompt`, and no step corresponds to an HTTP fetch.
- [ ] Check quota afterwards with the README command; confirm the spend matches expectations
      (2 units per uncached route search, 4 if the window widened).
- [ ] PR description must flag the edit to `tests/test_supervisor.py` for Person A, and raise
      narrowing `needs` on follow-up turns as an integration item (design doc D6).
