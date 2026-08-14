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

# Statuses a passenger cannot act on: the flight has gone, been called off, or is
# closed to boarding. Live validation on 14/8/2026 caught the model recommending a
# flight already marked "Departed" - it flagged the status in one run and ignored it
# in the next. Whether an option is catchable at all is not a judgement call, so it
# is decided here rather than left to a prompt.
#
# A deny-list, not an allow-list, on purpose: an unfamiliar status AeroDataBox adds
# later should still reach the passenger rather than silently emptying the list and
# triggering a "nothing could be verified" refusal.
UNUSABLE_STATUSES = frozenset({
    "departed", "arrived", "enroute", "approaching", "diverted",
    "canceled", "cancelled", "canceleduncertain", "cancelleduncertain", "gateclosed",
})

_cache: dict[tuple, list[dict]] = {}


def _is_usable(status: str | None) -> bool:
    return str(status or "").strip().lower() not in UNUSABLE_STATUSES


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
            if (option["destination"] == destination and option["depart"]
                    and _is_usable(option["status"])):
                option["origin"] = origin
                found.append(option)
        if len(found) >= MIN_OPTIONS:
            break
        start = end

    if found:
        _cache[cache_key] = found
    return found
