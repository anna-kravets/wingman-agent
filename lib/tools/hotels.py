"""Places to sleep near an airport, from OpenStreetMap via Overpass. No key, no LLM call.

OSM knows a property's name and where it is, and after that it thins out fast: of
20 hotels near TLV, one carried a star rating. What it does carry often enough to
matter is *contact detail* - a phone number on roughly a quarter of them. That is
the most valuable field here, because the thing this agent cannot do is check a
price or a free room, and a phone number hands that job to someone who can.

Distance is the one number we can state exactly, and it is also what decides
whether a passenger makes a 06:00 departure.
"""

import math
import time

import httpx

from lib.tools import airports, live_data_enabled

# The public instance intermittently returns 504; the mirror is the fallback.
MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT = "wingman-agent/0.1 (course project)"
ATTEMPTS_PER_MIRROR = 2
RETRY_PAUSE_SECONDS = 3.0
MAX_RESULTS = 8
TIMEOUT_SECONDS = 60
EARTH_RADIUS_KM = 6371

# A stranded passenger needs a bed, not specifically a hotel.
ACCOMMODATION = ("hotel", "hostel", "guest_house", "motel", "apartment", "resort")

# Ramon/Eilat returned nothing at 12km and was written up as an OSM coverage gap.
# It sits 19km from Eilat, which has 141 places to stay: the limitation was this
# radius, not the data. Overpass is keyless and unmetered, so widening is free -
# it costs a little latency and nothing else.
SEARCH_RADII_KM = (12.0, 25.0, 40.0)
MIN_RESULTS = 3

_cache: dict[tuple, list[dict]] = {}


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return round(EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a)), 1)


def _query(lat: float, lon: float, radius_m: int) -> str:
    selectors = "".join(
        f'node["tourism"="{kind}"](around:{radius_m},{lat},{lon});'
        f'way["tourism"="{kind}"](around:{radius_m},{lat},{lon});'
        for kind in ACCOMMODATION
    )
    return f"[out:json][timeout:40];({selectors});out center tags;"


def _fetch(query: str) -> list[dict]:
    """Try each mirror, twice, before giving up.

    Overpass throttles on slot contention and recovers within seconds. A live run
    refused to propose any accommodation because of exactly that, moments after
    the same query had returned eight results - so a single blip must not cost
    the passenger a bed. Bounded at MIRRORS x ATTEMPTS_PER_MIRROR requests.
    """
    last_error: Exception | None = None
    for url in MIRRORS:
        for attempt in range(ATTEMPTS_PER_MIRROR):
            try:
                response = httpx.post(url, data={"data": query},
                                      headers={"User-Agent": USER_AGENT},
                                      timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.json().get("elements") or []
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_error = exc
                if attempt + 1 < ATTEMPTS_PER_MIRROR:
                    time.sleep(RETRY_PAUSE_SECONDS)
    raise last_error if last_error else RuntimeError("no Overpass mirror configured")


def _detail(tags: dict, *keys: str) -> str | None:
    """First of `keys` that OSM actually filled in. OSM spells contact details
    both as `phone` and `contact:phone`, and either may be the one present."""
    for key in keys:
        value = (tags.get(key) or "").strip()
        if value:
            return value
    return None


def _address(tags: dict) -> str | None:
    street = _detail(tags, "addr:street")
    if not street:
        return None
    number = _detail(tags, "addr:housenumber")
    return f"{number} {street}" if number else street


def _place(element: dict, airport: dict) -> dict | None:
    """One OSM element, reduced to what is worth a passenger's attention.

    Fields OSM did not fill in are left out entirely rather than sent as nulls:
    every key costs prompt tokens against a $13 project budget.
    """
    tags = element.get("tags") or {}
    centre = element.get("center") or {}
    name = _detail(tags, "name:en", "name")   # OSM names are often local-language
    lat = element.get("lat", centre.get("lat"))
    lon = element.get("lon", centre.get("lon"))
    if not (name and lat is not None and lon is not None):
        return None

    place = {
        "name": name,
        "distance_km": _distance_km(airport["lat"], airport["lon"], lat, lon),
        "area": _detail(tags, "addr:city") or airport["city"],
    }
    optional = {
        "kind": tags.get("tourism") if tags.get("tourism") != "hotel" else None,
        "phone": _detail(tags, "phone", "contact:phone"),
        "website": _detail(tags, "website", "contact:website"),
        "address": _address(tags),
        "stars": _detail(tags, "stars"),
        "breakfast": _detail(tags, "breakfast"),
        "wheelchair": _detail(tags, "wheelchair"),
        "internet": _detail(tags, "internet_access"),
        "brand": _detail(tags, "brand"),
    }
    place.update({k: v for k, v in optional.items() if v})
    return place


def search(iata: str, radius_km: float | None = None) -> list[dict]:
    """Real places to stay near the airport, nearest first. [] on any failure.

    With no `radius_km`, widens the search until it has something worth showing.
    Pass one explicitly to pin the search to a single radius.
    """
    if not (iata and live_data_enabled()):
        return []
    airport = airports.lookup(iata)
    if not airport:
        return []

    cache_key = (airport["iata"], radius_km)
    if cache_key in _cache:
        return _cache[cache_key]

    radii = (radius_km,) if radius_km is not None else SEARCH_RADII_KM
    found: list[dict] = []
    for radius in radii:
        try:
            elements = _fetch(_query(airport["lat"], airport["lon"], int(radius * 1000)))
        except Exception:
            break
        found = [p for p in (_place(e, airport) for e in elements) if p]
        if len(found) >= MIN_RESULTS:
            break

    found.sort(key=lambda place: place["distance_km"])
    found = found[:MAX_RESULTS]
    if found:
        _cache[cache_key] = found
    return found
