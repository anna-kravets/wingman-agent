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
