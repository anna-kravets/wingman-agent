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
