"""IATA -> coordinates, and plain city names -> IATA. Static, offline, no quota.

Exists so hotel search can turn "the passenger is stranded at TLV" into a
point on the map without spending an API call on geocoding.

Passengers do not talk in IATA codes. They say "Dublin", and every downstream
consumer - the flight search, the hotel search, the rights routing - only
speaks codes, so an unresolved city name used to surface as "DUBLIN is not an
airport I can look up". `resolve` closes that gap for the cities where the
answer is not a guess; where it is (London, New York), `candidates` gives the
shortlist to ask about instead.
"""

import re
import unicodedata

from lib.tools.airports_data import AIRPORTS


def lookup(iata: str | None) -> dict | None:
    """The airport, or None if the code is unknown."""
    if not iata:
        return None
    code = iata.strip().upper()
    row = AIRPORTS.get(code)
    if not row:
        return None
    name, city, country, lat, lon, large = row
    return {"iata": code, "name": name, "city": city, "country": country,
            "lat": lat, "lon": lon, "large": large}


def _normalise(text: str | None) -> str:
    """Lower-case, unaccented, punctuation-free. "Málaga" and "malaga" must match."""
    stripped = unicodedata.normalize("NFKD", text or "")
    plain = "".join(c for c in stripped if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", plain)).strip()


# OurAirports writes the city as "Paris (Roissy-en-France, Val-d'Oise)", so the
# qualifier is dropped. Matching the name needs word boundaries: without them
# "kos" hits Kosice and "milan" hits every Milano.
_CITY = {code: _normalise(re.sub(r"\(.*", "", row[1] or "")) for code, row in AIRPORTS.items()}
_NAME = {code: _normalise(row[0]) for code, row in AIRPORTS.items()}


def candidates(place: str | None) -> list[str]:
    """Airports a passenger could mean by this city or airport name.

    Narrowed in the order the ambiguity actually resolves: the country's main
    airport beats a same-named regional field (Manchester GB over Manchester
    NH), the city itself beats a name that merely contains it (Porto over Porto
    Alegre), and an airport named for the city beats one that is not (Barcelona
    El Prat over Barcelona in Venezuela). One survivor is an answer; several
    mean the passenger has to pick (London, New York, Paris).
    """
    query = _normalise(place)
    if not query:
        return []
    word = re.compile(rf"\b{re.escape(query)}\b")
    found = [code for code in AIRPORTS if _CITY[code] == query or word.search(_NAME[code])]
    for narrower in (lambda c: AIRPORTS[c][5], lambda c: _CITY[c] == query,
                     lambda c: word.search(_NAME[c])):
        if len(found) == 1:
            break
        found = [code for code in found if narrower(code)] or found
    return sorted(found)


def resolve(place: str | None) -> str | None:
    """The IATA code the passenger means, or None if it is a code already, unknown
    or genuinely ambiguous."""
    if not place:
        return None
    if lookup(place):
        return place.strip().upper()
    found = candidates(place)
    return found[0] if len(found) == 1 else None


def describe(iata: str) -> str:
    """"DUB (Dublin, IE)" - for asking the passenger which airport they meant."""
    airport = lookup(iata)
    if not airport:
        return iata
    where = ", ".join(p for p in (airport["city"], airport["country"]) if p)
    return f"{iata} ({where})" if where else iata


def unknown_reason(place: str) -> str:
    """Why this is not usable as an airport, in words the passenger can act on."""
    found = candidates(place)
    if len(found) > 1:
        return (f"{place} has more than one airport - "
                f"{', '.join(describe(code) for code in found)}. Which one?")
    return (f"{place} is not an airport I can look up - it may be closed, or the "
            f"code may be wrong. Check it and tell me again.")
