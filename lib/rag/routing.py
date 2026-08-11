"""Deterministic routing from an itinerary to Pinecone namespaces."""

from __future__ import annotations

from dataclasses import dataclass


AIRLINE_NAMESPACES = {
    "AA": "coc-aa",
    "AC": "coc-ac",
    "AF": "coc-af",
    "AL": "coc-al",
    "BA": "coc-ba",
    "DL": "coc-dl",
    "EK": "coc-ek",
    "FR": "coc-fr",
    "LH": "coc-lh",
    "LY": "coc-ly",
    "RK": "coc-rk",
    "RR": "coc-rr",
    "UA": "coc-ua",
}

AIRLINE_ALIASES = {
    "AMERICAN": "AA",
    "AMERICAN AIRLINES": "AA",
    "AIR CANADA": "AC",
    "AIR FRANCE": "AF",
    "MALTA AIR": "AL",
    "BRITISH AIRWAYS": "BA",
    "DELTA": "DL",
    "DELTA AIR LINES": "DL",
    "EMIRATES": "EK",
    "RYANAIR": "FR",
    "LUFTHANSA": "LH",
    "EL AL": "LY",
    "ELAL": "LY",
    "RYANAIR UK": "RK",
    "BUZZ": "RR",
    "UNITED": "UA",
    "UNITED AIRLINES": "UA",
}

# The agent receives IATA airport codes, not countries. These sets cover the main
# airports served by the selected corpus airlines. Unknown routes deliberately fall
# back to all three rights namespaces so retrieval loses precision, never recall.
ISRAEL_AIRPORTS = {"TLV", "HFA", "ETM"}
US_AIRPORTS = {
    "ATL", "AUS", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW",
    "EWR", "FLL", "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "MCO",
    "MIA", "MSP", "ORD", "PDX", "PHL", "PHX", "SAN", "SEA", "SFO",
    "SJC", "SLC", "TPA",
}
EU_EEA_AIRPORTS = {
    "AMS", "ARN", "ATH", "BCN", "BER", "BRU", "BUD", "CDG", "CPH",
    "DUB", "DUS", "FCO", "FRA", "HEL", "KEF", "LIS", "MAD", "MUC",
    "MXP", "NAP", "NCE", "OSL", "ORY", "OTP", "PRG", "RIX", "SOF",
    "STR", "TLL", "VIE", "VNO", "WAW", "ZAG",
}
EU_EEA_CARRIERS = {"AF", "AL", "FR", "LH", "RR"}
RIGHTS_NAMESPACES = ("rights-eu", "rights-us", "rights-il")


@dataclass(frozen=True)
class RetrievalRoute:
    airline_code: str | None
    jurisdictions: tuple[str, ...]
    namespaces: tuple[str, ...]
    caveats: tuple[str, ...] = ()


def _airline_code(request: dict) -> str | None:
    airline = str(request.get("airline") or "").strip().upper()
    if airline in AIRLINE_NAMESPACES:
        return airline
    if airline in AIRLINE_ALIASES:
        return AIRLINE_ALIASES[airline]
    flight_number = str(request.get("flight_number") or "").strip().upper().replace(" ", "")
    prefix = flight_number[:2]
    return prefix if prefix in AIRLINE_NAMESPACES else None


def route_request(request: dict) -> RetrievalRoute:
    """Choose the carrier contract and plausibly applicable passenger-rights law.

    This is routing, not a legal conclusion. The model still has to apply eligibility
    conditions from the retrieved text to the passenger's facts.
    """

    airline = _airline_code(request)
    origin = str(request.get("origin") or "").strip().upper()
    destination = str(request.get("destination") or "").strip().upper()
    jurisdictions: list[str] = []

    if origin in ISRAEL_AIRPORTS or destination in ISRAEL_AIRPORTS:
        jurisdictions.append("IL")
    if origin in US_AIRPORTS or destination in US_AIRPORTS:
        jurisdictions.append("US")
    # EU261 applies to departures from the EU/EEA, and to arrivals there when the
    # operating carrier is an EU/EEA carrier. Carrier identity remains a fact the
    # final assessment must verify when codeshares are involved.
    if origin in EU_EEA_AIRPORTS or (
        destination in EU_EEA_AIRPORTS and airline in EU_EEA_CARRIERS
    ):
        jurisdictions.append("EU")

    caveats: list[str] = []
    namespaces: list[str] = []
    if airline:
        namespaces.append(AIRLINE_NAMESPACES[airline])
    else:
        caveats.append("The airline could not be mapped to a Contract of Carriage namespace.")

    if jurisdictions:
        namespaces.extend(f"rights-{code.lower()}" for code in jurisdictions)
    else:
        namespaces.extend(RIGHTS_NAMESPACES)
        caveats.append(
            "The route could not be classified deterministically, so all supported rights "
            "jurisdictions were searched; applicability must be treated as uncertain."
        )

    return RetrievalRoute(
        airline_code=airline,
        jurisdictions=tuple(jurisdictions),
        namespaces=tuple(dict.fromkeys(namespaces)),
        caveats=tuple(caveats),
    )
