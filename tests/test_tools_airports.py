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
