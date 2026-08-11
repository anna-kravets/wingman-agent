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
