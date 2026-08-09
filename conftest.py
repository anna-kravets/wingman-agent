"""Puts the repo root on sys.path so tests can `from api.index import app`.

pytest adds the directory containing the root conftest.py to sys.path, so the file
existing here is most of the job; the explicit insert keeps it working when tests are
run from somewhere other than the repo root.
"""

import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _no_live_data(monkeypatch):
    """No test may spend API quota or hit a public endpoint."""
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "0")
