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


# Credentials that must never reach a test. `api/index.py` calls load_dotenv() at
# import, so without this the suite silently inherits whoever's .env is on disk and
# passes or fails according to what they happen to have configured - which is how a
# green suite turned red the moment a real Supabase project appeared in one.
CREDENTIALS = (
    "SUPABASE_URL", "SUPABASE_KEY",
    "LLMOD_API_KEY", "LLMOD_API_BASE",
    "AERODATABOX_API_KEY", "AERODATABOX_API_HOST",
    "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
)


@pytest.fixture(autouse=True)
def _no_live_data(monkeypatch):
    """No test may spend API quota, bill the LLM budget, or touch a database.

    A test that genuinely needs one of these sets it itself with monkeypatch,
    which overrides this. Everything else runs hermetically, which is what the
    README promises and what makes the suite mean the same thing on every machine.
    """
    monkeypatch.setenv("WINGMAN_LIVE_DATA", "0")
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
