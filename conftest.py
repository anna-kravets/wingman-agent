"""Puts the repo root on sys.path so tests can `from api.index import app`.

pytest adds the directory containing the root conftest.py to sys.path, so the file
existing here is most of the job; the explicit insert keeps it working when tests are
run from somewhere other than the repo root.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
