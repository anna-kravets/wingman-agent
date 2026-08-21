"""The GUI's markdown renderer lives inline in public/index.html.

There is no JS test setup and adding one for a handful of pure functions is not
worth it, so this slices the renderer out of the page and runs it under node.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[1] / "public" / "index.html"

HARNESS = """
const fs = require("fs");
const page = fs.readFileSync(process.argv[1], "utf8");
const start = page.indexOf("function escapeHtml(");
const end = page.indexOf("function messageElement(");
eval(page.slice(start, end));
process.stdout.write(renderMarkdown(JSON.parse(process.argv[2])));
"""


def render(markdown: str) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    return subprocess.run(
        [node, "-e", HARNESS, "--", str(INDEX), json.dumps(markdown)],
        capture_output=True, text=True, check=True,
    ).stdout


def test_blank_lines_between_items_keep_counting():
    """The models write loose lists, which used to render as "1." three times."""
    html = render("1. First\n\n2. Second\n\n3. Third")
    assert '<ol start="2">' in html
    assert '<ol start="3">' in html


def test_a_list_that_is_not_split_has_no_start_attribute():
    assert "<ol><li>First</li><li>Second</li></ol>" in render("1. First\n2. Second")
