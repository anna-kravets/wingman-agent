"""Report which citations a saved answer actually turns into source links.

Mirrors `citationAliases` and `linkifyCitations` (public/index.html) so the question
"did this answer link its sources" can be asked offline, on artifacts already paid
for, instead of by eye in the browser. Costs no API calls.

    python scripts/check_citation_links.py troubleshooting/*.json

Exits non-zero when a cited passage produced no link, or when several citations
collapse onto one alias so the first one silently wins every occurrence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.citations import strip_gloss  # noqa: E402


# Kept in step with citationAliases/citationAliasPattern (public/index.html) and with
# lib.agents.supervisor.PROVISION_RE.
PROVISION_RE = re.compile(
    r"(?:s\.|section|Art\.|Article)\s*\d+[\w().-]*(?:\([^)]*\))*?(?:-\([^)]*\))?",
    re.IGNORECASE,
)
ARTICLE_WORD = re.compile(r"\b(?:articles?|arts?\.)\s*", re.IGNORECASE)
SECTION_WORD = re.compile(r"\b(?:sections?|ss?\.)\s*|§\s*", re.IGNORECASE)
ARTICLE_ALTERNATION = r"(?:articles?|arts?\.)\s*"
SECTION_ALTERNATION = r"(?:sections?|ss?\.|§)\s*"
ID_RE = re.compile(r"^S\d+$")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _aliases(citations: list[dict]) -> list[tuple[str, str]]:
    """(alias, citation id) in the exact order and with the exact dedupe the GUI uses."""

    aliases: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(alias: str, citation_id: str) -> None:
        clean = re.sub(r"[;,]+$", "", str(alias or "").strip())
        key = clean.lower()
        if len(clean) < 4 or key in seen or not ID_RE.match(citation_id or ""):
            return
        seen.add(key)
        aliases.append((clean, citation_id))

    for citation in citations:
        citation_id = str(citation.get("id") or "")
        add(f"[{citation_id}]", citation_id)
        for reference in citation.get("references") or []:
            bare = strip_gloss(reference)
            for candidate in (str(reference), bare):
                add(candidate, citation_id)
                for provision in PROVISION_RE.findall(candidate):
                    add(provision, citation_id)

    return sorted(aliases, key=lambda pair: len(pair[0]), reverse=True)


def _alias_pattern(alias: str) -> re.Pattern:
    """The GUI's citationAliasPattern: either spelling of a provision word matches."""

    marked = ARTICLE_WORD.sub("\x01", _escape_html(alias))
    marked = re.escape(SECTION_WORD.sub("\x02", marked))
    pattern = marked.replace(re.escape("\x01"), ARTICLE_ALTERNATION)
    return re.compile(pattern.replace(re.escape("\x02"), SECTION_ALTERNATION), re.IGNORECASE)


def _blocks(answer: str) -> list[str]:
    """Every string the GUI hands to renderInlineMarkdown, near enough.

    renderMarkdown splits on blank lines and each list item is its own inline call,
    so an alias never matches across a line break and placeholders never leak between
    lines. Splitting on every newline reproduces both.
    """
    return [line for line in _escape_html(answer).split("\n") if line.strip()]


def _link_counts(answer: str, citations: list[dict]) -> dict[str, int]:
    """How many clickable sources each citation id actually gets."""

    counts = {str(citation.get("id")): 0 for citation in citations}
    aliases = _aliases(citations)
    for block in _blocks(answer):
        for index, (alias, citation_id) in enumerate(aliases):
            # The GUI parks each match in a placeholder, so a longer alias that already
            # matched hides its text from every shorter alias behind it.
            block, hits = _alias_pattern(alias).subn(f" CITATION{index} ", block)
            counts[citation_id] = counts.get(citation_id, 0) + hits
    return counts


def _collisions(citations: list[dict]) -> list[tuple[str, str, list[str]]]:
    """(alias, id that wins, ids that lose it) — the same reference under two [S#]."""

    winners = dict(_aliases(citations))
    by_alias: dict[str, list[str]] = {}
    for citation in citations:
        citation_id = str(citation.get("id") or "")
        for reference in citation.get("references") or []:
            bare = strip_gloss(reference)
            candidates = [str(reference), bare]
            candidates += PROVISION_RE.findall(str(reference)) + PROVISION_RE.findall(bare)
            for candidate in candidates:
                clean = re.sub(r"[;,]+$", "", candidate.strip())
                if len(clean) >= 4:
                    by_alias.setdefault(clean.lower(), []).append(citation_id)

    collisions = []
    for alias, winner in winners.items():
        losers = [i for i in dict.fromkeys(by_alias.get(alias.lower(), [])) if i != winner]
        if losers:
            collisions.append((alias, winner, losers))
    return collisions


def _turns(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("turns") or [payload]
    return [turn for turn in payload if isinstance(turn, dict)]


def check_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clean = True
    for number, turn in enumerate(_turns(payload), start=1):
        results = turn.get("results") or {}
        citations = results.get("citations") or []
        answer = str(turn.get("response") or "")
        if not citations:
            # A rights answer with no citations at all is the loudest failure of the
            # three: no chips, no dialog, nothing to link. extract_citations drops the
            # whole array when the model's "source" fields carry bare [S#] labels and
            # no prose, so this is not the same as "no legal question was asked".
            if results.get("rights"):
                print(f"{path.as_posix()} turn {number}")
                print("   !! rights answer carries no citations at all")
                clean = False
            continue

        counts = _link_counts(answer, citations)
        unlinked = sorted(i for i, count in counts.items() if not count)
        collisions = _collisions(citations)
        total = sum(counts.values())

        print(f"{path.as_posix()} turn {number}")
        print(f"  citations {len(citations)}  linked {len(citations) - len(unlinked)}"
              f"  links {total}")
        for citation in citations:
            citation_id = str(citation.get("id"))
            marker = "  " if counts.get(citation_id) else "!!"
            references = "; ".join(str(r) for r in citation.get("references") or [])
            print(f"   {marker} {citation_id} x{counts.get(citation_id, 0)}"
                  f"  {citation.get('namespace', '')}  {references[:90]}")
        for alias, winner, losers in collisions:
            print(f"   !! collision {alias!r}: {winner} wins, "
                  f"{', '.join(losers)} unreachable")
        if unlinked or collisions:
            clean = False
    return clean


def _self_test() -> None:
    citations = [
        {"id": "S1", "references": ["Art. 8(1)(a) of EU 261"]},
        {"id": "S2", "references": ["Rule 24 (snacks and meals)"]},
        {"id": "S3", "references": ["Rule 24 (rerouting)"]},
        {"id": "S4", "references": ["14 CFR Part 259 § 259.8 (notice of delays)"]},
        {"id": "S5", "references": ["Aviation Services Law s.7(a)"]},
        {"id": "S6", "references": []},
    ]
    answer = ("Article 8(1)(a) lets you choose.\nRule 24 covers meals.\n"
              "14 CFR Part 259 § 259.8 says they must tell you.\n"
              "Section 7(a) is the one to quote.")
    counts = _link_counts(answer, citations)
    # S1 matches although the answer spells out "Article"; S2 and S4 match once their
    # gloss is dropped; S3 loses the "Rule 24" alias to S2; S5 matches "Section 7(a)"
    # against "s.7(a)"; S6 was cited with no alias at all, so nothing can link it.
    assert counts == {"S1": 1, "S2": 1, "S3": 0, "S4": 1, "S5": 1, "S6": 0}, counts
    assert dict(_aliases(citations))["Art. 8(1)(a)"] == "S1"
    assert _collisions(citations) == [("Rule 24", "S2", ["S3"])], _collisions(citations)
    # The two provision families stay apart: an article never links a section.
    assert _link_counts("Art. 7(a) is different.", [citations[4]]) == {"S5": 0}
    # A subsection bracket is not a gloss.
    assert strip_gloss("EU261 Art. 8(1)(a)") == "EU261 Art. 8(1)(a)"
    assert strip_gloss("Rule 24 (snacks and meals)") == "Rule 24"
    print("self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.paths:
        parser.error("Pass at least one saved conversation or artifact JSON.")
    return 0 if all([check_file(path) for path in args.paths]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
