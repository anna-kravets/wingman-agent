"""Passive provenance extraction for DocumentationAgent evidence.

DocumentationAgent already sends labelled evidence blocks to the model.  This
module copies those same blocks out of the recorded step prompt for the UI.  It
does not change retrieval, source ordering, prompts, reflection, or the agent's
legal assessment.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable


_BLOCK = re.compile(
    r"(?ms)^\[(?P<id>S\d+)\]\s+(?P<document>[^\r\n]+)\r?\n"
    r"(?P<header>.*?)\r?\nExcerpt:\r?\n"
    r"(?P<excerpt>.*?)"
    r"(?=\r?\n\r?\n\[S\d+\]|\r?\n\r?\nWrite the grounded draft\.|\Z)"
)

_GLOSS = re.compile(r"\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*$")

_FIELDS = {
    "Namespace": "namespace",
    "Document type": "document_type",
    "Section": "section",
    "Provision ID": "provision_id",
    "Legal citation": "legal_citation",
    "Official source": "source_url",
}


def strip_gloss(reference: str) -> str:
    """Drop the explanatory tail a model attaches to a citation, keeping the citation.

    "Rule 24 (snacks and meals)" becomes "Rule 24", which is the form an answer
    actually repeats and therefore the only form the GUI can link. The trailing
    bracket of "Art. 8(1)(a)" is part of the citation, so it stays: a gloss is prose,
    which means it contains a space or is longer than a subsection label.

    Mirrored by stripCitationGloss in public/index.html.
    """

    reference = str(reference).strip()
    match = _GLOSS.search(reference)
    if not match:
        return reference
    inner = match.group(1)
    if " " not in inner and len(inner) < 5:
        return reference
    return reference[:match.start()].strip()


def _source_references(payload: dict) -> dict[str, list[str]]:
    """Every [S#] the legal assessment cited, with any human-readable alias beside it.

    A cited source with no alias still gets an entry, holding an empty list. The model
    sometimes writes "source": "[S2], [S3]" and nothing else; those passages have a
    document, a section, an excerpt and an official URL worth showing, even though
    there is no phrase in the answer for the GUI to turn into an inline link.
    """

    references: dict[str, list[str]] = {}
    rights = payload if isinstance(payload, dict) else {}
    for entitlement in rights.get("entitlements") or []:
        if not isinstance(entitlement, dict):
            continue
        source = str(entitlement.get("source") or "")
        matches = list(re.finditer(r"\[(S\d+)\]\s*", source))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
            label = source[match.end():end].strip(" ;,[]")
            aliases = references.setdefault(match.group(1), [])
            if label and label not in aliases:
                aliases.append(label)
    return references


def _evidence_prompt(steps: Iterable[dict]) -> str:
    for step in steps:
        if not isinstance(step, dict) or step.get("module") != "DocumentationAgent":
            continue
        prompt = step.get("prompt")
        user_prompt = prompt.get("user_prompt") if isinstance(prompt, dict) else None
        if isinstance(user_prompt, str) and "Retrieved evidence:\n" in user_prompt:
            return user_prompt.split("Retrieved evidence:\n", 1)[1]
    return ""


def extract_citations(payload: dict, steps: Iterable[dict]) -> list[dict]:
    """Return only evidence sources actually cited by DocumentationAgent."""

    references = _source_references(payload)
    if not references:
        return []

    citations: list[dict] = []
    for match in _BLOCK.finditer(_evidence_prompt(steps)):
        citation_id = match.group("id")
        if citation_id not in references:
            continue

        fields: dict[str, str] = {}
        for line in match.group("header").splitlines():
            label, separator, value = line.partition(":")
            key = _FIELDS.get(label.strip())
            if separator and key:
                fields[key] = value.strip()

        excerpt = match.group("excerpt").strip()
        identity = "|".join((
            fields.get("namespace", ""),
            fields.get("provision_id", ""),
            fields.get("section", ""),
            excerpt,
        ))
        citations.append({
            "id": citation_id,
            "source_key": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
            "document": match.group("document").strip(),
            "namespace": fields.get("namespace", ""),
            "document_type": fields.get("document_type", ""),
            "section": fields.get("section", ""),
            "provision_id": fields.get("provision_id", ""),
            "legal_citation": fields.get("legal_citation", ""),
            "source_url": fields.get("source_url", ""),
            "excerpt": excerpt,
            "references": references[citation_id],
        })

    return citations

def citations_from_results(results: object) -> list[dict]:
    """Validate the small citation surface before returning stored history."""

    if not isinstance(results, dict) or not isinstance(results.get("citations"), list):
        return []
    citations = []
    for citation in results["citations"]:
        if not isinstance(citation, dict):
            continue
        if not isinstance(citation.get("id"), str) or not isinstance(
                citation.get("excerpt"), str):
            continue
        citations.append(citation)
    return citations
