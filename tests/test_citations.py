from lib.citations import citations_from_results, extract_citations
from lib.steps import make_step


EVIDENCE = """Passenger facts:
Cancelled flight.

Retrieved evidence:
[S1] Regulation (EC) No 261/2004
Namespace: rights-eu
Retrieval score: 0.765432
Selection reason: coverage_recovery:refund_rerouting
Document type: passenger_rights
Section: Regulation (EC) No 261/2004 > Article 8
Provision ID: eu_regulation_261_2004:article_8
Legal citation: Regulation (EC) No 261/2004; CELEX 32004R0261
Official source: https://example.test/eu261.pdf
Excerpt:
Passengers shall be offered reimbursement or re-routing.

[S2] Airline conditions
Namespace: coc-lh
Retrieval score: 0.500000
Selection reason: semantic
Document type: conditions_of_carriage
Section: Article 5
Provision ID:
Legal citation:
Official source: https://example.test/coc
Excerpt:
An unrelated passage.

Write the grounded draft."""


def test_only_cited_evidence_is_exposed_with_its_exact_excerpt():
    payload = {
        "entitlements": [{
            "kind": "refund",
            "summary": "Choose reimbursement.",
            "source": "[S1] EU 261 Art. 8(1)(a)",
            "confidence": "high",
        }],
        "next_actions": [],
        "caveats": [],
    }
    steps = [make_step("DocumentationAgent", "system", EVIDENCE, payload)]

    citations = extract_citations(payload, steps)

    assert len(citations) == 1
    assert citations[0]["id"] == "S1"
    assert citations[0]["document"] == "Regulation (EC) No 261/2004"
    assert citations[0]["section"].endswith("Article 8")
    assert citations[0]["excerpt"] == (
        "Passengers shall be offered reimbursement or re-routing."
    )
    assert citations[0]["references"] == ["EU 261 Art. 8(1)(a)"]
    assert len(citations[0]["source_key"]) == 20


def test_missing_or_malformed_stored_citations_are_not_returned():
    valid = {"id": "S3", "excerpt": "Exact source text."}

    assert citations_from_results(None) == []
    assert citations_from_results({"citations": [None, {"id": 3}, valid]}) == [valid]
