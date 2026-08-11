"""Deterministic coverage rules must detect missing legal topics."""

import json
from pathlib import Path

from langchain_core.documents import Document

from lib.rag.coverage import (
    coverage_report,
    fallback_queries_for,
    passage_covers,
    requirements_for,
)
from lib.rag.retrieve import RetrievedPassage


def hit(namespace, document_id, text):
    chunk_id = f"{namespace}-{document_id}-{len(text)}"
    return RetrievedPassage(
        document=Document(
            id=chunk_id,
            page_content=text,
            metadata={
                "chunk_id": chunk_id,
                "document_id": document_id,
                "section_path": text,
            },
        ),
        score=0.9,
        namespace=namespace,
    )


def test_cancellation_requirements_cover_every_applicable_authority():
    requirements = requirements_for(
        "cancelled", ("coc-ly", "rights-eu", "rights-il")
    )
    keys = {item.key for item in requirements}

    assert "coc-ly:airline_policy" in keys
    assert "rights-eu:care" in keys
    assert "rights-eu:refund_rerouting" in keys
    assert "rights-eu:applicability" in keys
    assert "rights-il:event_entitlement" in keys
    assert "rights-il:eligibility" in keys
    assert "rights-il:benefit_details" in keys
    assert "rights-il:compensation_amounts" in keys
    assert "rights-il:current_amendments" in keys

    by_key = {item.key: item for item in requirements}
    assert by_key["rights-eu:applicability"].provision_ids == (
        "eu_regulation_261_2004:article_3",
    )
    assert by_key["rights-il:applicability"].provision_ids == (
        "il_aviation_services_law_2012:section_1",
    )


def test_irrelevant_domestic_section_does_not_satisfy_cancellation_coverage():
    requirements = requirements_for("cancelled", ("rights-il",))
    passages = [
        hit(
            "rights-il",
            "il_aviation_services_law_2012",
            "Section 18 - Application to domestic flights",
        )
    ]

    report = coverage_report(requirements, passages)

    assert "rights-il:event_entitlement" in report.missing
    assert "rights-il:care" in report.missing
    assert "rights-il:refund_rerouting" in report.missing


def test_airline_table_of_contents_does_not_count_as_policy_coverage():
    requirement = requirements_for("cancelled", ("coc-ly",))[0]

    assert not passage_covers(
        requirement,
        namespace="coc-ly",
        metadata={"section_path": "Table of Contents"},
        text="Cancellation, delays, refunds, rerouting, meals and hotels",
    )
    assert passage_covers(
        requirement,
        namespace="coc-ly",
        metadata={"section_path": "Schedule changes"},
        text="If we cancel your flight, we will offer a refund or rerouting.",
    )


def test_each_coverage_topic_has_deterministic_fallback_queries():
    requirements = requirements_for(
        "cancelled", ("coc-ly", "rights-eu", "rights-il", "rights-us")
    )

    assert all(len(fallback_queries_for(requirement)) >= 2 for requirement in requirements)


def test_applicability_requires_the_decisive_scope_provision_and_language():
    requirements = requirements_for("cancelled", ("rights-eu", "rights-il"))
    by_key = {requirement.key: requirement for requirement in requirements}

    assert not passage_covers(
        by_key["rights-eu:applicability"],
        namespace="rights-eu",
        metadata={
            "document_id": "eu_regulation_261_2004",
            "provision_id": "eu_regulation_261_2004:article_3",
            "section_path": "Article 3(4)",
        },
        text="This Regulation shall only apply to passengers transported by fixed-wing aircraft.",
    )
    assert passage_covers(
        by_key["rights-eu:applicability"],
        namespace="rights-eu",
        metadata={
            "document_id": "eu_regulation_261_2004",
            "provision_id": "eu_regulation_261_2004:article_3",
            "section_path": "Article 3(1)",
        },
        text="This Regulation applies to passengers departing from an airport located in a Member State.",
    )
    assert not passage_covers(
        by_key["rights-il:applicability"],
        namespace="rights-il",
        metadata={
            "document_id": "il_aviation_services_law_2012",
            "provision_id": "il_aviation_services_law_2012:section_18",
            "section_path": "Section 18 - Application to domestic flights",
        },
        text="This Law applies to a flight whose departure and destination are within Israel.",
    )
    assert passage_covers(
        by_key["rights-il:applicability"],
        namespace="rights-il",
        metadata={
            "document_id": "il_aviation_services_law_2012",
            "provision_id": "il_aviation_services_law_2012:section_1",
            "section_path": "Section 1 - Definitions",
        },
        text='"Flight" means a flight departing from or arriving in the territory of the State of Israel.',
    )


def test_labeled_cases_are_fully_represented_by_mandatory_bundles():
    cases_path = Path(__file__).resolve().parents[1] / "evals" / "provision_coverage_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    for case in cases:
        requirements = requirements_for(
            case["disruption"], tuple(case["namespaces"])
        )
        actual = {
            provision_id
            for requirement in requirements
            for provision_id in requirement.provision_ids
        }
        assert set(case["expected_provision_ids"]) <= actual, case["case_id"]


def test_eu_article_5_cancellation_clause_satisfies_event_entitlement():
    requirement = next(
        requirement
        for requirement in requirements_for("cancelled", ("rights-eu",))
        if requirement.topic == "event_entitlement"
    )

    assert passage_covers(
        requirement,
        namespace="rights-eu",
        metadata={
            "document_id": "eu_regulation_261_2004",
            "provision_id": "eu_regulation_261_2004:article_5",
            "section_path": "Article 5 > 1. In case of cancellation of a flight",
        },
        text=(
            "In case of cancellation of a flight, passengers concerned shall: "
            "be offered assistance in accordance with Articles 8 and 9 and have "
            "the right to compensation in accordance with Article 7."
        ),
    )


def test_us_denied_boarding_bundle_uses_actual_federal_remedies():
    requirements = requirements_for("denied_boarding", ("rights-us",))
    by_topic = {requirement.topic: requirement for requirement in requirements}

    assert "refund_rerouting" not in by_topic
    assert by_topic["compensation_domestic"].provision_ids == (
        "us_14_cfr_part_250_oversales:section_250_5",
    )
    assert by_topic["compensation_international"].provision_ids == (
        "us_14_cfr_part_250_oversales:section_250_5",
    )
    assert by_topic["notice"].provision_ids == (
        "us_14_cfr_part_250_oversales:section_250_9",
    )
    assert by_topic["compensation_limits"].document_ids == (
        "us_part_250_compensation_adjustment_2024",
    )
    assert by_topic["current_enforcement"].document_ids == (
        "us_part_250_adjustment_enforcement_notice_2025",
    )


def test_us_tarmac_care_recognizes_section_259_4_terms():
    by_topic = {
        requirement.topic: requirement
        for requirement in requirements_for("delayed", ("rights-us",))
    }
    metadata = {
        "document_id": "us_14_cfr_part_259_enhanced_protections",
        "provision_id": "us_14_cfr_part_259_enhanced_protections:section_259_4",
        "section_path": "14 CFR 259.4 - Lengthy Tarmac Delays",
    }

    assert passage_covers(
        by_topic["care"],
        namespace="rights-us",
        metadata=metadata,
        text=(
            "The carrier must provide adequate food and potable water no later "
            "than two hours after the tarmac delay begins."
        ),
    )
    assert passage_covers(
        by_topic["tarmac_deplaning"],
        namespace="rights-us",
        metadata=metadata,
        text=(
            "For domestic flights, the carrier shall provide passengers the op-portunity "
            "to deplane before the tarmac delay exceeds three hours, subject to exceptions."
        ),
    )
    assert not passage_covers(
        by_topic["tarmac_deplaning"],
        namespace="rights-us",
        metadata=metadata,
        text="The carrier must provide adequate food and potable water.",
    )


def test_us_part_250_compensation_subsections_are_not_interchangeable():
    domestic = next(
        requirement
        for requirement in requirements_for("denied_boarding", ("rights-us",))
        if requirement.topic == "compensation_domestic"
    )
    international = next(
        requirement
        for requirement in requirements_for("denied_boarding", ("rights-us",))
        if requirement.topic == "compensation_international"
    )

    assert passage_covers(
        domestic,
        namespace="rights-us",
        metadata={
            "document_id": "us_14_cfr_part_250_oversales",
            "provision_id": "us_14_cfr_part_250_oversales:section_250_5",
            "section_path": "Section 250.5 - Amount of denied boarding compensation",
        },
        text=(
            "The carrier shall pay compensation in interstate air transportation "
            "to passengers denied boarding involuntarily from an oversold flight "
            "when arrival is more than one hour but less than two hours late."
        ),
    )
    assert not passage_covers(
        international,
        namespace="rights-us",
        metadata={
            "document_id": "us_14_cfr_part_250_oversales",
            "provision_id": "us_14_cfr_part_250_oversales:section_250_5",
        },
        text="The carrier shall pay compensation in interstate air transportation.",
    )
    assert not passage_covers(
        international,
        namespace="rights-us",
        metadata={
            "document_id": "us_14_cfr_part_250_oversales",
            "provision_id": "us_14_cfr_part_250_oversales:section_250_5",
        },
        text=(
            "The carrier shall pay compensation to passengers in for-eign air "
            "transportation, but this overlap ends before the timing tiers."
        ),
    )
    assert passage_covers(
        international,
        namespace="rights-us",
        metadata={
            "document_id": "us_14_cfr_part_250_oversales",
            "provision_id": "us_14_cfr_part_250_oversales:section_250_5",
        },
        text=(
            "The carrier shall pay compensation to passengers in for-eign air "
            "trans-portation who are denied boarding involuntarily at a U.S. airport "
            "when arrival is more than one hour but less than four hours late."
        ),
    )
