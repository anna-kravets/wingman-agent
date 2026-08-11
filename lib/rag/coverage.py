"""Deterministic legal-topic coverage requirements for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageRequirement:
    namespace: str
    topic: str
    query: str
    document_ids: tuple[str, ...] = ()
    provision_ids: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.topic}"


@dataclass(frozen=True)
class CoverageReport:
    required: tuple[str, ...]
    covered: tuple[str, ...]
    missing: tuple[str, ...]
    recovery_attempted: tuple[str, ...] = ()
    fallback_attempted: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def as_prompt(self) -> str:
        return "\n".join(
            (
                f"Required topics: {', '.join(self.required) or 'none'}",
                f"Covered topics: {', '.join(self.covered) or 'none'}",
                f"Primary recovery attempted: {', '.join(self.recovery_attempted) or 'none'}",
                f"Alternate-query fallback attempted: {', '.join(self.fallback_attempted) or 'none'}",
                f"Still missing: {', '.join(self.missing) or 'none'}",
            )
        )


PRIMARY_DOCUMENTS = {
    "rights-eu": ("eu_regulation_261_2004",),
    "rights-il": ("il_aviation_services_law_2012",),
}


FALLBACK_TOPIC_PHRASES = {
    "airline_policy": (
        "contractual obligation after schedule irregularity carrier will refund or transport passenger",
        "conditions of carriage involuntary refund rebooking meals hotel passenger right",
    ),
    "applicability": (
        "exact scope article conditions under which this law shall apply to a passenger",
        "covered flights operating carrier origin destination eligibility jurisdiction",
    ),
    "event_entitlement": (
        "exact cancellation delay denied boarding section passenger shall receive benefits",
        "passenger entitlement triggered by this flight disruption",
    ),
    "eligibility": (
        "exact eligibility conditions ticket check-in cancellation passenger entitlement",
        "conditions that exclude or preserve passenger benefits",
    ),
    "benefit_details": (
        "exact types of benefits assistance reimbursement replacement flight compensation",
        "statutory definitions and deadlines for each passenger remedy",
    ),
    "refund_rerouting": (
        "exact reimbursement or re-routing article passenger choice ticket refund replacement flight",
        "right to choose refund earliest rerouting or later convenient transport",
    ),
    "care": (
        "exact right to care article meals refreshments hotel accommodation communications transport",
        "free assistance during cancellation or delay food lodging telephone airport hotel transfer",
    ),
    "tarmac_deplaning": (
        "14 CFR 259.4 domestic tarmac delay opportunity to deplane before three hours exceptions",
        "US DOT lengthy tarmac delay deplaning threshold safety security air traffic control exception",
    ),
    "compensation": (
        "exact monetary compensation article amount distance band payment entitlement",
        "cash compensation schedule passenger amount kilometres arrival delay",
    ),
    "compensation_domestic": (
        "interstate denied boarding compensation one hour two hours 200 percent 400 percent",
        "domestic oversold flight compensation alternate transportation arrival delay",
    ),
    "compensation_international": (
        "foreign air transportation denied boarding compensation one hour four hours",
        "international oversold flight from United States compensation arrival delay",
    ),
    "compensation_amounts": (
        "exact compensation schedule distance bands and monetary amounts",
        "first schedule flight distance passenger compensation table",
    ),
    "refund_timing": (
        "exact prompt refund deadline and original form of payment",
        "seven business days twenty calendar days refund method",
    ),
    "notice": (
        "exact written explanation of denied boarding compensation and boarding priorities",
        "carrier must furnish passenger notice immediately after denied boarding",
    ),
    "compensation_limits": (
        "current denied boarding compensation limits 1075 2150 effective date",
        "inflation adjustment 200 percent 400 percent liability cap",
    ),
    "current_enforcement": (
        "current enforcement date for revised denied boarding compensation limits",
        "DOT enforcement discretion March 20 2025 compensation adjustment",
    ),
    "exceptions": (
        "exact exception clause extraordinary circumstances advance notice burden of proof",
        "when passenger is not entitled to compensation cancellation exception defence",
    ),
    "current_amendments": (
        "Amendment No. 2 2025 temporary provision special situation Section 9A exact text",
        "current Israeli Aviation Services Law amendment changed passenger benefits",
    ),
}


def fallback_queries_for(requirement: CoverageRequirement) -> tuple[str, ...]:
    """Return deterministic alternate phrasings for a failed primary recovery."""

    authority = {
        "rights-eu": "EU Regulation 261/2004",
        "rights-il": "Israel Aviation Services Law",
        "rights-us": "US DOT passenger protection regulation",
    }.get(requirement.namespace, "airline contract of carriage")
    return tuple(
        f"{authority} {phrase}"
        for phrase in FALLBACK_TOPIC_PHRASES.get(requirement.topic, ())
    )


def _law_requirements(namespace: str, disruption: str) -> list[CoverageRequirement]:
    documents = PRIMARY_DOCUMENTS.get(namespace, ())
    label = {"rights-eu": "EU261", "rights-il": "Israeli Aviation Services Law"}[namespace]
    event = {
        "cancelled": "cancelled flight cancellation",
        "delayed": "delayed departure long delay",
        "denied_boarding": "denied boarding refusal to transport oversales",
    }.get(disruption, "flight disruption")
    document = documents[0]
    if namespace == "rights-eu":
        event_article = {
            "cancelled": "5",
            "delayed": "6",
            "denied_boarding": "4",
        }.get(disruption, "1")

        def article(number: str) -> tuple[str, ...]:
            return (f"{document}:article_{number}",)

        return [
            CoverageRequirement(
                namespace,
                "applicability",
                f"{label} Article 3(1) passengers departing Member State airport third country Community carrier scope",
                documents,
                article("3"),
            ),
            CoverageRequirement(
                namespace,
                "event_entitlement",
                f"{label} Article {event_article} {event} passenger shall receive assistance",
                documents,
                article(event_article),
            ),
            CoverageRequirement(
                namespace,
                "refund_rerouting",
                f"{label} Article 8 {event} reimbursement refund re-routing passenger choice",
                documents,
                article("8"),
            ),
            CoverageRequirement(
                namespace,
                "care",
                f"{label} Article 9 {event} meals refreshments hotel accommodation airport transport communications",
                documents,
                article("9"),
            ),
            CoverageRequirement(
                namespace,
                "compensation",
                f"{label} Article 7 {event} compensation amount distance band payment",
                documents,
                article("7"),
            ),
            CoverageRequirement(
                namespace,
                "exceptions",
                f"{label} Article 5 cancellation notice extraordinary circumstances burden of proof",
                documents,
                article("5"),
            ),
        ]

    event_section = {
        "cancelled": "6",
        "delayed": "7",
        "denied_boarding": "5",
    }.get(disruption, "2")

    def section(number: str) -> tuple[str, ...]:
        return (f"{document}:section_{number}",)

    return [
        CoverageRequirement(
            namespace,
            "applicability",
            f'{label} Section 1 definition "Flight" departing from or arriving in Israel',
            documents,
            section("1"),
        ),
        CoverageRequirement(
            namespace,
            "eligibility",
            f"{label} Section 2 conditions for entitlement cancelled flight check-in ticket",
            documents,
            section("2"),
        ),
        CoverageRequirement(
            namespace,
            "event_entitlement",
            f"{label} Section {event_section} {event} passenger entitled benefits",
            documents,
            section(event_section),
        ),
        CoverageRequirement(
            namespace,
            "benefit_details",
            f"{label} Section 3 types of benefits assistance services reimbursement replacement flight ticket compensation",
            documents,
            section("3"),
        ),
        CoverageRequirement(
            namespace,
            "refund_rerouting",
            f"{label} Section 3 reimbursement replacement flight ticket passenger choice timing",
            documents,
            section("3"),
        ),
        CoverageRequirement(
            namespace,
            "care",
            f"{label} Section 3 assistance food hotel transport communications telephone",
            documents,
            section("3"),
        ),
        CoverageRequirement(
            namespace,
            "compensation_amounts",
            f"{label} First Schedule monetary compensation distance amount",
            documents,
            (f"{document}:first_schedule",),
        ),
        CoverageRequirement(
            namespace,
            "exceptions",
            f"{label} Section {event_section} {event} exception not entitled burden of proof",
            documents,
            section(event_section),
        ),
    ]


def _us_requirements(disruption: str) -> list[CoverageRequirement]:
    if disruption == "denied_boarding":
        return [
            CoverageRequirement(
                "rights-us",
                "applicability",
                "US DOT 14 CFR Part 250 applicability oversales denied boarding passenger",
                ("us_14_cfr_part_250_oversales",),
                ("us_14_cfr_part_250_oversales:section_250_2",),
            ),
            CoverageRequirement(
                "rights-us",
                "compensation_domestic",
                "US DOT 14 CFR 250.5(a) interstate domestic denied boarding compensation one hour two hours fare percentage",
                ("us_14_cfr_part_250_oversales",),
                ("us_14_cfr_part_250_oversales:section_250_5",),
            ),
            CoverageRequirement(
                "rights-us",
                "compensation_international",
                "US DOT 14 CFR 250.5(b) foreign air transportation denied boarding compensation one hour four hours fare percentage",
                ("us_14_cfr_part_250_oversales",),
                ("us_14_cfr_part_250_oversales:section_250_5",),
            ),
            CoverageRequirement(
                "rights-us",
                "exceptions",
                "US DOT denied boarding compensation exceptions eligibility volunteers",
                ("us_14_cfr_part_250_oversales",),
                ("us_14_cfr_part_250_oversales:section_250_6",),
            ),
            CoverageRequirement(
                "rights-us",
                "notice",
                "US DOT 14 CFR 250.9 written explanation denied boarding compensation boarding priorities",
                ("us_14_cfr_part_250_oversales",),
                ("us_14_cfr_part_250_oversales:section_250_9",),
            ),
            CoverageRequirement(
                "rights-us",
                "compensation_limits",
                "US DOT 2024 adjustment denied boarding compensation limits $1,075 $2,150 effective January 2025",
                ("us_part_250_compensation_adjustment_2024",),
            ),
            CoverageRequirement(
                "rights-us",
                "current_enforcement",
                "US DOT 2025 enforcement date revised denied boarding compensation limits March 20 2025",
                ("us_part_250_adjustment_enforcement_notice_2025",),
            ),
        ]
    requirements = [
        CoverageRequirement(
            "rights-us",
            "applicability",
            "US DOT airline cancellation significant delay refund applicability covered flight",
            ("us_14_cfr_part_260_refunds",),
            ("us_14_cfr_part_260_refunds:section_260_3",),
        ),
        CoverageRequirement(
            "rights-us",
            "refund_rerouting",
            "US DOT cancelled significantly changed flight prompt automatic refund original payment",
            ("us_14_cfr_part_260_refunds",),
            ("us_14_cfr_part_260_refunds:section_260_6",),
        ),
        CoverageRequirement(
            "rights-us",
            "refund_timing",
            "US DOT 14 CFR 260.10 prompt refund timing original form of payment",
            ("us_14_cfr_part_260_refunds",),
            ("us_14_cfr_part_260_refunds:section_260_10",),
        ),
    ]
    if disruption == "delayed":
        requirements.extend(
            (
                CoverageRequirement(
                    "rights-us",
                    "care",
                    "US DOT 14 CFR 259.4 tarmac delay food potable water medical assistance lavatory",
                    ("us_14_cfr_part_259_enhanced_protections",),
                    ("us_14_cfr_part_259_enhanced_protections:section_259_4",),
                ),
                CoverageRequirement(
                    "rights-us",
                    "tarmac_deplaning",
                    "US DOT 14 CFR 259.4 domestic tarmac delay opportunity to deplane before three hours exceptions",
                    ("us_14_cfr_part_259_enhanced_protections",),
                    ("us_14_cfr_part_259_enhanced_protections:section_259_4",),
                ),
            )
        )
    return requirements


def requirements_for(disruption: str | None, namespaces: tuple[str, ...]) -> tuple[CoverageRequirement, ...]:
    disruption = disruption or "unknown"
    requirements: list[CoverageRequirement] = []
    for namespace in namespaces:
        if namespace.startswith("coc-"):
            requirements.append(
                CoverageRequirement(
                    namespace,
                    "airline_policy",
                    f"airline conditions of carriage {disruption} schedule change cancellation "
                    "delay refund rerouting care hotel meals we will carrier shall passenger right",
                )
            )
        elif namespace in {"rights-eu", "rights-il"}:
            requirements.extend(_law_requirements(namespace, disruption))
            if namespace == "rights-il":
                requirements.append(
                    CoverageRequirement(
                        namespace,
                        "current_amendments",
                        "Israeli Aviation Services Law 2025 amendment temporary provision special situation changes passenger benefits",
                        ("il_aviation_services_amendment_2_2025",),
                        ("il_aviation_services_amendment_2_2025:section_9a",),
                    )
                )
        elif namespace == "rights-us":
            requirements.extend(_us_requirements(disruption))
    return tuple(requirements)


def passage_covers(requirement: CoverageRequirement, *, namespace: str, metadata: dict, text: str) -> bool:
    if namespace != requirement.namespace:
        return False
    document_id = str(metadata.get("document_id", ""))
    if requirement.document_ids and document_id not in requirement.document_ids:
        return False
    provision_id = str(metadata.get("provision_id", ""))
    if requirement.provision_ids and provision_id not in requirement.provision_ids:
        return False
    haystack = f"{metadata.get('section_path', '')}\n{text}".lower()
    # Government PDF extraction can split words at a printed line break (for
    # example, ``for-eign`` and ``trans-portation``). Keep the original text for
    # legal punctuation checks and a dehyphenated form for phrase validation.
    dehyphenated = re.sub(r"(?<=\w)-\s*(?=\w)", "", haystack)
    topic = requirement.topic
    if topic == "applicability":
        if requirement.namespace == "rights-eu":
            return "passengers departing from an airport located" in haystack or (
                "departing from an airport located in a third country" in haystack
                and "community carrier" in haystack
            )
        if requirement.namespace == "rights-il":
            return "flight departing from or arriving in the territory of the state of israel" in haystack
        return any(term in haystack for term in ("shall apply", "application", "applicability", "conditions for entitlement", "scope"))
    if topic == "eligibility":
        return any(term in haystack for term in ("conditions for entitlement", "presented themself", "cancelled"))
    if topic == "event_entitlement":
        return any(term in haystack for term in ("cancelled flight", "cancellation of a flight", "delayed departure", "denied boarding", "refusal to transport")) and any(
            term in haystack
            for term in (
                "entitled",
                "shall receive",
                "shall be offered",
                "be offered assistance",
                "have the right",
                "benefits",
            )
        )
    if topic == "refund_rerouting":
        return any(term in haystack for term in ("reimbursement", "refund", "rerouting", "re-routing", "replacement flight", "alternative transport"))
    if topic == "care":
        care_terms = (
            "refreshment",
            "meal",
            "hotel",
            "accommodation",
            "airport and the hotel",
            "telephone calls",
            "communications",
        )
        if requirement.namespace == "rights-us":
            care_terms += (
                "food",
                "potable water",
                "medical attention",
                "operable lavatory",
            )
        return any(term in haystack for term in care_terms)
    if topic == "tarmac_deplaning":
        return "opportunity to deplane" in dehyphenated and any(
            term in dehyphenated for term in ("three hours", "four hours")
        )
    if topic == "compensation":
        return any(
            term in haystack
            for term in (
                "monetary compensation",
                "cash compensation",
                "compensation amount",
                "receive compensation",
                "compensation specified",
                "pay compensation",
                "compensation shall be",
                "denied boarding compensation",
            )
        )
    if topic == "compensation_domestic":
        return (
            "shall pay compensation in interstate air transportation"
            in dehyphenated
            and "less than two hours" in dehyphenated
        )
    if topic == "compensation_international":
        return (
            "shall pay compensation to passengers in foreign air transportation"
            in dehyphenated
            and "less than four hours" in dehyphenated
        )
    if topic == "compensation_amounts":
        return any(term in haystack for term in ("first schedule", "flight distance", "amount of compensation"))
    if topic == "benefit_details":
        return all(term in haystack for term in ("assistance services", "reimbursement", "replacement flight"))
    if topic == "refund_timing":
        return "prompt refunds" in haystack or "refund" in haystack and any(
            term in haystack for term in ("7 business days", "20 calendar days", "original form")
        )
    if topic == "notice":
        return "written" in haystack and any(
            term in haystack for term in ("explanation", "statement", "boarding priority")
        )
    if topic == "compensation_limits":
        return "$1,075" in haystack and "$2,150" in haystack
    if topic == "current_enforcement":
        return "march 20, 2025" in haystack and "enforcement" in haystack
    if topic == "exceptions":
        return any(term in haystack for term in ("extraordinary circumstances", "not entitled", "notice of the cancellation", "burden of proof", "exception"))
    if topic == "airline_policy":
        # A table of contents often contains every disruption keyword but no rule.
        # Require both a relevant subject and language that states an obligation,
        # entitlement, choice, or reserved contractual right.
        section_path = str(metadata.get("section_path", "")).lower()
        if "table of contents" in section_path:
            return False
        has_subject = any(
            term in haystack
            for term in ("cancel", "delay", "schedule", "refund", "rerout", "hotel", "meal")
        )
        has_rule = any(
            term in haystack
            for term in (
                "we will",
                "we shall",
                "carrier will",
                "carrier shall",
                "passenger is entitled",
                "passengers are entitled",
                "you are entitled",
                "shall be offered",
                "right to",
                "reserve the right",
                "may choose",
            )
        )
        return has_subject and has_rule
    if topic == "current_amendments":
        return any(term in haystack for term in ("amendment no. 2", "temporary provision", "special situation", "section 9a"))
    return False


def coverage_report(
    requirements: tuple[CoverageRequirement, ...],
    passages: list,
    *,
    recovery_attempted: tuple[str, ...] = (),
    fallback_attempted: tuple[str, ...] = (),
) -> CoverageReport:
    covered = []
    for requirement in requirements:
        if any(
            passage_covers(
                requirement,
                namespace=passage.namespace,
                metadata=passage.document.metadata,
                text=passage.document.page_content,
            )
            for passage in passages
        ):
            covered.append(requirement.key)
    required = tuple(requirement.key for requirement in requirements)
    covered_tuple = tuple(key for key in required if key in set(covered))
    return CoverageReport(
        required=required,
        covered=covered_tuple,
        missing=tuple(key for key in required if key not in set(covered_tuple)),
        recovery_attempted=recovery_attempted,
        fallback_attempted=fallback_attempted,
    )
