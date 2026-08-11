"""No-cost tests for DocumentationAgent routing, retrieval, and reflection."""

import pytest
from langchain_core.documents import Document

from lib.agents import documentation_agent
from lib.llm import LLMError
from lib.rag.coverage import CoverageReport
from lib.rag.retrieve import PineconeRetriever, RetrievalResult, RetrievedPassage
from lib.rag.routing import route_request
from lib.steps import make_step


def passage(chunk_id="chunk-1", document_id="eu_regulation_261_2004", namespace="rights-eu"):
    return RetrievedPassage(
        document=Document(
            id=chunk_id,
            page_content="In case of cancellation, passengers shall be offered reimbursement or rerouting.",
            metadata={
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_title": "Regulation (EC) No 261/2004",
                "document_type": "regulation",
                "section_path": "Article 8",
                "legal_citation": "EU 261 Art. 8",
                "source_url": "https://example.invalid/eu261",
            },
        ),
        score=0.9,
        namespace=namespace,
    )


def complete_request(**updates):
    request = {
        "airline": "LH",
        "flight_number": "LH318",
        "origin": "TLV",
        "destination": "FRA",
        "disruption": "cancelled",
        "party_size": 1,
        "local_now": "2026-08-10T20:00:00",
        "_passenger_prompt": "My flight was cancelled. What am I owed?",
    }
    request.update(updates)
    return request


def retrieval_result(*passages):
    return RetrievalResult(
        passages=tuple(passages),
        coverage=CoverageReport(required=(), covered=(), missing=()),
        embedding_queries=("test query",),
    )


def test_route_selects_airline_and_both_applicable_laws():
    route = route_request(complete_request())
    assert route.namespaces == ("coc-lh", "rights-il", "rights-eu")


def test_non_eu_carrier_arriving_in_eu_does_not_assume_eu261():
    route = route_request(complete_request(airline="LY", flight_number="LY357"))
    assert route.namespaces == ("coc-ly", "rights-il")


def test_unknown_airports_fall_back_to_all_rights_namespaces():
    route = route_request(complete_request(origin="XXX", destination="YYY"))
    assert route.namespaces == ("coc-lh", "rights-eu", "rights-us", "rights-il")
    assert route.caveats


def test_reflection_makes_three_traced_json_calls(monkeypatch):
    monkeypatch.setattr(
        documentation_agent,
        "retrieve_with_coverage",
        lambda query, namespaces, disruption: retrieval_result(passage()),
    )
    draft = {
        "regulation": "EU 261/2004",
        "entitlements": [],
        "next_actions": [],
        "caveats": ["draft"],
    }
    critique = {"problems": ["Add the supported rerouting entitlement."], "verdict": "revise"}
    refined = {
        "regulation": "EU 261/2004",
        "entitlements": [
            {
                "kind": "rebooking",
                "summary": "Ask for rerouting.",
                "source": "[S1] EU 261 Art. 8",
                "confidence": "high",
            }
        ],
        "next_actions": ["Ask the carrier for rerouting."],
        "caveats": [],
    }
    responses = iter((draft, critique, refined))
    calls = []

    def fake_call(
        module,
        system_prompt,
        user_prompt,
        *,
        expect_json=False,
        max_completion_tokens=None,
    ):
        response = next(responses)
        calls.append(
            (module, system_prompt, user_prompt, expect_json, max_completion_tokens)
        )
        return response, make_step(module, system_prompt, user_prompt, response)

    monkeypatch.setattr(documentation_agent.llm, "call", fake_call)
    payload, steps = documentation_agent.run(complete_request(), [])

    assert payload == refined
    assert len(steps) == 3
    assert [step["module"] for step in steps] == ["DocumentationAgent"] * 3
    assert all(call[3] is True for call in calls)
    assert [call[4] for call in calls] == [10_000, 10_000, 10_000]
    assert all("[S1]" in call[2] for call in calls)
    assert "My flight was cancelled" in calls[0][2]
    assert "Reflection audit contract:" in calls[1][2]
    assert "Reflection audit contract:" in calls[2][2]
    assert "confirm that exact excerpt supports the adjacent claim" in calls[1][1]
    assert "repeat the reflection audit on the revised answer itself" in calls[2][1]


def test_reflection_audit_contract_lists_required_and_missing_topics():
    coverage = CoverageReport(
        required=("rights-us:compensation_domestic", "rights-us:notice"),
        covered=("rights-us:compensation_domestic",),
        missing=("rights-us:notice",),
    )

    checklist = documentation_agent._reflection_audit_checklist(coverage)

    assert "rights-us:compensation_domestic" in checklist
    assert "rights-us:notice" in checklist
    assert "Topics still missing retrieval evidence: rights-us:notice" in checklist
    assert "addressed, inapplicable" in checklist


def test_final_payload_accepts_refund_and_rejects_unknown_entitlement_kind():
    refund = {
        "regulation": "US DOT",
        "entitlements": [
            {
                "kind": "refund",
                "summary": "Return the unused airfare.",
                "source": "[S1] 14 CFR Part 260",
                "confidence": "high",
            }
        ],
        "next_actions": [],
        "caveats": [],
    }

    assert documentation_agent._validate_payload(refund) == refund

    invalid = {**refund, "entitlements": [{**refund["entitlements"][0], "kind": "voucher"}]}
    with pytest.raises(ValueError, match="invalid kind: voucher"):
        documentation_agent._validate_payload(invalid)


def test_draft_and_refinement_schemas_offer_refund_as_its_own_kind():
    assert '"refund"' in documentation_agent.DRAFT_SYSTEM_PROMPT
    assert '"refund"' in documentation_agent.REFINE_SYSTEM_PROMPT


def test_failure_keeps_successful_and_failed_call_steps(monkeypatch):
    monkeypatch.setattr(
        documentation_agent,
        "retrieve_with_coverage",
        lambda query, namespaces, disruption: retrieval_result(passage()),
    )
    drafted = {
        "regulation": "EU 261/2004",
        "entitlements": [],
        "next_actions": [],
        "caveats": [],
    }
    failed = make_step("DocumentationAgent", "critique", "user", {"error": "timed out"})
    calls = 0

    def fake_call(
        module,
        system_prompt,
        user_prompt,
        *,
        expect_json=False,
        max_completion_tokens=None,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            return drafted, make_step(module, system_prompt, user_prompt, drafted)
        raise LLMError("timed out", steps=[failed])

    monkeypatch.setattr(documentation_agent.llm, "call", fake_call)
    with pytest.raises(LLMError) as caught:
        documentation_agent.run(complete_request(), [])

    assert len(caught.value.steps) == 2
    assert caught.value.steps[1] == failed


def test_selected_evidence_is_not_truncated():
    long_text = "legal clause " * 2_000
    item = passage()
    item.document.page_content = long_text

    context = documentation_agent._evidence_context([item])

    assert context.endswith(long_text)


def test_recovery_evidence_is_prioritized_without_dropping_required_topics():
    ordinary = [passage(chunk_id=f"ordinary-{number}") for number in range(10)]
    recovered = [
        RetrievedPassage(
            document=passage(chunk_id=f"recovered-{number}").document,
            score=0.5,
            namespace="rights-eu",
            recovery_added=True,
            selection_reason=f"coverage_recovery:topic-{number}",
        )
        for number in range(18)
    ]

    selected = documentation_agent._select_passages(ordinary + recovered)

    assert len(selected) == documentation_agent.MAX_SOURCES
    assert all(item in selected for item in recovered)
    assert sum(item.recovery_added for item in selected) == 18


def test_companion_document_never_displaces_recovery_evidence(monkeypatch):
    monkeypatch.setattr(documentation_agent, "MAX_SOURCES", 2)
    recovered = [
        RetrievedPassage(
            document=passage(chunk_id=f"required-{number}").document,
            score=0.5,
            namespace="rights-eu",
            recovery_added=True,
            selection_reason=f"coverage_recovery:topic-{number}",
        )
        for number in range(2)
    ]
    companion = RetrievedPassage(
        document=passage(chunk_id="companion").document,
        score=0.4,
        namespace="rights-eu",
        relationship_added=True,
        selection_reason="companion_document",
    )

    selected = documentation_agent._select_passages(recovered + [companion])

    assert selected == recovered


def test_reflection_limit_can_be_raised_to_two_rounds(monkeypatch):
    monkeypatch.setattr(documentation_agent, "MAX_REFLECTION_ROUNDS", 2)
    monkeypatch.setattr(
        documentation_agent,
        "retrieve_with_coverage",
        lambda query, namespaces, disruption: retrieval_result(passage()),
    )
    assessment = {
        "regulation": "EU 261/2004",
        "entitlements": [],
        "next_actions": [],
        "caveats": [],
    }
    responses = iter(
        (
            assessment,
            {"problems": ["first issue"], "verdict": "revise"},
            assessment,
            {"problems": [], "verdict": "accept"},
            assessment,
        )
    )

    def fake_call(module, system_prompt, user_prompt, **kwargs):
        response = next(responses)
        return response, make_step(module, system_prompt, user_prompt, response)

    monkeypatch.setattr(documentation_agent.llm, "call", fake_call)
    payload, steps = documentation_agent.run(complete_request(), [])

    assert payload == assessment
    assert len(steps) == 5  # one draft plus two critique/refine rounds
    assert "Reflection round: 2 of 2" in steps[3]["prompt"]["user_prompt"]


def test_accept_verdict_stops_before_unused_later_rounds(monkeypatch):
    monkeypatch.setattr(documentation_agent, "MAX_REFLECTION_ROUNDS", 3)
    monkeypatch.setattr(
        documentation_agent,
        "retrieve_with_coverage",
        lambda query, namespaces, disruption: retrieval_result(passage()),
    )
    assessment = {
        "regulation": "EU 261/2004",
        "entitlements": [],
        "next_actions": [],
        "caveats": [],
    }
    responses = iter(
        (
            assessment,
            {"problems": [], "verdict": "accept"},
            assessment,
        )
    )

    def fake_call(module, system_prompt, user_prompt, **kwargs):
        response = next(responses)
        return response, make_step(module, system_prompt, user_prompt, response)

    monkeypatch.setattr(documentation_agent.llm, "call", fake_call)
    _, steps = documentation_agent.run(complete_request(), [])

    assert len(steps) == 3


class FakeEmbeddings:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.0] * 1536


class BatchFakeEmbeddings:
    def __init__(self):
        self.document_calls = []
        self.query_by_code = {}

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        vectors = []
        first_code = len(self.query_by_code) + 1
        for code, query in enumerate(texts, start=first_code):
            self.query_by_code[code] = query
            vectors.append([float(code)] + [0.0] * 1535)
        return vectors


class CoverageFakeIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.queries = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        code = int(kwargs["vector"][0])
        query = self.embeddings.query_by_code[code]
        metadata_filter = kwargs.get("filter") or {}
        clauses = metadata_filter.get("$and", [metadata_filter])
        fields = {
            key: value
            for clause in clauses
            for key, value in clause.items()
        }
        document_filter = fields.get("document_id", {})
        document_id = document_filter.get("$eq")
        if not document_id and document_filter.get("$in"):
            document_id = document_filter["$in"][0]
        provision_filter = fields.get("provision_id", {})
        provision_id = provision_filter.get("$eq")
        if not provision_id and provision_filter.get("$in"):
            provision_id = provision_filter["$in"][0]
        if not document_id:
            document_id = (
                f"{kwargs['namespace']}-general"
                if code == 1
                else f"{kwargs['namespace']}-policy"
            )
        if code == 1:
            text = "generic introductory language"
        elif provision_id == "eu_regulation_261_2004:article_3":
            text = "This Regulation applies to passengers departing from an airport located in a Member State."
        elif provision_id == "il_aviation_services_law_2012:section_1":
            text = '"Flight" means a flight departing from or arriving in the territory of the State of Israel.'
        else:
            text = query
        return {
            "matches": [
                {
                    "id": f"{kwargs['namespace']}-{code}-{document_id}",
                    "score": 0.9,
                    "metadata": {
                        "text": text,
                        "document_id": document_id,
                        "provision_id": provision_id,
                        "chunk_id": f"{kwargs['namespace']}-{code}-{document_id}",
                        "section_path": query,
                    },
                }
            ]
        }


class ConditionalFallbackIndex(CoverageFakeIndex):
    def query(self, **kwargs):
        self.queries.append(kwargs)
        code = int(kwargs["vector"][0])
        query = self.embeddings.query_by_code[code]
        is_fallback = "contractual obligation after schedule irregularity" in query
        text = (
            "After a schedule cancellation, the carrier will refund or transport the passenger."
            if is_fallback
            else "generic introductory language"
        )
        return {
            "matches": [
                {
                    "id": f"coc-lh-{code}",
                    "score": 0.9,
                    "metadata": {
                        "text": text,
                        "document_id": "lh_coc",
                        "chunk_id": f"coc-lh-{code}",
                        "section_path": "Schedule irregularities",
                    },
                }
            ]
        }


class FakeIndex:
    def __init__(self):
        self.queries = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        metadata_filter = kwargs.get("filter", {})
        clauses = metadata_filter.get("$and", [metadata_filter])
        fields = {key: value for clause in clauses for key, value in clause.items()}
        document_id = fields.get("document_id", {}).get("$eq") or (
            "eu_interpretative_guidelines_2024"
            if kwargs["namespace"] == "rights-eu"
            else "lh_coc"
        )
        return {
            "matches": [
                {
                    "id": f"{document_id}-chunk",
                    "score": 0.9,
                    "metadata": {
                        "text": "relevant text",
                        "document_id": document_id,
                        "chunk_id": f"{document_id}-chunk",
                        "chunker_version": "legal-markdown-v3-provisions",
                    },
                }
            ]
        }


def test_multi_namespace_retrieval_embeds_query_once_and_adds_companion_law():
    embeddings = FakeEmbeddings()
    index = FakeIndex()
    retriever = PineconeRetriever(index, embeddings)

    hits = retriever.search(
        "cancelled flight",
        namespaces=("coc-lh", "rights-eu"),
        k=2,
        per_namespace_k=1,
    )

    assert embeddings.queries == ["cancelled flight"]
    assert {hit.namespace for hit in hits[:2]} == {"coc-lh", "rights-eu"}
    assert any(
        hit.document.metadata["document_id"] == "eu_regulation_261_2004"
        and hit.relationship_added
        for hit in hits
    )


def test_coverage_recovery_batches_embeddings_and_refetches_missing_topics():
    embeddings = BatchFakeEmbeddings()
    index = CoverageFakeIndex(embeddings)
    retriever = PineconeRetriever(index, embeddings)

    result = retriever.search_with_coverage(
        "passenger problem",
        namespaces=("coc-lh", "rights-eu", "rights-il"),
        disruption="cancelled",
        per_namespace_k=2,
    )

    assert len(embeddings.document_calls) == 1
    assert result.embedding_batches == 1
    assert result.coverage.is_complete
    assert result.coverage.recovery_attempted
    assert not result.coverage.fallback_attempted
    assert any(hit.recovery_added for hit in result.passages)
    assert {
        hit.namespace for hit in result.passages if hit.recovery_added
    } == {"coc-lh", "rights-eu", "rights-il"}


def test_incomplete_primary_recovery_uses_one_conditional_fallback_batch():
    embeddings = BatchFakeEmbeddings()
    index = ConditionalFallbackIndex(embeddings)
    retriever = PineconeRetriever(index, embeddings)

    result = retriever.search_with_coverage(
        "passenger problem",
        namespaces=("coc-lh",),
        disruption="cancelled",
        per_namespace_k=2,
    )

    assert len(embeddings.document_calls) == 2
    assert result.embedding_batches == 2
    assert result.coverage.is_complete
    assert result.coverage.fallback_attempted == ("coc-lh:airline_policy",)
    assert any(
        hit.selection_reason == "coverage_fallback:airline_policy:variant_1"
        for hit in result.passages
    )
    assert any(query["top_k"] == 10 for query in index.queries)
