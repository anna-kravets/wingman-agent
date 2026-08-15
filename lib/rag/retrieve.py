"""LangChain query embeddings plus namespace-scoped Pinecone retrieval."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from pinecone import Pinecone

from lib.llm import paid_calls_enabled
from lib.rag.coverage import (
    CoverageReport,
    coverage_report,
    fallback_queries_for,
    passage_covers,
    requirements_for,
)
from lib.rag.llmod_embeddings import LLModEmbeddings


EMBEDDING_MODEL = "MB5R2CF-azure/text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
DEFAULT_INDEX_NAME = "wingman-legal-docs"
PROVISION_CHUNKER_VERSION = "legal-markdown-v3-provisions"
COC_CHUNKER_VERSION = "legal-markdown-v2"
ALL_NAMESPACES = (
    "coc-aa", "coc-ac", "coc-af", "coc-al", "coc-ba", "coc-dl", "coc-ek",
    "coc-fr", "coc-lh", "coc-ly", "coc-rk", "coc-rr", "coc-ua",
    "rights-eu", "rights-il", "rights-us",
)
RELATIONSHIPS = {
    "eu_interpretative_guidelines_2024": ("eu_regulation_261_2004",),
    "eu_regulation_261_2004": ("eu_interpretative_guidelines_2024",),
    "us_14_cfr_part_250_oversales": (
        "us_part_250_compensation_adjustment_2024",
        "us_part_250_adjustment_enforcement_notice_2025",
    ),
    "us_part_250_compensation_adjustment_2024": (
        "us_14_cfr_part_250_oversales",
        "us_part_250_adjustment_enforcement_notice_2025",
    ),
    "us_part_250_adjustment_enforcement_notice_2025": (
        "us_14_cfr_part_250_oversales",
        "us_part_250_compensation_adjustment_2024",
    ),
    "us_14_cfr_part_259_enhanced_protections": (
        "us_part_259_amendment_passenger_rights_summary_2026",
    ),
    "us_part_259_amendment_passenger_rights_summary_2026": (
        "us_14_cfr_part_259_enhanced_protections",
    ),
}


def _chunker_version_filter(namespace: str) -> dict[str, dict[str, str]]:
    version = (
        COC_CHUNKER_VERSION
        if namespace.startswith("coc-")
        else PROVISION_CHUNKER_VERSION
    )
    return {"chunker_version": {"$eq": version}}


def _get(value: object, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _required(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise ValueError(f"Missing required environment variable: {' or '.join(names)}")


@dataclass(frozen=True)
class RetrievedPassage:
    document: Document
    score: float
    namespace: str
    relationship_added: bool = False
    recovery_added: bool = False
    selection_reason: str = "semantic"


@dataclass(frozen=True)
class RetrievalResult:
    passages: tuple[RetrievedPassage, ...]
    coverage: CoverageReport
    embedding_queries: tuple[str, ...]
    embedding_batches: int = 1


class PineconeRetriever:
    """Search namespaces with one primary and at most one fallback embedding batch."""

    def __init__(self, index: Any, embeddings: Any) -> None:
        self.index = index
        self.embeddings = embeddings

    @staticmethod
    def _passage(match: object, namespace: str, *, related: bool = False) -> RetrievedPassage:
        metadata = dict(_get(match, "metadata", {}) or {})
        text = str(metadata.pop("text", ""))
        chunk_id = str(_get(match, "id", metadata.get("chunk_id", "")))
        metadata.setdefault("chunk_id", chunk_id)
        return RetrievedPassage(
            document=Document(page_content=text, metadata=metadata, id=chunk_id),
            score=float(_get(match, "score", 0.0)),
            namespace=namespace,
            relationship_added=related,
        )

    def search(
        self,
        query: str,
        *,
        namespaces: tuple[str, ...],
        k: int = 6,
        per_namespace_k: int = 4,
        include_related: bool = True,
    ) -> list[RetrievedPassage]:
        if not query.strip():
            raise ValueError("Retrieval query must not be empty")
        if not namespaces:
            raise ValueError("At least one Pinecone namespace is required")
        unknown = sorted(set(namespaces) - set(ALL_NAMESPACES))
        if unknown:
            raise ValueError(f"Unknown Pinecone namespace(s): {', '.join(unknown)}")

        # This is the only embedding-model call in one DocumentationAgent run.
        vector = self.embeddings.embed_query(query)
        if len(vector) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding dimension is {len(vector)}; expected {EMBEDDING_DIMENSION}"
            )

        by_namespace: dict[str, list[RetrievedPassage]] = {}
        all_hits: list[RetrievedPassage] = []
        for namespace in namespaces:
            response = self.index.query(
                vector=vector,
                namespace=namespace,
                top_k=per_namespace_k,
                filter=_chunker_version_filter(namespace),
                include_metadata=True,
                include_values=False,
            )
            hits = [
                self._passage(match, namespace)
                for match in (_get(response, "matches", []) or [])
            ]
            hits.sort(key=lambda hit: hit.score, reverse=True)
            by_namespace[namespace] = hits
            all_hits.extend(hits)

        # Preserve at least the strongest result from every requested namespace,
        # then fill the remaining slots globally by score.
        selected = [hits[0] for hits in by_namespace.values() if hits]
        selected_ids = {
            str(hit.document.metadata.get("chunk_id", hit.document.id or ""))
            for hit in selected
        }
        for hit in sorted(all_hits, key=lambda item: item.score, reverse=True):
            chunk_id = str(hit.document.metadata.get("chunk_id", hit.document.id or ""))
            if chunk_id not in selected_ids:
                selected.append(hit)
                selected_ids.add(chunk_id)
            if len(selected) >= max(k, len(by_namespace)):
                break

        if include_related:
            present_documents = {
                str(hit.document.metadata.get("document_id", "")) for hit in selected
            }
            required: list[tuple[str, str]] = []
            for hit in selected:
                metadata = hit.document.metadata
                document_id = str(metadata.get("document_id", ""))
                related_ids = list(RELATIONSHIPS.get(document_id, ()))
                for field in ("must_read_with_document_id", "amends_document_id"):
                    if metadata.get(field):
                        related_ids.append(str(metadata[field]))
                for related_id in related_ids:
                    if related_id not in present_documents:
                        required.append((related_id, hit.namespace))
                        present_documents.add(related_id)

            for document_id, namespace in required:
                response = self.index.query(
                    vector=vector,
                    namespace=namespace,
                    top_k=1,
                    filter={
                        "$and": [
                            {"document_id": {"$eq": document_id}},
                            _chunker_version_filter(namespace),
                        ]
                    },
                    include_metadata=True,
                    include_values=False,
                )
                matches = _get(response, "matches", []) or []
                if matches:
                    selected.append(self._passage(matches[0], namespace, related=True))

        return selected

    def search_with_coverage(
        self,
        query: str,
        *,
        namespaces: tuple[str, ...],
        disruption: str | None,
        initial_k: int = 8,
        per_namespace_k: int = 8,
        minimum_per_namespace: int = 2,
        recovery_k: int = 4,
        fallback_recovery_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve broadly, then deterministically recover missing legal topics.

        The initial and primary-recovery queries are embedded in one batch. Only when
        coverage remains incomplete is a second batch of alternate queries embedded.
        """

        if not query.strip():
            raise ValueError("Retrieval query must not be empty")
        if not namespaces:
            raise ValueError("At least one Pinecone namespace is required")
        unknown = sorted(set(namespaces) - set(ALL_NAMESPACES))
        if unknown:
            raise ValueError(f"Unknown Pinecone namespace(s): {', '.join(unknown)}")

        requirements = requirements_for(disruption, namespaces)
        primary_queries = tuple(dict.fromkeys((query, *(item.query for item in requirements))))

        def embed_batch(queries: tuple[str, ...]) -> dict[str, list[float]]:
            vectors = self.embeddings.embed_documents(list(queries))
            if len(vectors) != len(queries):
                raise RuntimeError("Embedding response count does not match coverage query count")
            if any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
                dimensions = sorted({len(vector) for vector in vectors})
                raise RuntimeError(
                    f"Unexpected embedding dimension(s): {dimensions}; expected {EMBEDDING_DIMENSION}"
                )
            return dict(zip(queries, vectors, strict=True))

        vector_by_query = embed_batch(primary_queries)
        all_embedding_queries = list(primary_queries)
        embedding_batches = 1

        by_namespace: dict[str, list[RetrievedPassage]] = {}
        all_initial: list[RetrievedPassage] = []
        for namespace in namespaces:
            response = self.index.query(
                vector=vector_by_query[query],
                namespace=namespace,
                top_k=per_namespace_k,
                filter=_chunker_version_filter(namespace),
                include_metadata=True,
                include_values=False,
            )
            hits = [
                self._passage(match, namespace)
                for match in (_get(response, "matches", []) or [])
            ]
            hits.sort(key=lambda hit: hit.score, reverse=True)
            by_namespace[namespace] = hits
            all_initial.extend(hits)

        # Round-robin selection prevents one high-scoring generic document from
        # crowding applicable laws out of the initial evidence set.
        selected: list[RetrievedPassage] = []
        selected_ids: set[str] = set()

        def add(passage: RetrievedPassage) -> bool:
            chunk_id = str(
                passage.document.metadata.get("chunk_id", passage.document.id or "")
            )
            if not chunk_id or chunk_id in selected_ids:
                return False
            selected.append(passage)
            selected_ids.add(chunk_id)
            return True

        for rank in range(minimum_per_namespace):
            for namespace in namespaces:
                hits = by_namespace.get(namespace, [])
                if rank < len(hits):
                    add(hits[rank])
        for hit in sorted(all_initial, key=lambda item: item.score, reverse=True):
            if len(selected) >= max(initial_k, len(namespaces) * minimum_per_namespace):
                break
            add(hit)

        initial_report = coverage_report(requirements, selected)
        missing_by_key = {
            requirement.key: requirement
            for requirement in requirements
            if requirement.key in set(initial_report.missing)
        }
        attempted: list[str] = []
        def recovery_candidates(
            requirement,
            focused_query: str,
            *,
            top_k: int,
        ) -> list[RetrievedPassage]:
            filter_clauses: list[dict[str, object]] = []
            filter_clauses.append(_chunker_version_filter(requirement.namespace))
            if len(requirement.document_ids) == 1:
                filter_clauses.append(
                    {"document_id": {"$eq": requirement.document_ids[0]}}
                )
            elif requirement.document_ids:
                filter_clauses.append(
                    {"document_id": {"$in": list(requirement.document_ids)}}
                )
            if len(requirement.provision_ids) == 1:
                filter_clauses.append(
                    {"provision_id": {"$eq": requirement.provision_ids[0]}}
                )
            elif requirement.provision_ids:
                filter_clauses.append(
                    {"provision_id": {"$in": list(requirement.provision_ids)}}
                )
            metadata_filter = None
            if len(filter_clauses) == 1:
                metadata_filter = filter_clauses[0]
            elif filter_clauses:
                metadata_filter = {"$and": filter_clauses}
            query_args = dict(
                vector=vector_by_query[focused_query],
                namespace=requirement.namespace,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
            )
            if metadata_filter is not None:
                query_args["filter"] = metadata_filter
            response = self.index.query(**query_args)
            candidates = [
                self._passage(match, requirement.namespace)
                for match in (_get(response, "matches", []) or [])
            ]
            candidates.sort(key=lambda hit: hit.score, reverse=True)
            return [
                hit
                for hit in candidates
                if passage_covers(
                    requirement,
                    namespace=hit.namespace,
                    metadata=hit.document.metadata,
                    text=hit.document.page_content,
                )
            ]

        def add_recovered(passage: RetrievedPassage, *, reason: str) -> None:
            add(
                RetrievedPassage(
                    document=passage.document,
                    score=passage.score,
                    namespace=passage.namespace,
                    recovery_added=True,
                    selection_reason=reason,
                )
            )

        for key in initial_report.missing:
            requirement = missing_by_key[key]
            attempted.append(key)
            matching = recovery_candidates(
                requirement,
                requirement.query,
                top_k=recovery_k,
            )
            if matching:
                add_recovered(
                    matching[0],
                    reason=f"coverage_recovery:{requirement.topic}",
                )

        primary_recovery_report = coverage_report(
            requirements,
            selected,
            recovery_attempted=tuple(attempted),
        )
        fallback_attempted: list[str] = []
        if primary_recovery_report.missing:
            fallback_requirements = [
                missing_by_key[key] for key in primary_recovery_report.missing
            ]
            fallback_queries = tuple(
                dict.fromkeys(
                    alternate
                    for requirement in fallback_requirements
                    for alternate in fallback_queries_for(requirement)
                )
            )
            if fallback_queries:
                vector_by_query.update(embed_batch(fallback_queries))
                all_embedding_queries.extend(fallback_queries)
                embedding_batches += 1

                for requirement in fallback_requirements:
                    variants = fallback_queries_for(requirement)
                    if not variants:
                        continue
                    fallback_attempted.append(requirement.key)
                    for variant_number, alternate_query in enumerate(variants, start=1):
                        matching = recovery_candidates(
                            requirement,
                            alternate_query,
                            top_k=fallback_recovery_k,
                        )
                        if matching:
                            add_recovered(
                                matching[0],
                                reason=(
                                    f"coverage_fallback:{requirement.topic}:"
                                    f"variant_{variant_number}"
                                ),
                            )
                            break

        final_report = coverage_report(
            requirements,
            selected,
            recovery_attempted=tuple(attempted),
            fallback_attempted=tuple(fallback_attempted),
        )
        return RetrievalResult(
            passages=tuple(selected),
            coverage=final_report,
            embedding_queries=tuple(all_embedding_queries),
            embedding_batches=embedding_batches,
        )


@lru_cache(maxsize=1)
def _production_retriever() -> PineconeRetriever:
    # Same guard as lib.llm._config, read at call time so api/index.py's load_dotenv()
    # cannot re-arm it: this path spends embeddings and Pinecone queries and reads
    # LLMOD_API_KEY directly, without going through lib.llm.call.
    if not paid_calls_enabled():
        raise ValueError("Paid retrieval is disabled (WINGMAN_ALLOW_LLM=0).")
    api_key = _required("LLMOD_API_KEY")
    base_url = _required("LLMOD_API_BASE").rstrip("/")
    pinecone_key = _required("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME).strip() or DEFAULT_INDEX_NAME

    embeddings = LLModEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=EMBEDDING_MODEL,
    )
    client = Pinecone(api_key=pinecone_key)
    return PineconeRetriever(client.Index(index_name), embeddings)


def retrieve(
    query: str,
    *,
    namespaces: tuple[str, ...],
    k: int = 6,
    per_namespace_k: int = 4,
) -> list[RetrievedPassage]:
    """Retrieve production evidence; construction is cached across warm requests."""

    return _production_retriever().search(
        query,
        namespaces=namespaces,
        k=k,
        per_namespace_k=per_namespace_k,
        include_related=True,
    )


def retrieve_with_coverage(
    query: str,
    *,
    namespaces: tuple[str, ...],
    disruption: str | None,
) -> RetrievalResult:
    """Production retrieval with deterministic missing-topic recovery."""

    return _production_retriever().search_with_coverage(
        query,
        namespaces=namespaces,
        disruption=disruption,
    )
