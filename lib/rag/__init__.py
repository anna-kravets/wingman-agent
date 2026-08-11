"""Runtime retrieval support for Wingman's DocumentationAgent."""

from lib.rag.coverage import CoverageReport, CoverageRequirement
from lib.rag.retrieve import RetrievalResult, RetrievedPassage, retrieve, retrieve_with_coverage
from lib.rag.routing import RetrievalRoute, route_request

__all__ = [
    "CoverageReport",
    "CoverageRequirement",
    "RetrievalResult",
    "RetrievedPassage",
    "RetrievalRoute",
    "retrieve",
    "retrieve_with_coverage",
    "route_request",
]
