"""Unified search result model used across retrieval and storage layers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Unified search result from vector database, hybrid search, and reranking.

    This is the single source of truth for search results across the entire
    retrieval pipeline: vector store, hybrid search, BM25, and reranker.

    Attributes:
        chunk_id: Unique chunk identifier
        content: Chunk text content
        score: Relevance/ranking score (higher is better)
        doc_id: Document identifier
        page: Page number in the source document
        chunk_index: Chunk index within the document
        metadata: Additional metadata (source-specific)
    """

    chunk_id: str
    content: str
    score: float
    doc_id: str = ""
    page: int = 0
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "doc_id": self.doc_id,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
