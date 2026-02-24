"""Pydantic models for the multimodal PDF RAG system."""

from app.models.chunk import (
    DocumentChunk,
    EmbeddedChunk,
    MultimodalChunk,
    TextChunk,
    VisualChunk,
)
from app.models.document import (
    DocumentInfo,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.models.error import ErrorResponse
from app.models.parsing import (
    ChartBlock,
    ImageBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)
from app.models.query import (
    MultimodalQueryResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
)

__all__ = [
    # Document models
    "DocumentMetadata",
    "DocumentUploadResponse",
    "DocumentInfo",
    # Parsing models
    "TextBlock",
    "ImageBlock",
    "ChartBlock",
    "TableBlock",
    "ParsedDocument",
    # Chunk models
    "TextChunk",
    "VisualChunk",
    "MultimodalChunk",
    "DocumentChunk",  # Legacy alias
    "EmbeddedChunk",
    # Query models
    "QueryRequest",
    "SourceReference",
    "QueryResponse",  # Legacy alias
    "MultimodalQueryResponse",
    # Error models
    "ErrorResponse",
]
