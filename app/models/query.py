"""Query-related Pydantic models for multimodal queries."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Multimodal query request."""

    question: str
    top_k: int = Field(default=15, ge=1, le=20, description="Number of results to retrieve")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    include_sources: bool = Field(default=True, description="Include source references")
    include_visual: bool = Field(default=True, description="Include visual content in retrieval")
    modality_filter: str | None = Field(
        default=None, description="Filter by modality: text, visual, multimodal, or None for all"
    )


class SourceReference(BaseModel):
    """Source citation with modality information."""

    doc_id: str
    filename: str
    page: int = Field(ge=0, description="Page number (non-negative)")
    chunk_content: str
    modality: str = Field(description="Modality: text, visual, or multimodal")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score between 0 and 1")
    visual_description: str | None = Field(
        default=None, description="VLM-generated description if visual"
    )


class MultimodalQueryResponse(BaseModel):
    """Multimodal query response."""

    answer: str
    sources: list[SourceReference]
    visual_sources: list[SourceReference] = Field(
        default_factory=list, description="Visual-specific sources"
    )
    processing_time: float = Field(ge=0.0, description="Processing time in seconds (non-negative)")
    modalities_used: list[str] = Field(
        default_factory=list, description="List of modalities used in response"
    )


# Legacy alias for backward compatibility
QueryResponse = MultimodalQueryResponse
