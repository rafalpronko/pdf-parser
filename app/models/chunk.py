"""Chunk-related Pydantic models for multimodal content."""

from typing import Any

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """Text-only chunk."""

    chunk_id: str
    doc_id: str
    content: str
    page: int = Field(ge=0, description="Page number (non-negative)")
    chunk_index: int = Field(ge=0, description="Chunk index (non-negative)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualChunk(BaseModel):
    """Visual-only chunk (image or chart)."""

    chunk_id: str
    doc_id: str
    image_data: bytes
    page: int = Field(ge=0, description="Page number (non-negative)")
    chunk_index: int = Field(ge=0, description="Chunk index (non-negative)")
    visual_type: str = Field(description="Visual type: image, chart, diagram")
    caption: str | None = Field(default=None, description="Caption or description")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultimodalChunk(BaseModel):
    """Multimodal chunk combining text and visual elements."""

    chunk_id: str
    doc_id: str
    text_content: str
    visual_elements: list[VisualChunk] = Field(default_factory=list)
    page: int = Field(ge=0, description="Page number (non-negative)")
    chunk_index: int = Field(ge=0, description="Chunk index (non-negative)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    """Chunk with embedding vector."""

    chunk: TextChunk | VisualChunk | MultimodalChunk
    embedding: list[float]
    modality: str = Field(description="Modality: text, visual, or multimodal")


# Legacy alias for backward compatibility
DocumentChunk = TextChunk
