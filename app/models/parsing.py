"""Parsing-related Pydantic models."""

from typing import Any

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """Extracted text block with layout information."""

    content: str
    page: int = Field(ge=0, description="Page number (non-negative)")
    bbox: tuple[float, float, float, float]
    font_size: float | None = None
    layout_type: str | None = Field(
        default=None, description="Layout type: heading, paragraph, list, etc."
    )


class ImageBlock(BaseModel):
    """Extracted image with visual features."""

    image_data: bytes
    page: int = Field(ge=0, description="Page number (non-negative)")
    bbox: tuple[float, float, float, float]
    format: str
    visual_features: dict[str, Any] | None = Field(
        default=None, description="Visual features extracted from image"
    )


class ChartBlock(BaseModel):
    """Extracted chart or diagram."""

    image_data: bytes
    page: int = Field(ge=0, description="Page number (non-negative)")
    bbox: tuple[float, float, float, float]
    chart_type: str | None = Field(
        default=None, description="Chart type: bar, line, pie, scatter, etc."
    )


class TableBlock(BaseModel):
    """Extracted table with structure."""

    rows: list[list[str]]
    page: int = Field(ge=0, description="Page number (non-negative)")
    bbox: tuple[float, float, float, float]
    headers: list[str] | None = Field(default=None, description="Table headers if detected")


class ParsedDocument(BaseModel):
    """Complete parsed document with multimodal content."""

    text_blocks: list[TextBlock]
    images: list[ImageBlock]
    charts: list[ChartBlock] = Field(default_factory=list)
    tables: list[TableBlock]
    num_pages: int = Field(ge=0, description="Number of pages (non-negative)")
    metadata: dict[str, Any]
