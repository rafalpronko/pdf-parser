"""Document-related Pydantic models."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for uploaded document."""

    filename: str
    content_type: str = "application/pdf"
    tags: list[str] = []
    description: str | None = None


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    doc_id: str
    filename: str
    status: str
    message: str
    created_at: datetime


class DocumentInfo(BaseModel):
    """Document information."""

    doc_id: str
    filename: str
    file_size: int = Field(ge=0, description="File size in bytes (non-negative)")
    num_pages: int = Field(ge=0, description="Number of pages (non-negative)")
    num_chunks: int = Field(ge=0, description="Number of chunks (non-negative)")
    created_at: datetime
    tags: list[str]
