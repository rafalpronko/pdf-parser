"""Property-based tests for Pydantic model validation.

Feature: pdf-rag-system
Properties tested:
- Property 21: Pydantic validation enforcement
- Property 22: Response serialization consistency
"""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.models import (
    ChartBlock,
    EmbeddedChunk,
    ImageBlock,
    MultimodalChunk,
    MultimodalQueryResponse,
    ParsedDocument,
    QueryRequest,
    SourceReference,
    TableBlock,
    TextBlock,
    TextChunk,
    VisualChunk,
)


# Feature: pdf-rag-system, Property 21: Pydantic validation enforcement
@given(
    page=st.integers(min_value=-10, max_value=1000),
    chunk_index=st.integers(min_value=-10, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_property_21_invalid_fields_rejected(page: int, chunk_index: int):
    """Property 21: Pydantic validation enforcement.

    For any API request with invalid data (missing required fields, wrong types,
    out-of-range values), the system should reject the request and return detailed
    field-level error messages.

    Validates: Requirements 6.1, 6.2, 6.5
    """
    # Negative page numbers should be rejected
    if page < 0:
        with pytest.raises(ValidationError) as exc_info:
            TextChunk(
                chunk_id="test",
                doc_id="doc1",
                content="test content",
                page=page,
                chunk_index=0,
            )
        assert "page" in str(exc_info.value).lower()

    # Negative chunk_index should be rejected
    if chunk_index < 0:
        with pytest.raises(ValidationError) as exc_info:
            TextChunk(
                chunk_id="test",
                doc_id="doc1",
                content="test content",
                page=0,
                chunk_index=chunk_index,
            )
        assert "chunk_index" in str(exc_info.value).lower()

    # Valid values should succeed
    if page >= 0 and chunk_index >= 0:
        chunk = TextChunk(
            chunk_id="test",
            doc_id="doc1",
            content="test content",
            page=page,
            chunk_index=chunk_index,
        )
        assert chunk.page == page
        assert chunk.chunk_index == chunk_index


# Feature: pdf-rag-system, Property 22: Response serialization consistency
@given(
    answer=st.text(min_size=1, max_size=1000),
    processing_time=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    num_sources=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=100, deadline=None)
def test_property_22_response_serialization(answer: str, processing_time: float, num_sources: int):
    """Property 22: Response serialization consistency.

    For any API response, the returned JSON should validate against the
    corresponding Pydantic response model schema, ensuring all required
    fields are present and types are correct.

    Validates: Requirements 6.3
    """
    # Create sources
    sources = [
        SourceReference(
            doc_id=f"doc{i}",
            filename=f"file{i}.pdf",
            page=i,
            chunk_content=f"content {i}",
            modality="text",
            relevance_score=0.9,
        )
        for i in range(num_sources)
    ]

    # Create response
    response = MultimodalQueryResponse(
        answer=answer,
        sources=sources,
        processing_time=processing_time,
        modalities_used=["text"],
    )

    # Serialize to JSON
    json_str = response.model_dump_json()
    json_dict = json.loads(json_str)

    # Deserialize back
    response_restored = MultimodalQueryResponse.model_validate(json_dict)

    # Verify round-trip consistency
    assert response_restored.answer == answer
    assert response_restored.processing_time == processing_time
    assert len(response_restored.sources) == num_sources
    assert isinstance(response_restored.sources, list)
    assert isinstance(response_restored.modalities_used, list)


def test_property_21_missing_required_fields():
    """Property 21: Missing required fields should be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        TextChunk(chunk_id="test", doc_id="doc1")  # Missing content, page, chunk_index

    error_str = str(exc_info.value).lower()
    assert "content" in error_str or "field required" in error_str


def test_property_21_wrong_types():
    """Property 21: Wrong types should be rejected."""
    with pytest.raises(ValidationError):
        TextChunk(
            chunk_id="test",
            doc_id="doc1",
            content="test",
            page="not_an_int",  # Should be int
            chunk_index=0,
        )


def test_property_22_nested_validation():
    """Property 22: Nested structures should validate all levels."""
    # Create valid nested structure
    visual_chunk = VisualChunk(
        chunk_id="v1",
        doc_id="doc1",
        image_data=b"fake_image_data",
        page=0,
        chunk_index=0,
        visual_type="image",
    )

    multimodal_chunk = MultimodalChunk(
        chunk_id="m1",
        doc_id="doc1",
        text_content="test content",
        visual_elements=[visual_chunk],
        page=0,
        chunk_index=0,
    )

    # Serialize and deserialize
    json_dict = multimodal_chunk.model_dump()
    restored = MultimodalChunk.model_validate(json_dict)

    assert restored.text_content == "test content"
    assert len(restored.visual_elements) == 1
    assert restored.visual_elements[0].visual_type == "image"


def test_property_21_query_request_validation():
    """Property 21: QueryRequest validates all constraints."""
    # Valid request
    req = QueryRequest(question="test question", top_k=5, temperature=0.7)
    assert req.top_k == 5

    # top_k out of range
    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=0)  # Must be >= 1

    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=25)  # Must be <= 20

    # temperature out of range
    with pytest.raises(ValidationError):
        QueryRequest(question="test", temperature=-0.1)  # Must be >= 0

    with pytest.raises(ValidationError):
        QueryRequest(question="test", temperature=2.5)  # Must be <= 2.0


def test_property_22_parsed_document_serialization():
    """Property 22: ParsedDocument serializes with all modalities."""
    text_block = TextBlock(content="test", page=0, bbox=(0, 0, 100, 100))
    image_block = ImageBlock(image_data=b"image", page=0, bbox=(0, 0, 50, 50), format="png")
    chart_block = ChartBlock(image_data=b"chart", page=0, bbox=(0, 0, 75, 75), chart_type="bar")
    table_block = TableBlock(rows=[["a", "b"], ["c", "d"]], page=0, bbox=(0, 0, 80, 80))

    doc = ParsedDocument(
        text_blocks=[text_block],
        images=[image_block],
        charts=[chart_block],
        tables=[table_block],
        num_pages=1,
        metadata={"source": "test"},
    )

    # Serialize
    json_dict = doc.model_dump()

    # Deserialize
    restored = ParsedDocument.model_validate(json_dict)

    assert len(restored.text_blocks) == 1
    assert len(restored.images) == 1
    assert len(restored.charts) == 1
    assert len(restored.tables) == 1
    assert restored.num_pages == 1


def test_property_21_embedded_chunk_validation():
    """Property 21: EmbeddedChunk validates chunk and embedding."""
    text_chunk = TextChunk(
        chunk_id="t1",
        doc_id="doc1",
        content="test",
        page=0,
        chunk_index=0,
    )

    # Valid embedding
    embedded = EmbeddedChunk(chunk=text_chunk, embedding=[0.1, 0.2, 0.3], modality="text")
    assert len(embedded.embedding) == 3

    # Empty embedding should still be valid (edge case)
    embedded_empty = EmbeddedChunk(chunk=text_chunk, embedding=[], modality="text")
    assert len(embedded_empty.embedding) == 0
