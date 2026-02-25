"""Tests for Pydantic models."""

import json
from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.models import (
    DocumentChunk,
    DocumentInfo,
    DocumentMetadata,
    DocumentUploadResponse,
    EmbeddedChunk,
    ErrorResponse,
    ImageBlock,
    ParsedDocument,
    QueryRequest,
    QueryResponse,
    SourceReference,
    TableBlock,
    TextBlock,
)


class TestDocumentModels:
    """Test document-related models."""

    def test_document_metadata_valid(self):
        """Test DocumentMetadata with valid data."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            content_type="application/pdf",
            tags=["test", "sample"],
            description="Test document",
        )
        assert metadata.filename == "test.pdf"
        assert metadata.content_type == "application/pdf"
        assert len(metadata.tags) == 2

    def test_document_metadata_defaults(self):
        """Test DocumentMetadata with default values."""
        metadata = DocumentMetadata(filename="test.pdf")
        assert metadata.content_type == "application/pdf"
        assert metadata.tags == []
        assert metadata.description is None

    def test_document_upload_response(self):
        """Test DocumentUploadResponse."""
        now = datetime.now()
        response = DocumentUploadResponse(
            doc_id="doc123",
            filename="test.pdf",
            status="processing",
            message="Document uploaded successfully",
            created_at=now,
        )
        assert response.doc_id == "doc123"
        assert response.status == "processing"

    def test_document_info(self):
        """Test DocumentInfo."""
        now = datetime.now()
        info = DocumentInfo(
            doc_id="doc123",
            filename="test.pdf",
            file_size=1024,
            num_pages=10,
            num_chunks=50,
            created_at=now,
            tags=["test"],
        )
        assert info.num_pages == 10
        assert info.num_chunks == 50


class TestParsingModels:
    """Test parsing-related models."""

    def test_text_block(self):
        """Test TextBlock model."""
        block = TextBlock(
            content="Sample text", page=1, bbox=(0.0, 0.0, 100.0, 50.0), font_size=12.0
        )
        assert block.content == "Sample text"
        assert block.page == 1
        assert len(block.bbox) == 4

    def test_image_block(self):
        """Test ImageBlock model."""
        block = ImageBlock(
            image_data=b"fake_image_data", page=2, bbox=(10.0, 10.0, 200.0, 150.0), format="png"
        )
        assert block.format == "png"
        assert block.page == 2

    def test_table_block(self):
        """Test TableBlock model."""
        block = TableBlock(rows=[["A", "B"], ["1", "2"]], page=3, bbox=(0.0, 0.0, 300.0, 100.0))
        assert len(block.rows) == 2
        assert block.rows[0][0] == "A"

    def test_parsed_document(self):
        """Test ParsedDocument model."""
        doc = ParsedDocument(
            text_blocks=[TextBlock(content="Text", page=1, bbox=(0, 0, 100, 50))],
            images=[],
            tables=[],
            num_pages=5,
            metadata={"author": "Test Author"},
        )
        assert doc.num_pages == 5
        assert len(doc.text_blocks) == 1
        assert doc.metadata["author"] == "Test Author"


class TestChunkModels:
    """Test chunk-related models."""

    def test_document_chunk(self):
        """Test DocumentChunk model."""
        chunk = DocumentChunk(
            chunk_id="chunk123",
            doc_id="doc123",
            content="Chunk content",
            page=1,
            chunk_index=0,
            metadata={"section": "intro"},
        )
        assert chunk.chunk_id == "chunk123"
        assert chunk.doc_id == "doc123"
        assert chunk.chunk_index == 0

    def test_embedded_chunk(self):
        """Test EmbeddedChunk model."""
        chunk = DocumentChunk(
            chunk_id="chunk123",
            doc_id="doc123",
            content="Content",
            page=1,
            chunk_index=0,
            metadata={},
        )
        embedded = EmbeddedChunk(chunk=chunk, embedding=[0.1, 0.2, 0.3], modality="text")
        assert len(embedded.embedding) == 3
        assert embedded.chunk.chunk_id == "chunk123"


class TestQueryModels:
    """Test query-related models."""

    def test_query_request_defaults(self):
        """Test QueryRequest with default values."""
        request = QueryRequest(question="What is this about?")
        assert request.question == "What is this about?"
        assert request.top_k == 15  # Updated to match actual default
        assert request.temperature == 0.7
        assert request.include_sources is True

    def test_query_request_validation(self):
        """Test QueryRequest validation."""
        # Valid request
        request = QueryRequest(question="Test?", top_k=10, temperature=0.5)
        assert request.top_k == 10

        # Invalid top_k (too high)
        with pytest.raises(ValidationError):
            QueryRequest(question="Test?", top_k=25)

        # Invalid top_k (too low)
        with pytest.raises(ValidationError):
            QueryRequest(question="Test?", top_k=0)

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            QueryRequest(question="Test?", temperature=3.0)

    def test_source_reference(self):
        """Test SourceReference model."""
        source = SourceReference(
            doc_id="doc123",
            filename="test.pdf",
            page=5,
            chunk_content="Relevant content",
            modality="text",
            relevance_score=0.95,
        )
        assert source.doc_id == "doc123"
        assert source.relevance_score == 0.95

    def test_query_response(self):
        """Test QueryResponse model."""
        source = SourceReference(
            doc_id="doc123",
            filename="test.pdf",
            page=1,
            chunk_content="Content",
            modality="text",
            relevance_score=0.9,
        )
        response = QueryResponse(answer="The answer is...", sources=[source], processing_time=0.5)
        assert response.answer == "The answer is..."
        assert len(response.sources) == 1
        assert response.processing_time == 0.5


class TestErrorModels:
    """Test error-related models."""

    def test_error_response(self):
        """Test ErrorResponse model."""
        now = datetime.now()
        error = ErrorResponse(
            error="ValidationError",
            detail="Invalid file format",
            timestamp=now,
            request_id="req123",
        )
        assert error.error == "ValidationError"
        assert error.detail == "Invalid file format"
        assert error.request_id == "req123"


class TestPydanticValidationProperties:
    """Property-based tests for Pydantic validation enforcement."""

    @given(
        top_k=st.integers().filter(lambda x: x < 1 or x > 20),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_query_request_top_k_validation(self, top_k):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with invalid data (missing required fields, wrong types,
        out-of-range values), the system should reject the request and return a 422
        status code with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that top_k values outside the valid range [1, 20] are rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="Test question", top_k=top_k)

        # Verify error contains field name
        error_str = str(exc_info.value)
        assert "top_k" in error_str.lower()

    @given(
        temperature=st.floats().filter(lambda x: x < 0.0 or x > 2.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_query_request_temperature_validation(self, temperature):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with invalid data (out-of-range temperature values),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that temperature values outside [0.0, 2.0] are rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="Test question", temperature=temperature)

        error_str = str(exc_info.value)
        assert "temperature" in error_str.lower()

    @given(
        page=st.integers().filter(lambda x: x < 0),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_text_block_negative_page_validation(self, page):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with invalid data (negative page numbers),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that negative page numbers are rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            TextBlock(content="Test content", page=page, bbox=(0.0, 0.0, 100.0, 50.0))

        error_str = str(exc_info.value)
        assert "page" in error_str.lower()

    @given(
        file_size=st.integers().filter(lambda x: x < 0),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_document_info_negative_file_size_validation(self, file_size):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with invalid data (negative file sizes),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that negative file sizes are rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            DocumentInfo(
                doc_id="doc123",
                filename="test.pdf",
                file_size=file_size,
                num_pages=10,
                num_chunks=50,
                created_at=datetime.now(),
                tags=[],
            )

        error_str = str(exc_info.value)
        assert "file_size" in error_str.lower()

    @given(
        num_pages=st.integers().filter(lambda x: x < 0),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_document_info_negative_num_pages_validation(self, num_pages):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with invalid data (negative page counts),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that negative page counts are rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            DocumentInfo(
                doc_id="doc123",
                filename="test.pdf",
                file_size=1024,
                num_pages=num_pages,
                num_chunks=50,
                created_at=datetime.now(),
                tags=[],
            )

        error_str = str(exc_info.value)
        assert "num_pages" in error_str.lower()

    @given(
        relevance_score=st.floats().filter(lambda x: x < 0.0 or x > 1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_source_reference_relevance_score_validation(self, relevance_score):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with invalid data (relevance scores outside [0, 1]),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that relevance scores outside [0.0, 1.0] are rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            SourceReference(
                doc_id="doc123",
                filename="test.pdf",
                page=1,
                chunk_content="Content",
                modality="text",
                relevance_score=relevance_score,
            )

        error_str = str(exc_info.value)
        assert "relevance_score" in error_str.lower()

    @given(
        wrong_type=st.integers(),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_type_validation_question_field(self, wrong_type):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with wrong types (integer instead of string for question),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that type mismatches are caught and reported.
        """
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question=wrong_type)

        error_str = str(exc_info.value)
        assert "question" in error_str.lower()

    @given(
        wrong_type=st.text(),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_type_validation_embedding_field(self, wrong_type):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with wrong types (string instead of list for embedding),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that type mismatches for complex types are caught.
        """
        chunk = DocumentChunk(
            chunk_id="chunk123",
            doc_id="doc123",
            content="Content",
            page=1,
            chunk_index=0,
            metadata={},
        )

        with pytest.raises(ValidationError) as exc_info:
            EmbeddedChunk(chunk=chunk, embedding=wrong_type)

        error_str = str(exc_info.value)
        assert "embedding" in error_str.lower()

    def test_property_21_missing_required_fields(self):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with missing required fields, the system should reject
        the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that missing required fields are caught.
        """
        # Missing 'question' field
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest()

        error_str = str(exc_info.value)
        assert "question" in error_str.lower()
        assert "required" in error_str.lower() or "missing" in error_str.lower()

    @given(
        invalid_bbox=st.lists(
            st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=10
        ).filter(lambda x: len(x) != 4)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_21_nested_validation_bbox(self, invalid_bbox):
        """Feature: pdf-rag-system, Property 21: Pydantic validation enforcement.

        For any API request with invalid nested data structures (bbox with wrong length),
        the system should reject the request with detailed field-level error messages.

        Validates: Requirements 6.1, 6.2, 6.5

        This test verifies that nested structure validation works correctly.
        """
        with pytest.raises(ValidationError) as exc_info:
            TextBlock(content="Test", page=1, bbox=tuple(invalid_bbox))

        error_str = str(exc_info.value)
        assert "bbox" in error_str.lower()


class TestResponseSerializationProperties:
    """Property-based tests for response serialization consistency."""

    @given(
        doc_id=st.text(min_size=1, max_size=100),
        filename=st.text(min_size=1, max_size=255),
        status=st.text(min_size=1, max_size=50),
        message=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_22_document_upload_response_serialization(
        self, doc_id, filename, status, message
    ):
        """Feature: pdf-rag-system, Property 22: Response serialization consistency.

        For any API response, the returned JSON should validate against the
        corresponding Pydantic response model schema, ensuring all required
        fields are present and types are correct.

        Validates: Requirements 6.3

        This test verifies that DocumentUploadResponse can be serialized to JSON
        and deserialized back to the same model.
        """
        now = datetime.now()
        response = DocumentUploadResponse(
            doc_id=doc_id, filename=filename, status=status, message=message, created_at=now
        )

        # Serialize to JSON
        json_str = response.model_dump_json()

        # Verify it's valid JSON
        json_data = json.loads(json_str)

        # Verify all required fields are present
        assert "doc_id" in json_data
        assert "filename" in json_data
        assert "status" in json_data
        assert "message" in json_data
        assert "created_at" in json_data

        # Deserialize back to model
        deserialized = DocumentUploadResponse.model_validate_json(json_str)

        # Verify values match
        assert deserialized.doc_id == doc_id
        assert deserialized.filename == filename
        assert deserialized.status == status
        assert deserialized.message == message

    @given(
        doc_id=st.text(min_size=1, max_size=100),
        filename=st.text(min_size=1, max_size=255),
        file_size=st.integers(min_value=0, max_value=1000000000),
        num_pages=st.integers(min_value=0, max_value=10000),
        num_chunks=st.integers(min_value=0, max_value=100000),
        tags=st.lists(st.text(min_size=1, max_size=50), max_size=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_22_document_info_serialization(
        self, doc_id, filename, file_size, num_pages, num_chunks, tags
    ):
        """Feature: pdf-rag-system, Property 22: Response serialization consistency.

        For any API response, the returned JSON should validate against the
        corresponding Pydantic response model schema.

        Validates: Requirements 6.3

        This test verifies DocumentInfo serialization consistency.
        """
        now = datetime.now()
        info = DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            file_size=file_size,
            num_pages=num_pages,
            num_chunks=num_chunks,
            created_at=now,
            tags=tags,
        )

        # Serialize to JSON
        json_str = info.model_dump_json()
        json_data = json.loads(json_str)

        # Verify all required fields are present
        assert "doc_id" in json_data
        assert "filename" in json_data
        assert "file_size" in json_data
        assert "num_pages" in json_data
        assert "num_chunks" in json_data
        assert "created_at" in json_data
        assert "tags" in json_data

        # Deserialize and verify
        deserialized = DocumentInfo.model_validate_json(json_str)
        assert deserialized.doc_id == doc_id
        assert deserialized.file_size == file_size
        assert deserialized.num_pages == num_pages
        assert deserialized.num_chunks == num_chunks
        assert deserialized.tags == tags

    @given(
        answer=st.text(min_size=1, max_size=5000),
        processing_time=st.floats(
            min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False
        ),
        num_sources=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_22_query_response_serialization(self, answer, processing_time, num_sources):
        """Feature: pdf-rag-system, Property 22: Response serialization consistency.

        For any API response, the returned JSON should validate against the
        corresponding Pydantic response model schema.

        Validates: Requirements 6.3

        This test verifies QueryResponse serialization with nested SourceReference objects.
        """
        sources = [
            SourceReference(
                doc_id=f"doc{i}",
                filename=f"file{i}.pdf",
                page=i + 1,
                chunk_content=f"Content {i}",
                modality="text",
                relevance_score=max(0.0, 0.95 - (i * 0.04)),  # Ensure score stays >= 0
            )
            for i in range(num_sources)
        ]

        response = QueryResponse(answer=answer, sources=sources, processing_time=processing_time)

        # Serialize to JSON
        json_str = response.model_dump_json()
        json_data = json.loads(json_str)

        # Verify all required fields are present
        assert "answer" in json_data
        assert "sources" in json_data
        assert "processing_time" in json_data

        # Verify sources is a list
        assert isinstance(json_data["sources"], list)
        assert len(json_data["sources"]) == num_sources

        # Verify each source has required fields
        for source_data in json_data["sources"]:
            assert "doc_id" in source_data
            assert "filename" in source_data
            assert "page" in source_data
            assert "chunk_content" in source_data
            assert "relevance_score" in source_data

        # Deserialize and verify
        deserialized = QueryResponse.model_validate_json(json_str)
        assert deserialized.answer == answer
        assert len(deserialized.sources) == num_sources
        assert deserialized.processing_time == processing_time

    @given(
        error=st.text(min_size=1, max_size=100),
        detail=st.text(min_size=1, max_size=1000),
        request_id=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_22_error_response_serialization(self, error, detail, request_id):
        """Feature: pdf-rag-system, Property 22: Response serialization consistency.

        For any API response, the returned JSON should validate against the
        corresponding Pydantic response model schema.

        Validates: Requirements 6.3

        This test verifies ErrorResponse serialization consistency.
        """
        now = datetime.now()
        error_response = ErrorResponse(
            error=error, detail=detail, timestamp=now, request_id=request_id
        )

        # Serialize to JSON
        json_str = error_response.model_dump_json()
        json_data = json.loads(json_str)

        # Verify all required fields are present
        assert "error" in json_data
        assert "detail" in json_data
        assert "timestamp" in json_data
        assert "request_id" in json_data

        # Deserialize and verify
        deserialized = ErrorResponse.model_validate_json(json_str)
        assert deserialized.error == error
        assert deserialized.detail == detail
        assert deserialized.request_id == request_id

    @given(
        content=st.text(min_size=1, max_size=10000),
        page=st.integers(min_value=0, max_value=10000),
        bbox_values=st.lists(
            st.floats(min_value=-1000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_22_text_block_serialization(self, content, page, bbox_values):
        """Feature: pdf-rag-system, Property 22: Response serialization consistency.

        For any API response, the returned JSON should validate against the
        corresponding Pydantic response model schema.

        Validates: Requirements 6.3

        This test verifies TextBlock serialization with tuple fields.
        """
        bbox = tuple(bbox_values)
        text_block = TextBlock(content=content, page=page, bbox=bbox, font_size=12.0)

        # Serialize to JSON
        json_str = text_block.model_dump_json()
        json_data = json.loads(json_str)

        # Verify all required fields are present
        assert "content" in json_data
        assert "page" in json_data
        assert "bbox" in json_data

        # Verify bbox is serialized as a list (JSON doesn't have tuples)
        assert isinstance(json_data["bbox"], list)
        assert len(json_data["bbox"]) == 4

        # Deserialize and verify
        deserialized = TextBlock.model_validate_json(json_str)
        assert deserialized.content == content
        assert deserialized.page == page
        assert len(deserialized.bbox) == 4

    @given(
        text_blocks=st.lists(
            st.builds(
                TextBlock,
                content=st.text(min_size=1, max_size=1000),
                page=st.integers(min_value=0, max_value=100),
                bbox=st.tuples(
                    st.floats(
                        min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
                    ),
                    st.floats(
                        min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
                    ),
                    st.floats(
                        min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
                    ),
                    st.floats(
                        min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
                    ),
                ),
            ),
            max_size=10,
        ),
        num_pages=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_22_parsed_document_nested_serialization(self, text_blocks, num_pages):
        """Feature: pdf-rag-system, Property 22: Response serialization consistency.

        For any API response with nested data structures, the returned JSON
        should validate against the corresponding Pydantic response model schema.

        Validates: Requirements 6.3, 6.5

        This test verifies ParsedDocument serialization with nested TextBlock objects.
        """
        parsed_doc = ParsedDocument(
            text_blocks=text_blocks,
            images=[],
            tables=[],
            num_pages=num_pages,
            metadata={"test": "metadata"},
        )

        # Serialize to JSON
        json_str = parsed_doc.model_dump_json()
        json_data = json.loads(json_str)

        # Verify all required fields are present
        assert "text_blocks" in json_data
        assert "images" in json_data
        assert "tables" in json_data
        assert "num_pages" in json_data
        assert "metadata" in json_data

        # Verify nested structure
        assert isinstance(json_data["text_blocks"], list)
        assert len(json_data["text_blocks"]) == len(text_blocks)

        # Deserialize and verify
        deserialized = ParsedDocument.model_validate_json(json_str)
        assert len(deserialized.text_blocks) == len(text_blocks)
        assert deserialized.num_pages == num_pages
        assert deserialized.metadata == {"test": "metadata"}
