"""Property-based tests for document service."""

import asyncio
import logging
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck, assume

from app.services.document_service import (
    DocumentService,
    DocumentProcessingError,
    ProcessingStatus,
)
from app.models.document import DocumentMetadata, DocumentUploadResponse
from app.models.parsing import ParsedDocument, TextBlock
from app.models.chunk import DocumentChunk, EmbeddedChunk
from app.storage.file_storage import FileMetadata


# Custom strategies for generating test data
@st.composite
def document_metadata_strategy(draw):
    """Generate valid document metadata."""
    filename = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('L', 'N'), min_codepoint=97, max_codepoint=122
    ))) + ".pdf"
    tags = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        max_size=5
    ))
    description = draw(st.one_of(
        st.none(),
        st.text(max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
    ))
    
    return DocumentMetadata(
        filename=filename,
        content_type="application/pdf",
        tags=tags,
        description=description
    )


@pytest.fixture
def mock_dependencies():
    """Create mocked dependencies for document service."""
    # Mock file storage
    file_storage = MagicMock()
    file_storage.save_file = AsyncMock()
    file_storage.delete_file = MagicMock(return_value=True)
    
    # Mock parser
    parser = MagicMock()
    parser.parse_pdf = MagicMock()
    
    # Mock chunker
    chunker = MagicMock()
    chunker.chunk_document = MagicMock()
    
    # Mock OpenAI client
    openai_client = MagicMock()
    openai_client.embed_batch = AsyncMock()
    openai_client.close = AsyncMock()
    
    # Mock vector store
    vector_store = MagicMock()
    vector_store.add_embeddings = AsyncMock()
    vector_store.delete_document = AsyncMock()
    
    return {
        "file_storage": file_storage,
        "parser": parser,
        "chunker": chunker,
        "openai_client": openai_client,
        "vector_store": vector_store,
    }


@pytest.fixture
def document_service(mock_dependencies, monkeypatch):
    """Create document service with mocked dependencies."""
    from app.config import Settings
    
    # Mock settings
    mock_settings = MagicMock(spec=Settings)
    mock_settings.chunk_size = 512
    mock_settings.chunk_overlap = 50
    mock_settings.openai_api_key = "sk-test-key"
    mock_settings.openai_model = "gpt-4o-mini"
    mock_settings.openai_embedding_model = "text-embedding-3-small"
    mock_settings.vector_db_path = "./data/vectordb"
    mock_settings.collection_name = "documents"
    
    monkeypatch.setattr("app.services.document_service.get_settings", lambda: mock_settings)
    
    return DocumentService(
        file_storage=mock_dependencies["file_storage"],
        parser=mock_dependencies["parser"],
        chunker=mock_dependencies["chunker"],
        openai_client=mock_dependencies["openai_client"],
        vector_store=mock_dependencies["vector_store"],
    )


class TestUniqueDocumentIdentifiers:
    """Property-based tests for Property 1: Unique document identifiers."""

    @pytest.mark.asyncio
    @given(
        num_documents=st.integers(min_value=2, max_value=20)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    async def test_property_1_unique_document_identifiers(
        self, document_service, mock_dependencies, num_documents
    ):
        """Feature: pdf-rag-system, Property 1: Unique document identifiers.
        
        For any valid PDF file uploaded to the system, the returned document 
        identifier should be unique and different from all previously generated 
        identifiers.
        
        Validates: Requirements 1.1
        """
        # Setup mocks to return unique IDs
        doc_ids = set()
        
        for i in range(num_documents):
            # Create unique file metadata for each upload
            file_metadata = FileMetadata(
                file_id=f"doc-{i}",
                filename=f"test{i}.pdf",
                file_size=1024,
                content_type="application/pdf",
                file_hash=f"hash{i}",
                upload_path=Path(f"/tmp/test{i}.pdf"),
                created_at=datetime.now(UTC),
            )
            mock_dependencies["file_storage"].save_file.return_value = file_metadata
            
            # Setup parser to return valid parsed document
            parsed_doc = ParsedDocument(
                text_blocks=[TextBlock(content=f"Content {i}", page=0, bbox=(0, 0, 100, 100))],
                images=[],
                tables=[],
                num_pages=1,
                metadata={},
            )
            mock_dependencies["parser"].parse_pdf.return_value = parsed_doc
            
            # Setup chunker
            chunks = [DocumentChunk(
                chunk_id=f"chunk-{i}",
                doc_id=f"doc-{i}",
                content=f"Content {i}",
                page=0,
                chunk_index=0,
                metadata={},
            )]
            mock_dependencies["chunker"].chunk_document.return_value = chunks
            
            # Setup embeddings
            mock_dependencies["openai_client"].embed_batch.return_value = [[0.1] * 384]
            mock_dependencies["vector_store"].add_embeddings.return_value = True
            
            # Process document
            metadata = DocumentMetadata(filename=f"test{i}.pdf")
            response = await document_service.process_document(
                file_content=f"pdf content {i}".encode(),
                metadata=metadata,
            )
            
            # Property: Document ID should be unique
            assert response.doc_id not in doc_ids, \
                f"Duplicate document ID generated: {response.doc_id}"
            
            doc_ids.add(response.doc_id)
        
        # Property: All document IDs should be unique
        assert len(doc_ids) == num_documents, \
            f"Expected {num_documents} unique IDs, got {len(doc_ids)}"


class TestUploadMetadataPersistence:
    """Property-based tests for Property 3: Upload metadata persistence round-trip."""

    @pytest.mark.asyncio
    @given(
        metadata=document_metadata_strategy()
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    async def test_property_3_upload_metadata_persistence_round_trip(
        self, document_service, mock_dependencies, metadata
    ):
        """Feature: pdf-rag-system, Property 3: Upload metadata persistence round-trip.
        
        For any successfully uploaded PDF document, querying the storage immediately 
        after upload should return metadata that matches the uploaded document's 
        properties (filename, size, upload time).
        
        Validates: Requirements 1.4
        """
        # Setup mocks
        file_size = 2048
        created_at = datetime.now(UTC)
        
        file_metadata = FileMetadata(
            file_id="test-doc-id",
            filename=metadata.filename,
            file_size=file_size,
            content_type=metadata.content_type,
            file_hash="abc123",
            upload_path=Path(f"/tmp/{metadata.filename}"),
            created_at=created_at,
        )
        mock_dependencies["file_storage"].save_file.return_value = file_metadata
        
        # Setup parser
        parsed_doc = ParsedDocument(
            text_blocks=[TextBlock(content="Test content", page=0, bbox=(0, 0, 100, 100))],
            images=[],
            tables=[],
            num_pages=1,
            metadata={},
        )
        mock_dependencies["parser"].parse_pdf.return_value = parsed_doc
        
        # Setup chunker
        chunks = [DocumentChunk(
            chunk_id="chunk-1",
            doc_id="test-doc-id",
            content="Test content",
            page=0,
            chunk_index=0,
            metadata={},
        )]
        mock_dependencies["chunker"].chunk_document.return_value = chunks
        
        # Setup embeddings
        mock_dependencies["openai_client"].embed_batch.return_value = [[0.1] * 384]
        mock_dependencies["vector_store"].add_embeddings.return_value = True
        
        # Process document
        response = await document_service.process_document(
            file_content=b"fake pdf content",
            metadata=metadata,
        )
        
        # Property: Response should contain correct metadata
        assert response.filename == metadata.filename, \
            f"Filename mismatch: expected {metadata.filename}, got {response.filename}"
        
        # Query the document immediately after upload
        doc_info = await document_service.get_document(response.doc_id)
        
        # Property: Retrieved metadata should match uploaded metadata
        assert doc_info.filename == metadata.filename, \
            f"Retrieved filename mismatch: expected {metadata.filename}, got {doc_info.filename}"
        
        assert doc_info.file_size == file_size, \
            f"File size mismatch: expected {file_size}, got {doc_info.file_size}"
        
        assert doc_info.tags == metadata.tags, \
            f"Tags mismatch: expected {metadata.tags}, got {doc_info.tags}"
        
        # Property: Created timestamp should be preserved
        assert doc_info.created_at == created_at, \
            f"Created timestamp mismatch: expected {created_at}, got {doc_info.created_at}"


class TestConcurrentUploadIndependence:
    """Property-based tests for Property 4: Concurrent upload independence."""

    @pytest.mark.asyncio
    @given(
        num_concurrent=st.integers(min_value=2, max_value=10)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    async def test_property_4_concurrent_upload_independence(
        self, document_service, mock_dependencies, num_concurrent
    ):
        """Feature: pdf-rag-system, Property 4: Concurrent upload independence.
        
        For any set of PDF documents uploaded concurrently, each upload should 
        complete successfully with a unique document ID, and the total number of 
        stored documents should equal the number of uploads.
        
        Validates: Requirements 1.5
        """
        # Reset document service state for this test
        document_service._documents.clear()
        document_service._status.clear()
        
        # Track call counts for mocks
        call_count = {"save_file": 0, "parse_pdf": 0, "chunk_document": 0}
        
        def save_file_side_effect(*args, **kwargs):
            idx = call_count["save_file"]
            call_count["save_file"] += 1
            return FileMetadata(
                file_id=f"doc-{idx}",
                filename=f"test{idx}.pdf",
                file_size=1024,
                content_type="application/pdf",
                file_hash=f"hash{idx}",
                upload_path=Path(f"/tmp/test{idx}.pdf"),
                created_at=datetime.now(UTC),
            )
        
        def parse_pdf_side_effect(*args, **kwargs):
            idx = call_count["parse_pdf"]
            call_count["parse_pdf"] += 1
            return ParsedDocument(
                text_blocks=[TextBlock(content=f"Content {idx}", page=0, bbox=(0, 0, 100, 100))],
                images=[],
                tables=[],
                num_pages=1,
                metadata={},
            )
        
        def chunk_document_side_effect(*args, **kwargs):
            idx = call_count["chunk_document"]
            call_count["chunk_document"] += 1
            doc_id = kwargs.get("doc_id", f"doc-{idx}")
            return [DocumentChunk(
                chunk_id=f"chunk-{idx}",
                doc_id=doc_id,
                content=f"Content {idx}",
                page=0,
                chunk_index=0,
                metadata={},
            )]
        
        mock_dependencies["file_storage"].save_file.side_effect = save_file_side_effect
        mock_dependencies["parser"].parse_pdf.side_effect = parse_pdf_side_effect
        mock_dependencies["chunker"].chunk_document.side_effect = chunk_document_side_effect
        mock_dependencies["openai_client"].embed_batch.return_value = [[0.1] * 384]
        mock_dependencies["vector_store"].add_embeddings.return_value = True
        
        # Create concurrent upload tasks
        async def upload_document(idx):
            metadata = DocumentMetadata(filename=f"test{idx}.pdf")
            return await document_service.process_document(
                file_content=f"pdf content {idx}".encode(),
                metadata=metadata,
            )
        
        # Execute uploads concurrently
        tasks = [upload_document(i) for i in range(num_concurrent)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Property: All uploads should succeed (no exceptions)
        successful_responses = [r for r in responses if isinstance(r, DocumentUploadResponse)]
        assert len(successful_responses) == num_concurrent, \
            f"Expected {num_concurrent} successful uploads, got {len(successful_responses)}"
        
        # Property: All document IDs should be unique
        doc_ids = [r.doc_id for r in successful_responses]
        unique_ids = set(doc_ids)
        assert len(unique_ids) == num_concurrent, \
            f"Expected {num_concurrent} unique IDs, got {len(unique_ids)} (duplicates detected)"
        
        # Property: Total number of stored documents should equal number of uploads
        all_docs = await document_service.list_documents()
        assert len(all_docs) == num_concurrent, \
            f"Expected {num_concurrent} stored documents, got {len(all_docs)}"


class TestProcessingStatusAccuracy:
    """Property-based tests for Property 25: Processing status accuracy."""

    @pytest.mark.asyncio
    @given(
        stage_to_fail=st.sampled_from([
            ProcessingStatus.PARSING,
            ProcessingStatus.CHUNKING,
            ProcessingStatus.EMBEDDING,
            ProcessingStatus.STORING,
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    async def test_property_25_processing_status_accuracy(
        self, document_service, mock_dependencies, stage_to_fail
    ):
        """Feature: pdf-rag-system, Property 25: Processing status accuracy.
        
        For any document being processed, querying the status endpoint should return 
        a state that accurately reflects the current processing stage (uploaded, 
        parsing, embedding, completed, failed).
        
        Validates: Requirements 9.4
        """
        # Setup mocks
        file_metadata = FileMetadata(
            file_id="test-doc-id",
            filename="test.pdf",
            file_size=1024,
            content_type="application/pdf",
            file_hash="abc123",
            upload_path=Path("/tmp/test.pdf"),
            created_at=datetime.now(UTC),
        )
        mock_dependencies["file_storage"].save_file.return_value = file_metadata
        
        # Configure mocks to fail at specific stage
        if stage_to_fail == ProcessingStatus.PARSING:
            mock_dependencies["parser"].parse_pdf.side_effect = Exception("Parsing failed")
        else:
            parsed_doc = ParsedDocument(
                text_blocks=[TextBlock(content="Test", page=0, bbox=(0, 0, 100, 100))],
                images=[],
                tables=[],
                num_pages=1,
                metadata={},
            )
            mock_dependencies["parser"].parse_pdf.return_value = parsed_doc
        
        if stage_to_fail == ProcessingStatus.CHUNKING:
            mock_dependencies["chunker"].chunk_document.side_effect = Exception("Chunking failed")
        else:
            chunks = [DocumentChunk(
                chunk_id="chunk-1",
                doc_id="test-doc-id",
                content="Test",
                page=0,
                chunk_index=0,
                metadata={},
            )]
            mock_dependencies["chunker"].chunk_document.return_value = chunks
        
        if stage_to_fail == ProcessingStatus.EMBEDDING:
            mock_dependencies["openai_client"].embed_batch.side_effect = Exception("Embedding failed")
        else:
            mock_dependencies["openai_client"].embed_batch.return_value = [[0.1] * 384]
        
        if stage_to_fail == ProcessingStatus.STORING:
            mock_dependencies["vector_store"].add_embeddings.side_effect = Exception("Storing failed")
        else:
            mock_dependencies["vector_store"].add_embeddings.return_value = True
        
        # Process document (should fail at specified stage)
        metadata = DocumentMetadata(filename="test.pdf")
        
        try:
            await document_service.process_document(
                file_content=b"fake pdf content",
                metadata=metadata,
            )
            # If we get here, processing succeeded (no failure injected)
            status = document_service.get_processing_status("test-doc-id")
            assert status == ProcessingStatus.COMPLETED
        except DocumentProcessingError:
            # Processing failed as expected
            status = document_service.get_processing_status("test-doc-id")
            
            # Property: Status should be FAILED when processing fails
            assert status == ProcessingStatus.FAILED, \
                f"Expected status FAILED after error at {stage_to_fail}, got {status}"


class TestErrorLoggingCompleteness:
    """Property-based tests for Property 26: Error logging completeness."""

    @pytest.mark.asyncio
    @given(
        error_message=st.text(min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    async def test_property_26_error_logging_completeness(
        self, document_service, mock_dependencies, error_message
    ):
        """Feature: pdf-rag-system, Property 26: Error logging completeness.
        
        For any error that occurs during processing, the system should create a log 
        entry containing timestamp, error type, error message, and contextual 
        information (doc_id, operation stage).
        
        Validates: Requirements 10.1, 10.4
        """
        # Setup mocks
        file_metadata = FileMetadata(
            file_id="test-doc-id",
            filename="test.pdf",
            file_size=1024,
            content_type="application/pdf",
            file_hash="abc123",
            upload_path=Path("/tmp/test.pdf"),
            created_at=datetime.now(UTC),
        )
        mock_dependencies["file_storage"].save_file.return_value = file_metadata
        
        # Make parser fail with custom error message
        mock_dependencies["parser"].parse_pdf.side_effect = Exception(error_message)
        
        # Process document (should fail)
        metadata = DocumentMetadata(filename="test.pdf")
        
        try:
            await document_service.process_document(
                file_content=b"fake pdf content",
                metadata=metadata,
            )
        except DocumentProcessingError:
            pass  # Expected
        
        # Property: Error should be logged in document data
        doc_data = document_service._documents.get("test-doc-id")
        assert doc_data is not None, "Document data not found after error"
        
        assert "errors" in doc_data, "No errors field in document data"
        assert len(doc_data["errors"]) > 0, "No errors logged"
        
        error_entry = doc_data["errors"][0]
        
        # Property: Error entry should contain timestamp
        assert "timestamp" in error_entry, "Error entry missing timestamp"
        assert isinstance(error_entry["timestamp"], datetime), \
            f"Timestamp is not datetime: {type(error_entry['timestamp'])}"
        
        # Property: Error entry should contain stage information
        assert "stage" in error_entry, "Error entry missing stage"
        assert error_entry["stage"] in [s.value for s in ProcessingStatus], \
            f"Invalid stage: {error_entry['stage']}"
        
        # Property: Error entry should contain error message
        assert "error" in error_entry, "Error entry missing error message"
        assert error_message in error_entry["error"], \
            f"Error message not found in log: expected '{error_message}' in '{error_entry['error']}'"
