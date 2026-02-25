"""Tests for document processing service."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.chunk import DocumentChunk
from app.models.document import DocumentMetadata
from app.models.parsing import ParsedDocument, TextBlock
from app.services.document_service import (
    DocumentProcessingError,
    DocumentService,
    ProcessingStatus,
)
from app.storage.file_storage import FileValidationError


@pytest.fixture
def mock_file_storage():
    """Mock file storage service."""
    storage = MagicMock()
    storage.save_file = AsyncMock()
    storage.delete_file = MagicMock(return_value=True)
    return storage


@pytest.fixture
def mock_parser():
    """Mock PDF parser."""
    parser = MagicMock()
    parser.parse_pdf = MagicMock()
    return parser


@pytest.fixture
def mock_chunker():
    """Mock text chunker."""
    chunker = MagicMock()
    chunker.chunk_document = MagicMock()
    return chunker


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = MagicMock()
    client.embed_batch = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = MagicMock()
    store.add_embeddings = AsyncMock()
    store.delete_document = AsyncMock()
    return store


@pytest.fixture
def document_service(
    mock_file_storage,
    mock_parser,
    mock_chunker,
    mock_openai_client,
    mock_vector_store,
    monkeypatch,
):
    """Create document service with mocked dependencies."""
    # Mock the settings to avoid requiring environment variables
    from app.config import Settings

    mock_settings = MagicMock(spec=Settings)
    mock_settings.chunk_size = 512
    mock_settings.chunk_overlap = 50
    mock_settings.openai_api_key = MagicMock()
    mock_settings.openai_api_key.get_secret_value.return_value = (
        "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
    )
    mock_settings.openai_model = "gpt-4o-mini"
    mock_settings.openai_embedding_model = "text-embedding-3-small"
    mock_settings.vector_db_path = "./data/vectordb"
    mock_settings.text_collection = "text_chunks"
    mock_settings.collection_name = "documents"
    mock_settings.enable_hybrid_search = True
    mock_settings.bm25_k1 = 1.5
    mock_settings.bm25_b = 0.75
    mock_settings.reranking_top_k = 40

    # Patch get_settings to return our mock
    monkeypatch.setattr("app.services.document_service.get_settings", lambda: mock_settings)

    return DocumentService(
        file_storage=mock_file_storage,
        parser=mock_parser,
        chunker=mock_chunker,
        openai_client=mock_openai_client,
        vector_store=mock_vector_store,
    )


@pytest.mark.asyncio
async def test_process_document_success(
    document_service,
    mock_file_storage,
    mock_parser,
    mock_chunker,
    mock_openai_client,
    mock_vector_store,
):
    """Test successful document processing through full pipeline."""
    # Setup mocks
    from datetime import UTC, datetime

    from app.storage.file_storage import FileMetadata

    file_metadata = FileMetadata(
        file_id="test-doc-id",
        filename="test.pdf",
        file_size=1024,
        content_type="application/pdf",
        file_hash="abc123",
        upload_path=Path("/tmp/test.pdf"),
        created_at=datetime.now(UTC),
    )
    mock_file_storage.save_file.return_value = file_metadata

    parsed_doc = ParsedDocument(
        text_blocks=[TextBlock(content="Test content", page=0, bbox=(0, 0, 100, 100))],
        images=[],
        tables=[],
        num_pages=1,
        metadata={},
    )
    mock_parser.parse_pdf.return_value = parsed_doc

    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            doc_id="test-doc-id",
            content="Test content",
            page=0,
            chunk_index=0,
            metadata={},
        )
    ]
    mock_chunker.chunk_document.return_value = chunks
    mock_chunker.chunk_with_structure.return_value = chunks

    embeddings = [[0.1, 0.2, 0.3]]
    mock_openai_client.embed_batch.return_value = embeddings

    mock_vector_store.add_embeddings.return_value = True

    # Process document
    metadata = DocumentMetadata(filename="test.pdf")
    response = await document_service.process_document(
        file_content=b"fake pdf content",
        metadata=metadata,
    )

    # Verify response
    assert response.doc_id == "test-doc-id"
    assert response.filename == "test.pdf"
    assert response.status == ProcessingStatus.COMPLETED.value

    # Verify all steps were called
    mock_file_storage.save_file.assert_called_once()
    mock_parser.parse_pdf.assert_called_once()
    mock_chunker.chunk_with_structure.assert_called_once()
    mock_openai_client.embed_batch.assert_called_once()
    mock_vector_store.add_embeddings.assert_called_once()

    # Verify status tracking
    status = document_service.get_processing_status("test-doc-id")
    assert status == ProcessingStatus.COMPLETED


@pytest.mark.asyncio
async def test_process_document_validation_error(
    document_service,
    mock_file_storage,
):
    """Test document processing with file validation error."""
    # Setup mock to raise validation error
    mock_file_storage.save_file.side_effect = FileValidationError("Invalid file format")

    # Process document should raise validation error
    metadata = DocumentMetadata(filename="test.txt")
    with pytest.raises(FileValidationError):
        await document_service.process_document(
            file_content=b"not a pdf",
            metadata=metadata,
        )


@pytest.mark.asyncio
async def test_process_document_parsing_error(
    document_service,
    mock_file_storage,
    mock_parser,
):
    """Test document processing with parsing error."""
    from datetime import UTC, datetime

    from app.storage.file_storage import FileMetadata

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
    mock_file_storage.save_file.return_value = file_metadata

    # Parser raises error
    mock_parser.parse_pdf.side_effect = Exception("Parsing failed")

    # Process document should raise processing error
    metadata = DocumentMetadata(filename="test.pdf")
    with pytest.raises(DocumentProcessingError):
        await document_service.process_document(
            file_content=b"fake pdf content",
            metadata=metadata,
        )

    # Verify status is failed
    status = document_service.get_processing_status("test-doc-id")
    assert status == ProcessingStatus.FAILED


@pytest.mark.asyncio
async def test_get_document(document_service):
    """Test retrieving document information."""
    from datetime import UTC, datetime

    from app.storage.file_storage import FileMetadata

    # Add a document to the service
    file_metadata = FileMetadata(
        file_id="test-doc-id",
        filename="test.pdf",
        file_size=1024,
        content_type="application/pdf",
        file_hash="abc123",
        upload_path=Path("/tmp/test.pdf"),
        created_at=datetime.now(UTC),
    )

    document_service._documents["test-doc-id"] = {
        "file_metadata": file_metadata,
        "document_metadata": DocumentMetadata(filename="test.pdf", tags=["test"]),
        "created_at": file_metadata.created_at,
        "num_pages": 5,
        "num_chunks": 10,
    }

    # Get document
    doc_info = await document_service.get_document("test-doc-id")

    assert doc_info.doc_id == "test-doc-id"
    assert doc_info.filename == "test.pdf"
    assert doc_info.file_size == 1024
    assert doc_info.num_pages == 5
    assert doc_info.num_chunks == 10
    assert doc_info.tags == ["test"]


@pytest.mark.asyncio
async def test_get_document_not_found(document_service):
    """Test retrieving non-existent document."""
    with pytest.raises(ValueError, match="not found"):
        await document_service.get_document("nonexistent-id")


@pytest.mark.asyncio
async def test_list_documents(document_service):
    """Test listing all documents."""
    from datetime import UTC, datetime, timedelta

    from app.storage.file_storage import FileMetadata

    # Add multiple documents
    for i in range(3):
        file_metadata = FileMetadata(
            file_id=f"doc-{i}",
            filename=f"test{i}.pdf",
            file_size=1024,
            content_type="application/pdf",
            file_hash=f"hash{i}",
            upload_path=Path(f"/tmp/test{i}.pdf"),
            created_at=datetime.now(UTC) - timedelta(hours=i),
        )

        document_service._documents[f"doc-{i}"] = {
            "file_metadata": file_metadata,
            "document_metadata": DocumentMetadata(filename=f"test{i}.pdf"),
            "created_at": file_metadata.created_at,
            "num_pages": i + 1,
            "num_chunks": (i + 1) * 2,
        }

    # List all documents
    docs = await document_service.list_documents()

    assert len(docs) == 3
    # Should be sorted by creation time (newest first)
    assert docs[0].doc_id == "doc-0"
    assert docs[1].doc_id == "doc-1"
    assert docs[2].doc_id == "doc-2"


@pytest.mark.asyncio
async def test_list_documents_pagination(document_service):
    """Test document listing with pagination."""
    from datetime import UTC, datetime

    from app.storage.file_storage import FileMetadata

    # Add multiple documents
    for i in range(5):
        file_metadata = FileMetadata(
            file_id=f"doc-{i}",
            filename=f"test{i}.pdf",
            file_size=1024,
            content_type="application/pdf",
            file_hash=f"hash{i}",
            upload_path=Path(f"/tmp/test{i}.pdf"),
            created_at=datetime.now(UTC),
        )

        document_service._documents[f"doc-{i}"] = {
            "file_metadata": file_metadata,
            "document_metadata": DocumentMetadata(filename=f"test{i}.pdf"),
            "created_at": file_metadata.created_at,
            "num_pages": 1,
            "num_chunks": 1,
        }

    # Test pagination
    docs_page1 = await document_service.list_documents(skip=0, limit=2)
    assert len(docs_page1) == 2

    docs_page2 = await document_service.list_documents(skip=2, limit=2)
    assert len(docs_page2) == 2

    docs_page3 = await document_service.list_documents(skip=4, limit=2)
    assert len(docs_page3) == 1


@pytest.mark.asyncio
async def test_delete_document(
    document_service,
    mock_file_storage,
    mock_vector_store,
):
    """Test deleting a document."""
    from datetime import UTC, datetime

    from app.storage.file_storage import FileMetadata

    # Add a document
    file_metadata = FileMetadata(
        file_id="test-doc-id",
        filename="test.pdf",
        file_size=1024,
        content_type="application/pdf",
        file_hash="abc123",
        upload_path=Path("/tmp/test.pdf"),
        created_at=datetime.now(UTC),
    )

    document_service._documents["test-doc-id"] = {
        "file_metadata": file_metadata,
        "document_metadata": DocumentMetadata(filename="test.pdf"),
        "created_at": file_metadata.created_at,
        "num_pages": 1,
        "num_chunks": 1,
    }
    document_service._status["test-doc-id"] = ProcessingStatus.COMPLETED

    # Delete document
    result = await document_service.delete_document("test-doc-id")

    assert result is True
    assert "test-doc-id" not in document_service._documents
    assert "test-doc-id" not in document_service._status

    # Verify cleanup was called
    mock_vector_store.delete_document.assert_called_once_with("test-doc-id")
    mock_file_storage.delete_file.assert_called_once_with("test-doc-id")


@pytest.mark.asyncio
async def test_delete_document_not_found(document_service):
    """Test deleting non-existent document."""
    with pytest.raises(ValueError, match="not found"):
        await document_service.delete_document("nonexistent-id")


@pytest.mark.asyncio
async def test_get_processing_status(document_service):
    """Test getting processing status."""
    document_service._status["test-doc-id"] = ProcessingStatus.PARSING

    status = document_service.get_processing_status("test-doc-id")
    assert status == ProcessingStatus.PARSING


@pytest.mark.asyncio
async def test_get_processing_status_not_found(document_service):
    """Test getting status for non-existent document."""
    with pytest.raises(ValueError, match="not found"):
        document_service.get_processing_status("nonexistent-id")


@pytest.mark.asyncio
async def test_error_logging(
    document_service,
    mock_file_storage,
    mock_parser,
):
    """Test that errors are logged with complete context."""
    from datetime import UTC, datetime

    from app.storage.file_storage import FileMetadata

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
    mock_file_storage.save_file.return_value = file_metadata

    # Parser raises error
    mock_parser.parse_pdf.side_effect = Exception("Parsing failed")

    # Process document
    metadata = DocumentMetadata(filename="test.pdf")
    with pytest.raises(DocumentProcessingError):
        await document_service.process_document(
            file_content=b"fake pdf content",
            metadata=metadata,
        )

    # Verify error was logged
    doc_data = document_service._documents["test-doc-id"]
    assert "errors" in doc_data
    assert len(doc_data["errors"]) > 0

    error = doc_data["errors"][0]
    assert "timestamp" in error
    assert "stage" in error
    assert "error" in error
    assert "Parsing failed" in error["error"]
