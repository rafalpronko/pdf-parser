"""Regression tests: DocumentService and QueryService must share one BM25 index.

Previously each service built and loaded its own ``BM25Index``. Uploads and deletes
only touched the DocumentService copy, so keyword search in QueryService did not see
new documents and kept returning (and citing) deleted ones until a restart.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.models.chunk import DocumentChunk
from app.models.document import DocumentMetadata
from app.models.parsing import ParsedDocument, TextBlock
from app.retrieval.bm25_index import BM25Index
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.storage.file_storage import FileMetadata

QUERY = "self-attention transformers"

DOC_ID = "doc-transformers"
CHUNK_ID = "doc-transformers-chunk-0"
CHUNK_TEXT = "Transformers rely on multi-head self-attention instead of recurrence."
# (chunk_id, doc_id, page, chunk_index, content)
TARGET_CHUNK = (CHUNK_ID, DOC_ID, 3, 0, CHUNK_TEXT)

# BM25Okapi gives a non-positive IDF to terms found in half or more of the corpus,
# so unrelated chunks are needed for the target chunk to score above zero.
OTHER_CHUNKS = [
    (
        "doc-resnet-chunk-0",
        "doc-resnet",
        1,
        0,
        "Residual networks add identity shortcut connections between layers.",
    ),
    (
        "doc-resnet-chunk-1",
        "doc-resnet",
        2,
        1,
        "Batch normalization stabilizes training of very deep convolutional networks.",
    ),
]


@pytest.fixture
def make_settings(tmp_path, monkeypatch):
    """Factory installing hermetic Settings (temp paths, dummy key) for both services."""

    def _make(*, hybrid: bool = True) -> Settings:
        settings = Settings(
            _env_file=None,
            openai_api_key=SecretStr("sk-test-dummy-key-for-unit-tests-only"),
            upload_dir=tmp_path / "uploads",
            vector_db_path=str(tmp_path / "vectordb"),
            enable_hybrid_search=hybrid,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_gpu=False,
        )
        monkeypatch.setattr("app.config._settings", settings)
        monkeypatch.setattr("app.services.document_service.get_settings", lambda: settings)
        monkeypatch.setattr("app.services.query_service.get_settings", lambda: settings)
        return settings

    return _make


def _mock_openai_client() -> MagicMock:
    client = MagicMock()
    client.embed_text = AsyncMock(return_value=[0.1] * 8)
    client.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])
    client.close = AsyncMock()
    return client


def _mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store.add_embeddings = AsyncMock(return_value=True)
    store.delete_document = AsyncMock(return_value=True)
    return store


def _build_document_service() -> DocumentService:
    """DocumentService with mocked I/O collaborators but its default BM25 index."""
    file_storage = MagicMock()
    file_storage.save_file = AsyncMock()
    file_storage.delete_file = MagicMock(return_value=True)

    return DocumentService(
        file_storage=file_storage,
        parser=MagicMock(),
        chunker=MagicMock(),
        openai_client=_mock_openai_client(),
        vector_store=_mock_vector_store(),
    )


def _build_query_service(**kwargs) -> QueryService:
    """QueryService with mocked OpenAI client and vector store."""
    return QueryService(
        openai_client=_mock_openai_client(),
        vector_store=_mock_vector_store(),
        **kwargs,
    )


def _add_chunks(index: BM25Index, chunks: list[tuple]) -> None:
    """Add chunks to a BM25 index the same way DocumentService does."""
    index.add_documents(
        [chunk_id for chunk_id, *_ in chunks],
        [content for *_, content in chunks],
        [
            {"doc_id": doc_id, "page": page, "chunk_index": chunk_index}
            for _, doc_id, page, chunk_index, _ in chunks
        ],
    )


def _persist_chunks(settings: Settings, chunks: list[tuple]) -> None:
    """Save a BM25 index with the given chunks to the configured persist path."""
    index = BM25Index(persist_path=Path(settings.vector_db_path) / "bm25_index.json")
    _add_chunks(index, chunks)
    index.save()


async def _keyword_chunk_ids(query_service: QueryService) -> list[str]:
    results = await query_service.hybrid_search.keyword_search(QUERY, top_k=5)
    return [result.chunk_id for result in results]


def test_query_service_defaults_to_document_service_bm25_index(make_settings):
    """QueryService(document_service=...) reuses the DocumentService's index object."""
    make_settings(hybrid=True)
    document_service = _build_document_service()

    query_service = _build_query_service(document_service=document_service)

    assert query_service.bm25_index is document_service.bm25_index
    assert query_service.hybrid_search.bm25_index is document_service.bm25_index


async def test_removed_and_added_chunks_are_visible_to_keyword_search(make_settings):
    """Changes made through document_service.bm25_index reach QueryService immediately."""
    settings = make_settings(hybrid=True)
    _persist_chunks(settings, [TARGET_CHUNK, *OTHER_CHUNKS])

    document_service = _build_document_service()
    query_service = _build_query_service(document_service=document_service)
    assert await _keyword_chunk_ids(query_service) == [CHUNK_ID]

    # Deleted chunks must not be served (and cited) by keyword search any more
    document_service.bm25_index.remove_documents([CHUNK_ID])
    assert await _keyword_chunk_ids(query_service) == []

    # Newly indexed chunks are searchable without a restart
    _add_chunks(document_service.bm25_index, [TARGET_CHUNK])
    assert await _keyword_chunk_ids(query_service) == [CHUNK_ID]


async def test_uploaded_then_deleted_document_tracks_keyword_search(make_settings):
    """Upload -> keyword hit; DELETE -> the deleted chunk is never returned again."""
    settings = make_settings(hybrid=True)
    _persist_chunks(settings, OTHER_CHUNKS)

    document_service = _build_document_service()
    query_service = _build_query_service(document_service=document_service)
    assert await _keyword_chunk_ids(query_service) == []

    document_service.file_storage.save_file.return_value = FileMetadata(
        file_id=DOC_ID,
        filename="attention.pdf",
        file_size=1024,
        content_type="application/pdf",
        file_hash="abc123",
        upload_path=Path("/nonexistent/attention.pdf"),
        created_at=datetime.now(UTC),
    )
    document_service.parser.parse_pdf.return_value = ParsedDocument(
        text_blocks=[TextBlock(content=CHUNK_TEXT, page=3, bbox=(0, 0, 100, 100))],
        images=[],
        tables=[],
        num_pages=3,
        metadata={},
    )
    document_service.chunker.chunk_with_structure.return_value = [
        DocumentChunk(
            chunk_id=CHUNK_ID,
            doc_id=DOC_ID,
            content=CHUNK_TEXT,
            page=3,
            chunk_index=0,
            metadata={},
        )
    ]

    await document_service.process_document(
        file_content=b"%PDF-fake", metadata=DocumentMetadata(filename="attention.pdf")
    )
    assert await _keyword_chunk_ids(query_service) == [CHUNK_ID]

    await document_service.delete_document(DOC_ID)
    assert await _keyword_chunk_ids(query_service) == []
    assert CHUNK_ID not in query_service.bm25_index.doc_ids


async def test_persisted_index_is_loaded_from_disk_once(make_settings, monkeypatch):
    """The shared index is loaded by DocumentService only, not again by QueryService."""
    settings = make_settings(hybrid=True)
    _persist_chunks(settings, [TARGET_CHUNK, *OTHER_CHUNKS])

    load_calls: list[BM25Index] = []
    original_load = BM25Index.load

    def counting_load(self: BM25Index) -> None:
        load_calls.append(self)
        original_load(self)

    monkeypatch.setattr(BM25Index, "load", counting_load)

    document_service = _build_document_service()
    query_service = _build_query_service(document_service=document_service)

    assert load_calls == [document_service.bm25_index]
    assert await _keyword_chunk_ids(query_service) == [CHUNK_ID]


def test_explicit_bm25_index_takes_precedence(make_settings):
    """An explicitly passed index wins over document_service.bm25_index."""
    make_settings(hybrid=True)
    document_service = _build_document_service()
    explicit_index = BM25Index(persist_path=None)

    query_service = _build_query_service(
        document_service=document_service, bm25_index=explicit_index
    )

    assert query_service.bm25_index is explicit_index
    assert query_service.hybrid_search.bm25_index is explicit_index


async def test_without_document_service_builds_and_loads_own_index(make_settings):
    """Standalone QueryService still creates its own index and loads it from disk."""
    settings = make_settings(hybrid=True)
    _persist_chunks(settings, [TARGET_CHUNK, *OTHER_CHUNKS])

    query_service = _build_query_service()

    assert isinstance(query_service.bm25_index, BM25Index)
    assert query_service.bm25_index.persist_path == (
        Path(settings.vector_db_path) / "bm25_index.json"
    )
    assert query_service.hybrid_search.bm25_index is query_service.bm25_index
    assert await _keyword_chunk_ids(query_service) == [CHUNK_ID]


def test_hybrid_search_disabled_has_no_bm25_index(make_settings):
    """With hybrid search disabled QueryService keeps no BM25 index or hybrid engine."""
    make_settings(hybrid=False)
    document_service = _build_document_service()

    query_service = _build_query_service(
        document_service=document_service, bm25_index=document_service.bm25_index
    )

    assert query_service.bm25_index is None
    assert query_service.hybrid_search is None


async def test_lifespan_wires_shared_bm25_index(make_settings, monkeypatch):
    """The running app's DocumentService and QueryService share one BM25 index."""
    make_settings(hybrid=True)

    import app.main as main_module

    monkeypatch.setattr(main_module, "DocumentService", _build_document_service)
    monkeypatch.setattr(main_module, "QueryService", _build_query_service)
    monkeypatch.setattr(main_module, "document_service", None)
    monkeypatch.setattr(main_module, "query_service", None)

    async with main_module.lifespan(main_module.app):
        document_service = main_module.document_service
        query_service = main_module.query_service

        assert isinstance(document_service, DocumentService)
        assert isinstance(query_service, QueryService)
        assert query_service.bm25_index is document_service.bm25_index
        assert query_service.hybrid_search.bm25_index is document_service.bm25_index

        # Chunks indexed at upload time are immediately searchable from the query side
        _add_chunks(document_service.bm25_index, [TARGET_CHUNK, *OTHER_CHUNKS])
        assert await _keyword_chunk_ids(query_service) == [CHUNK_ID]
