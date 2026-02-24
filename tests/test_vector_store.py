"""Tests for VectorStore implementation."""

import pytest
from app.models.chunk import DocumentChunk, EmbeddedChunk
from app.storage.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create a temporary vector store for testing."""
    store = VectorStore(
        persist_directory=str(tmp_path / "vectordb"),
        collection_name="test_documents",
    )
    yield store
    # Cleanup
    store.reset()


@pytest.fixture
def sample_embedded_chunks():
    """Create sample embedded chunks for testing."""
    chunks = []
    for i in range(3):
        chunk = DocumentChunk(
            chunk_id=f"chunk_{i}",
            doc_id="doc_1",
            content=f"This is test content for chunk {i}",
            page=i,
            chunk_index=i,
            metadata={"source": "test"},
        )
        # Create simple embeddings (in reality these would be from a model)
        embedding = [float(i) * 0.1] * 384  # 384-dimensional embedding
        chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))
    return chunks


@pytest.mark.asyncio
async def test_add_embeddings(vector_store, sample_embedded_chunks):
    """Test adding embeddings to vector store."""
    result = await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )
    assert result is True

    # Verify chunks were added
    count = await vector_store.count_chunks(doc_id="doc_1")
    assert count == 3


@pytest.mark.asyncio
async def test_add_empty_embeddings_raises_error(vector_store):
    """Test that adding empty embeddings raises ValueError."""
    with pytest.raises(ValueError, match="Cannot add empty embeddings list"):
        await vector_store.add_embeddings(embeddings=[], doc_id="doc_1")


@pytest.mark.asyncio
async def test_search(vector_store, sample_embedded_chunks):
    """Test similarity search."""
    # Add embeddings
    await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )

    # Search with a query embedding
    query_embedding = [0.15] * 384
    results = await vector_store.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    assert len(results) == 2
    # Results should be ordered by relevance
    assert results[0].relevance_score >= results[1].relevance_score
    # Scores should be between 0 and 1
    for result in results:
        assert 0.0 <= result.relevance_score <= 1.0


@pytest.mark.asyncio
async def test_search_with_empty_embedding_raises_error(vector_store):
    """Test that searching with empty embedding raises ValueError."""
    with pytest.raises(ValueError, match="Query embedding cannot be empty"):
        await vector_store.search(query_embedding=[], top_k=5)


@pytest.mark.asyncio
async def test_search_with_invalid_top_k_raises_error(vector_store):
    """Test that searching with invalid top_k raises ValueError."""
    query_embedding = [0.1] * 384
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        await vector_store.search(query_embedding=query_embedding, top_k=0)


@pytest.mark.asyncio
async def test_delete_document(vector_store, sample_embedded_chunks):
    """Test document deletion with cascade."""
    # Add embeddings
    await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )

    # Verify chunks exist
    count_before = await vector_store.count_chunks(doc_id="doc_1")
    assert count_before == 3

    # Delete document
    result = await vector_store.delete_document(doc_id="doc_1")
    assert result is True

    # Verify chunks are gone
    count_after = await vector_store.count_chunks(doc_id="doc_1")
    assert count_after == 0


@pytest.mark.asyncio
async def test_delete_nonexistent_document(vector_store):
    """Test deleting a document that doesn't exist."""
    # Should not raise an error
    result = await vector_store.delete_document(doc_id="nonexistent")
    assert result is True


@pytest.mark.asyncio
async def test_idempotent_reprocessing(vector_store, sample_embedded_chunks):
    """Test that reprocessing a document updates rather than duplicates."""
    # Add embeddings first time
    await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )
    count_first = await vector_store.count_chunks(doc_id="doc_1")
    assert count_first == 3

    # Add embeddings again (reprocessing)
    await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )
    count_second = await vector_store.count_chunks(doc_id="doc_1")

    # Should still have same number of chunks, not doubled
    assert count_second == 3


@pytest.mark.asyncio
async def test_get_chunk(vector_store, sample_embedded_chunks):
    """Test retrieving a specific chunk by ID."""
    # Add embeddings
    await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )

    # Get a specific chunk
    result = await vector_store.get_chunk(chunk_id="chunk_0")
    assert result is not None
    assert result.chunk_id == "chunk_0"
    assert result.doc_id == "doc_1"
    assert result.content == "This is test content for chunk 0"
    assert result.page == 0


@pytest.mark.asyncio
async def test_get_nonexistent_chunk(vector_store):
    """Test getting a chunk that doesn't exist."""
    result = await vector_store.get_chunk(chunk_id="nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_count_chunks_all(vector_store, sample_embedded_chunks):
    """Test counting all chunks in collection."""
    # Add embeddings for multiple documents
    await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )

    # Create chunks for another document
    more_chunks = []
    for i in range(2):
        chunk = DocumentChunk(
            chunk_id=f"chunk_doc2_{i}",
            doc_id="doc_2",
            content=f"Content for doc 2 chunk {i}",
            page=i,
            chunk_index=i,
            metadata={},
        )
        embedding = [float(i) * 0.2] * 384
        more_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

    await vector_store.add_embeddings(
        embeddings=more_chunks,
        doc_id="doc_2",
    )

    # Count all chunks
    total_count = await vector_store.count_chunks()
    assert total_count == 5  # 3 from doc_1 + 2 from doc_2

    # Count chunks for specific document
    doc1_count = await vector_store.count_chunks(doc_id="doc_1")
    assert doc1_count == 3

    doc2_count = await vector_store.count_chunks(doc_id="doc_2")
    assert doc2_count == 2


@pytest.mark.asyncio
async def test_search_with_doc_filter(vector_store, sample_embedded_chunks):
    """Test searching with document ID filter."""
    # Add embeddings for doc_1
    await vector_store.add_embeddings(
        embeddings=sample_embedded_chunks,
        doc_id="doc_1",
    )

    # Add embeddings for doc_2
    more_chunks = []
    for i in range(2):
        chunk = DocumentChunk(
            chunk_id=f"chunk_doc2_{i}",
            doc_id="doc_2",
            content=f"Different content for doc 2 chunk {i}",
            page=i,
            chunk_index=i,
            metadata={},
        )
        embedding = [float(i) * 0.3] * 384
        more_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

    await vector_store.add_embeddings(
        embeddings=more_chunks,
        doc_id="doc_2",
    )

    # Search only in doc_1
    query_embedding = [0.1] * 384
    results = await vector_store.search(
        query_embedding=query_embedding,
        top_k=10,
        doc_id="doc_1",
    )

    # Should only return results from doc_1
    assert len(results) == 3
    for result in results:
        assert result.doc_id == "doc_1"
