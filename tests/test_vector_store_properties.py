"""Property-based tests for VectorStore implementation."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.models.chunk import DocumentChunk, EmbeddedChunk
from app.storage.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create a temporary vector store for testing."""
    import uuid

    # Use unique collection name for each test to ensure isolation
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    store = VectorStore(
        persist_directory=str(tmp_path / "vectordb"),
        collection_name=collection_name,
    )
    # Reset before yielding to ensure clean state
    try:
        store.reset()
    except Exception:
        pass
    yield store
    # Cleanup after test
    try:
        store.reset()
    except Exception:
        pass  # Ignore cleanup errors


# Strategy for generating valid embeddings
def embedding_strategy(dimension: int = 384):
    """Generate valid embedding vectors."""
    return st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=dimension,
        max_size=dimension,
    )


# Strategy for generating document chunks
def chunk_strategy(doc_id: str = None, chunk_id_prefix: str = "chunk"):
    """Generate valid document chunks with unique IDs."""
    # Use UUIDs to ensure uniqueness
    import uuid

    return st.builds(
        DocumentChunk,
        chunk_id=st.just(f"{chunk_id_prefix}_{uuid.uuid4().hex[:8]}"),
        doc_id=st.just(doc_id)
        if doc_id
        else st.text(
            min_size=1, max_size=50, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
        content=st.text(
            min_size=1, max_size=1000, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
        page=st.integers(min_value=0, max_value=100),
        chunk_index=st.integers(min_value=0, max_value=1000),
        metadata=st.dictionaries(
            keys=st.text(
                min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
            ),  # lowercase letters only
            values=st.one_of(
                st.text(max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
                st.integers(min_value=-1000, max_value=1000),
            ),
            max_size=5,
        ),
    )


# Strategy for generating embedded chunks
def embedded_chunk_strategy(doc_id: str = None, dimension: int = 384):
    """Generate valid embedded chunks."""
    return st.builds(
        EmbeddedChunk,
        chunk=chunk_strategy(doc_id=doc_id),
        embedding=embedding_strategy(dimension=dimension),
        modality=st.just("text"),
    )


class TestEmbeddingStorageRoundTrip:
    """Property-based tests for embedding storage round-trip."""

    @pytest.mark.asyncio
    @given(
        num_chunks=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_14_embedding_storage_round_trip(self, vector_store, num_chunks):
        """Feature: pdf-rag-system, Property 14: Embedding storage round-trip.

        For any embedded chunk stored in the vector database, retrieving it by its
        chunk ID should return the same embedding vector and all associated metadata
        (doc_id, page, content).

        Validates: Requirements 4.2, 4.3
        """
        # Generate chunks with unique IDs
        embedded_chunks = []
        for i in range(num_chunks):
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=f"Test content {i}",
                page=i % 5,
                chunk_index=i,
                metadata={"index": i, "type": "test"},
            )
            embedding = [float(i) * 0.1] * 384
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text"))

        # Store embeddings
        await vector_store.add_embeddings(
            embeddings=embedded_chunks,
            doc_id="test_doc",
        )

        # Property: For each stored chunk, retrieval should return the same data
        for embedded_chunk in embedded_chunks:
            chunk = embedded_chunk.chunk

            # Retrieve the chunk
            retrieved = await vector_store.get_chunk(chunk.chunk_id)

            # Property: Chunk should be found
            assert retrieved is not None, f"Chunk {chunk.chunk_id} not found after storage"

            # Property: doc_id should match
            assert retrieved.doc_id == chunk.doc_id, (
                f"doc_id mismatch: expected {chunk.doc_id}, got {retrieved.doc_id}"
            )

            # Property: page should match
            assert retrieved.page == chunk.page, (
                f"page mismatch: expected {chunk.page}, got {retrieved.page}"
            )

            # Property: content should match
            assert retrieved.content == chunk.content, (
                f"content mismatch for chunk {chunk.chunk_id}"
            )

            # Property: chunk_index should match
            assert retrieved.chunk_index == chunk.chunk_index, (
                f"chunk_index mismatch: expected {chunk.chunk_index}, got {retrieved.chunk_index}"
            )

            # Property: metadata should be preserved (excluding system fields)
            for key, value in chunk.metadata.items():
                assert key in retrieved.metadata, f"metadata key '{key}' missing after round-trip"
                assert retrieved.metadata[key] == value, f"metadata value mismatch for key '{key}'"

    @pytest.mark.asyncio
    @given(
        num_chunks=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_14_round_trip_with_varying_chunk_counts(self, vector_store, num_chunks):
        """Feature: pdf-rag-system, Property 14: Embedding storage round-trip.

        For any number of chunks, storage and retrieval should preserve all data.

        Validates: Requirements 4.2, 4.3
        """
        # Generate chunks with fixed dimension (384 - standard for text-embedding-3-small)
        dimension = 384
        embedded_chunks = []
        for i in range(num_chunks):
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=f"Content {i}",
                page=i % 10,
                chunk_index=i,
                metadata={"index": i},
            )
            embedding = [float(i % 100) / 100.0] * dimension
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text"))

        # Store embeddings
        await vector_store.add_embeddings(
            embeddings=embedded_chunks,
            doc_id="test_doc",
        )

        # Property: All chunks should be retrievable
        count = await vector_store.count_chunks(doc_id="test_doc")
        assert count == num_chunks, f"Expected {num_chunks} chunks, found {count}"

        # Property: Each chunk should have correct data
        for i, embedded_chunk in enumerate(embedded_chunks):
            retrieved = await vector_store.get_chunk(f"chunk_{i}")
            assert retrieved is not None
            assert retrieved.content == f"Content {i}"
            assert retrieved.page == i % 10
            assert retrieved.chunk_index == i


class TestReprocessingIdempotence:
    """Property-based tests for reprocessing idempotence."""

    @pytest.mark.asyncio
    @given(
        num_chunks=st.integers(min_value=1, max_value=10),
        num_reprocesses=st.integers(min_value=2, max_value=5),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_15_reprocessing_idempotence(
        self, vector_store, num_chunks, num_reprocesses
    ):
        """Feature: pdf-rag-system, Property 15: Reprocessing idempotence.

        For any document processed multiple times, the second and subsequent
        processings should update existing embeddings rather than create duplicates,
        and the total number of chunks in the database should equal the number from
        a single processing.

        Validates: Requirements 4.4
        """

        # Create chunks with fixed IDs for reprocessing
        def create_chunks():
            chunks = []
            for i in range(num_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{i}",
                    doc_id="test_doc",
                    content=f"Content {i}",
                    page=i,
                    chunk_index=i,
                    metadata={},
                )
                embedding = [float(i) * 0.1] * 384
                chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text"))
            return chunks

        # Process the document multiple times
        for iteration in range(num_reprocesses):
            embedded_chunks = create_chunks()
            await vector_store.add_embeddings(
                embeddings=embedded_chunks,
                doc_id="test_doc",
            )

            # Property: Count should remain constant after first processing
            count = await vector_store.count_chunks(doc_id="test_doc")
            assert count == num_chunks, (
                f"Iteration {iteration}: Expected {num_chunks} chunks, found {count}"
            )

        # Property: Final count should equal original chunk count (no duplicates)
        final_count = await vector_store.count_chunks(doc_id="test_doc")
        assert final_count == num_chunks, (
            f"After {num_reprocesses} reprocesses: expected {num_chunks} chunks, found {final_count}"
        )

        # Property: All chunks should still be retrievable with correct data
        for i in range(num_chunks):
            retrieved = await vector_store.get_chunk(f"chunk_{i}")
            assert retrieved is not None, (
                f"Chunk chunk_{i} not found after {num_reprocesses} reprocesses"
            )
            assert retrieved.content == f"Content {i}"

    @pytest.mark.asyncio
    @given(
        num_chunks_first=st.integers(min_value=1, max_value=10),
        num_chunks_second=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_15_reprocessing_with_different_chunk_counts(
        self, vector_store, num_chunks_first, num_chunks_second
    ):
        """Feature: pdf-rag-system, Property 15: Reprocessing idempotence.

        For any document reprocessed with a different number of chunks, the second
        processing should replace all chunks from the first processing.

        Validates: Requirements 4.4
        """
        # First processing with num_chunks_first chunks
        first_chunks = []
        for i in range(num_chunks_first):
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=f"First content {i}",
                page=i,
                chunk_index=i,
                metadata={},
            )
            embedding = [float(i) * 0.1] * 384
            first_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text"))

        await vector_store.add_embeddings(
            embeddings=first_chunks,
            doc_id="test_doc",
        )

        # Verify first processing
        count_after_first = await vector_store.count_chunks(doc_id="test_doc")
        assert count_after_first == num_chunks_first

        # Second processing with num_chunks_second chunks
        second_chunks = []
        for i in range(num_chunks_second):
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=f"Second content {i}",
                page=i,
                chunk_index=i,
                metadata={},
            )
            embedding = [float(i) * 0.2] * 384
            second_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text"))

        await vector_store.add_embeddings(
            embeddings=second_chunks,
            doc_id="test_doc",
        )

        # Property: Count should equal second processing count (not sum)
        count_after_second = await vector_store.count_chunks(doc_id="test_doc")
        assert count_after_second == num_chunks_second, (
            f"Expected {num_chunks_second} chunks after reprocessing, found {count_after_second}"
        )

        # Property: Chunks should have content from second processing
        for i in range(min(num_chunks_second, 3)):  # Check first few chunks
            retrieved = await vector_store.get_chunk(f"chunk_{i}")
            if retrieved:
                assert retrieved.content == f"Second content {i}", (
                    f"Chunk {i} has wrong content: {retrieved.content}"
                )


class TestTransactionalConsistency:
    """Property-based tests for transactional consistency."""

    @pytest.mark.asyncio
    @given(
        num_chunks=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_29_transactional_consistency_on_success(self, vector_store, num_chunks):
        """Feature: pdf-rag-system, Property 29: Transactional consistency.

        For any document processing operation that completes successfully, all
        chunks should be stored together, ensuring no partial data remains.

        Validates: Requirements 10.5
        """
        # Generate chunks
        embedded_chunks = []
        for i in range(num_chunks):
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=f"Content {i}",
                page=i,
                chunk_index=i,
                metadata={},
            )
            embedding = [float(i) * 0.1] * 384
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text"))

        # Store embeddings
        await vector_store.add_embeddings(
            embeddings=embedded_chunks,
            doc_id="test_doc",
        )

        # Property: All chunks should be stored (no partial storage)
        count = await vector_store.count_chunks(doc_id="test_doc")
        assert count == num_chunks, (
            f"Expected {num_chunks} chunks, found {count} (partial storage detected)"
        )

        # Property: All chunks should be retrievable
        for i in range(num_chunks):
            retrieved = await vector_store.get_chunk(f"chunk_{i}")
            assert retrieved is not None, f"Chunk {i} not found (inconsistent storage)"
            assert retrieved.doc_id == "test_doc"
            assert retrieved.content == f"Content {i}"

    @pytest.mark.asyncio
    @given(
        num_chunks=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_29_no_orphaned_chunks_after_deletion(self, vector_store, num_chunks):
        """Feature: pdf-rag-system, Property 29: Transactional consistency.

        For any document deletion operation, all chunks associated with the document
        should be removed, ensuring no orphaned chunks remain.

        Validates: Requirements 10.5
        """
        # Generate and store chunks
        embedded_chunks = []
        for i in range(num_chunks):
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=f"Content {i}",
                page=i,
                chunk_index=i,
                metadata={},
            )
            embedding = [float(i) * 0.1] * 384
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text"))

        await vector_store.add_embeddings(
            embeddings=embedded_chunks,
            doc_id="test_doc",
        )

        # Verify chunks exist
        count_before = await vector_store.count_chunks(doc_id="test_doc")
        assert count_before == num_chunks

        # Delete document
        await vector_store.delete_document(doc_id="test_doc")

        # Property: No chunks should remain for this document
        count_after = await vector_store.count_chunks(doc_id="test_doc")
        assert count_after == 0, f"Found {count_after} orphaned chunks after deletion (expected 0)"

        # Property: Individual chunk retrieval should return None
        for i in range(num_chunks):
            retrieved = await vector_store.get_chunk(f"chunk_{i}")
            assert retrieved is None, (
                f"Chunk {i} still exists after document deletion (orphaned chunk)"
            )

    @pytest.mark.asyncio
    @given(
        num_docs=st.integers(min_value=2, max_value=5),
        chunks_per_doc=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_29_deletion_isolation_between_documents(
        self, vector_store, num_docs, chunks_per_doc
    ):
        """Feature: pdf-rag-system, Property 29: Transactional consistency.

        For any document deletion operation, only chunks from that specific document
        should be removed, and chunks from other documents should remain intact.

        Validates: Requirements 10.5
        """
        # Reset store to ensure clean state for this example
        vector_store.reset()

        # Store chunks for multiple documents
        all_doc_ids = []
        for doc_idx in range(num_docs):
            doc_id = f"doc_{doc_idx}"
            all_doc_ids.append(doc_id)

            embedded_chunks = []
            for chunk_idx in range(chunks_per_doc):
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    content=f"Content for {doc_id} chunk {chunk_idx}",
                    page=chunk_idx,
                    chunk_index=chunk_idx,
                    metadata={},
                )
                embedding = [float(doc_idx * 10 + chunk_idx) * 0.1] * 384
                embedded_chunks.append(
                    EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text")
                )

            await vector_store.add_embeddings(
                embeddings=embedded_chunks,
                doc_id=doc_id,
            )

        # Verify all documents have correct chunk counts
        total_before = await vector_store.count_chunks()
        assert total_before == num_docs * chunks_per_doc, (
            f"Expected {num_docs * chunks_per_doc} total chunks, found {total_before}"
        )

        # Delete one document (the middle one)
        doc_to_delete = all_doc_ids[num_docs // 2]
        await vector_store.delete_document(doc_id=doc_to_delete)

        # Property: Deleted document should have no chunks
        deleted_count = await vector_store.count_chunks(doc_id=doc_to_delete)
        assert deleted_count == 0, (
            f"Deleted document {doc_to_delete} still has {deleted_count} chunks"
        )

        # Property: Other documents should still have all their chunks
        for doc_id in all_doc_ids:
            if doc_id != doc_to_delete:
                count = await vector_store.count_chunks(doc_id=doc_id)
                assert count == chunks_per_doc, (
                    f"Document {doc_id} has {count} chunks, expected {chunks_per_doc} (deletion affected wrong document)"
                )

        # Property: Total count should be reduced by exactly chunks_per_doc
        total_after = await vector_store.count_chunks()
        expected_total = (num_docs - 1) * chunks_per_doc
        assert total_after == expected_total, (
            f"Total chunks: expected {expected_total}, found {total_after}"
        )

    @pytest.mark.asyncio
    async def test_property_29_empty_embeddings_list_fails_cleanly(self, vector_store):
        """Feature: pdf-rag-system, Property 29: Transactional consistency.

        For any invalid operation (like adding empty embeddings), the system should
        fail cleanly without leaving partial data.

        Validates: Requirements 10.5
        """
        # Property: Adding empty embeddings should raise ValueError
        with pytest.raises(ValueError, match="Cannot add empty embeddings list"):
            await vector_store.add_embeddings(
                embeddings=[],
                doc_id="test_doc",
            )

        # Property: No partial data should exist after failed operation
        count = await vector_store.count_chunks(doc_id="test_doc")
        assert count == 0, f"Found {count} chunks after failed operation (partial data detected)"
