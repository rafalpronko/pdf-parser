"""Property-based tests for BM25Index.

Feature: rag-enhancements
Property 31: BM25 index updates on document processing
Property 32: BM25 index cleanup on deletion
Property 33: BM25 index persistence
"""

import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from app.models.chunk import TextChunk
from app.retrieval.bm25_index import BM25Index


# Strategies for generating test data
@st.composite
def text_chunk_strategy(draw):
    """Generate random TextChunk for testing."""
    chunk_id = draw(
        st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
        )
    )
    doc_id = draw(
        st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
        )
    )
    content = draw(st.text(min_size=10, max_size=200))
    page = draw(st.integers(min_value=0, max_value=100))
    chunk_index = draw(st.integers(min_value=0, max_value=50))

    return TextChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        page=page,
        chunk_index=chunk_index,
        metadata={},
    )


class TestBM25IndexProperties:
    """Property-based tests for BM25Index."""

    @given(st.lists(text_chunk_strategy(), min_size=1, max_size=10))
    def test_property_31_index_updates_on_processing(self, chunks):
        """Property 31: For any document processed, chunks should be in BM25 index.

        Validates: Requirements 9.1
        """
        index = BM25Index()

        # Extract data from chunks
        doc_ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.content for chunk in chunks]
        metadata = [{"doc_id": chunk.doc_id, "page": chunk.page} for chunk in chunks]

        # Add documents
        index.add_documents(doc_ids, texts, metadata)

        # Verify all documents are in index
        assert len(index.doc_ids) == len(chunks)
        for doc_id in doc_ids:
            assert doc_id in index.doc_ids

    @given(st.lists(text_chunk_strategy(), min_size=2, max_size=10, unique_by=lambda x: x.chunk_id))
    def test_property_32_index_cleanup_on_deletion(self, chunks):
        """Property 32: After deletion, chunks should not be in BM25 index.

        Validates: Requirements 9.2
        """
        index = BM25Index()

        # Add all chunks (now guaranteed unique chunk_ids)
        doc_ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.content for chunk in chunks]
        metadata = [{"doc_id": chunk.doc_id} for chunk in chunks]
        index.add_documents(doc_ids, texts, metadata)

        # Remove first chunk
        to_remove = [doc_ids[0]]
        index.remove_documents(to_remove)

        # Verify removed chunk is not in index
        assert doc_ids[0] not in index.doc_ids
        # Verify other chunks are still there
        for doc_id in doc_ids[1:]:
            assert doc_id in index.doc_ids

    @given(st.lists(text_chunk_strategy(), min_size=1, max_size=5))
    def test_property_33_index_persistence(self, chunks):
        """Property 33: After save/load, index should contain same data.

        Validates: Requirements 9.4
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "bm25_index.pkl"

            # Create and populate index
            index1 = BM25Index(persist_path=persist_path)
            doc_ids = [chunk.chunk_id for chunk in chunks]
            texts = [chunk.content for chunk in chunks]
            metadata = [{"doc_id": chunk.doc_id} for chunk in chunks]
            index1.add_documents(doc_ids, texts, metadata)

            # Save index
            index1.save()

            # Load into new index
            index2 = BM25Index(persist_path=persist_path)
            index2.load()

            # Verify data matches
            assert index2.doc_ids == index1.doc_ids
            assert index2.texts == index1.texts
            assert len(index2.tokenized_corpus) == len(index1.tokenized_corpus)


class TestBM25IndexUnit:
    """Unit tests for BM25Index."""

    def test_add_and_search(self):
        """Test basic add and search functionality."""
        index = BM25Index()

        doc_ids = ["doc1", "doc2", "doc3"]
        texts = [
            "machine learning algorithms",
            "deep learning neural networks",
            "natural language processing",
        ]
        metadata = [{"source": f"doc{i}"} for i in range(1, 4)]

        index.add_documents(doc_ids, texts, metadata)

        # Search for "learning"
        results = index.search("learning", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(doc_id in doc_ids for doc_id, _ in results)

    def test_rebuild_from_chunks(self):
        """Test rebuilding index from chunks."""
        chunks = [
            TextChunk(
                chunk_id=f"chunk{i}",
                doc_id=f"doc{i}",
                content=f"content {i}",
                page=0,
                chunk_index=i,
                metadata={},
            )
            for i in range(3)
        ]

        index = BM25Index()
        index.rebuild_from_chunks(chunks)

        assert len(index.doc_ids) == 3
        assert all(f"chunk{i}" in index.doc_ids for i in range(3))

    def test_corruption_detection(self):
        """Test corruption detection."""
        index = BM25Index()

        # Empty index should not be corrupted
        assert not index.detect_corruption()

        # Add some data
        index.add_documents(["doc1"], ["test content"], [{"source": "test"}])

        # Valid index should not be corrupted
        assert not index.detect_corruption()

        # Manually corrupt by mismatching lengths
        index.texts.append("extra text")
        assert index.detect_corruption()
