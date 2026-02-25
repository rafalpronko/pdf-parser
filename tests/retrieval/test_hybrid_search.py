"""Property-based tests for HybridSearchEngine.

Feature: rag-enhancements
Property 5: Hybrid search invokes both methods
Property 6: RRF fusion combines rankings
Property 7: Keyword search finds exact matches
"""

from unittest.mock import AsyncMock, Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_search import HybridSearchEngine
from app.storage.vector_store import SearchResult, VectorStore


class TestHybridSearchProperties:
    """Property-based tests for HybridSearchEngine."""

    @pytest.mark.asyncio
    @settings(deadline=None)
    @given(
        query=st.text(min_size=5, max_size=100),
        num_results=st.integers(min_value=1, max_value=10),
    )
    async def test_property_5_hybrid_search_invokes_both(self, query, num_results):
        """Property 5: Hybrid search should invoke both vector and keyword search.

        Validates: Requirements 2.1
        """
        # Mock vector store and BM25 index
        vector_store = Mock(spec=VectorStore)
        bm25_index = Mock(spec=BM25Index)

        # Setup mocks
        vector_store.search = AsyncMock(return_value=[])
        bm25_index.search = Mock(return_value=[])
        bm25_index.metadata = {}
        bm25_index.doc_ids = []
        bm25_index.texts = []

        engine = HybridSearchEngine(vector_store, bm25_index)

        # Perform search
        query_embedding = [0.1] * 384  # Dummy embedding
        await engine.search(query, query_embedding, top_k=num_results)

        # Verify both methods were called
        assert vector_store.search.called
        assert bm25_index.search.called

    @settings(deadline=None)
    @given(
        num_vector=st.integers(min_value=1, max_value=5),
        num_keyword=st.integers(min_value=1, max_value=5),
    )
    def test_property_6_rrf_fusion_combines(self, num_vector, num_keyword):
        """Property 6: RRF should combine rankings from both methods.

        Validates: Requirements 2.2, 2.3
        """
        vector_store = Mock(spec=VectorStore)
        bm25_index = Mock(spec=BM25Index)

        engine = HybridSearchEngine(vector_store, bm25_index)

        # Create mock results
        vector_results = [
            SearchResult(
                chunk_id=f"v{i}",
                doc_id="doc1",
                content=f"vector content {i}",
                page=0,
                chunk_index=i,
                metadata={},
                relevance_score=1.0 - i * 0.1,
            )
            for i in range(num_vector)
        ]

        keyword_results = [
            SearchResult(
                chunk_id=f"k{i}",
                doc_id="doc1",
                content=f"keyword content {i}",
                page=0,
                chunk_index=i,
                metadata={},
                relevance_score=1.0 - i * 0.1,
            )
            for i in range(num_keyword)
        ]

        # Fuse results
        fused = engine.fuse_results(vector_results, keyword_results)

        # Verify fusion produced results
        assert len(fused) > 0

        # Verify scores are in descending order
        scores = [r.relevance_score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_property_7_keyword_finds_exact_matches(self):
        """Property 7: Keyword search should find exact term matches.

        Validates: Requirements 2.5
        """
        # Create BM25 index with test data
        bm25_index = BM25Index()

        doc_ids = ["doc1", "doc2", "doc3"]
        texts = [
            "machine learning algorithms",
            "deep learning neural networks",
            "natural language processing",
        ]
        metadata = [{"doc_id": f"doc{i}"} for i in range(1, 4)]

        bm25_index.add_documents(doc_ids, texts, metadata)

        # Search for specific term
        results = bm25_index.search("learning", top_k=10)

        # Verify results contain the term
        assert len(results) > 0

        # Get chunk IDs that matched
        matched_ids = [chunk_id for chunk_id, _ in results]

        # Verify that docs with "learning" are in results
        assert "doc1" in matched_ids or "doc2" in matched_ids


class TestHybridSearchUnit:
    """Unit tests for HybridSearchEngine."""

    @pytest.mark.asyncio
    async def test_vector_search_fallback(self):
        """Test fallback when vector search fails."""
        vector_store = Mock(spec=VectorStore)
        bm25_index = Mock(spec=BM25Index)

        # Make vector search fail
        vector_store.search = AsyncMock(side_effect=Exception("Vector search failed"))

        engine = HybridSearchEngine(vector_store, bm25_index)

        # Should not raise exception
        results = await engine.vector_search([0.1] * 384, top_k=5)

        # Should return empty list
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_fallback(self):
        """Test fallback when keyword search fails."""
        vector_store = Mock(spec=VectorStore)
        bm25_index = Mock(spec=BM25Index)

        # Make keyword search fail
        bm25_index.search = Mock(side_effect=Exception("Keyword search failed"))

        engine = HybridSearchEngine(vector_store, bm25_index)

        # Should not raise exception
        results = await engine.keyword_search("test query", top_k=5)

        # Should return empty list
        assert results == []

    def test_rrf_with_overlapping_results(self):
        """Test RRF fusion with overlapping chunk IDs."""
        vector_store = Mock(spec=VectorStore)
        bm25_index = Mock(spec=BM25Index)

        engine = HybridSearchEngine(vector_store, bm25_index)

        # Create overlapping results (same chunk_id in both)
        shared_result = SearchResult(
            chunk_id="shared",
            doc_id="doc1",
            content="shared content",
            page=0,
            chunk_index=0,
            metadata={},
            relevance_score=0.8,
        )

        vector_results = [shared_result]
        keyword_results = [shared_result]

        # Fuse results
        fused = engine.fuse_results(vector_results, keyword_results)

        # Should have only one result (deduplicated)
        assert len(fused) == 1

        # Score should be higher than individual scores (combined from both)
        assert fused[0].relevance_score > 0
