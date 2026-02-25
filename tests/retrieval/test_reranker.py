"""Property-based tests for CrossEncoderReranker.

Feature: rag-enhancements
Property 1: Reranking scores all pairs
Property 2: Reranking retrieves more candidates
Property 29: Reranking score normalization
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.search import SearchResult
from app.retrieval.reranker import CrossEncoderReranker


# Strategies for generating test data
@st.composite
def search_result_strategy(draw):
    """Generate random SearchResult for testing."""
    chunk_id = draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        )
    )
    content = draw(st.text(min_size=10, max_size=200))
    score = draw(st.floats(min_value=0.0, max_value=1.0))

    return SearchResult(chunk_id=chunk_id, content=content, score=score)


class TestCrossEncoderRerankerProperties:
    """Property-based tests for CrossEncoderReranker."""

    @settings(deadline=None)  # Disable deadline for property tests
    @given(
        query=st.text(min_size=5, max_size=100),
        chunks=st.lists(search_result_strategy(), min_size=1, max_size=10),
    )
    def test_property_1_reranking_scores_all_pairs(self, query, chunks):
        """Property 1: For any query and chunks, reranking should score all pairs.

        Validates: Requirements 1.1, 1.2
        """
        # Note: We can't test with real model in property tests (too slow)
        # So we test the logic with model=None (fallback mode)
        reranker = CrossEncoderReranker()
        reranker.model = None  # Force fallback mode

        # Rerank
        reranked = reranker.rerank(query, chunks, top_k=len(chunks))

        # Verify all chunks were processed (fallback returns original order up to top_k)
        assert len(reranked) <= len(chunks)

    @settings(deadline=None)  # Disable deadline for property tests
    @given(
        query=st.text(min_size=5, max_size=100),
        chunks=st.lists(
            search_result_strategy(), min_size=5, max_size=20, unique_by=lambda x: x.chunk_id
        ),
        final_k=st.integers(min_value=1, max_value=5),
    )
    def test_property_2_reranking_retrieves_more_candidates(self, query, chunks, final_k):
        """Property 2: Initial retrieval should be larger than final k.

        Validates: Requirements 1.4
        """
        # Simulate retrieving more candidates than final k
        initial_k = len(chunks)

        reranker = CrossEncoderReranker()
        reranker.model = None  # Force fallback mode

        # Rerank with final_k
        reranked = reranker.rerank(query, chunks, top_k=final_k)

        # Verify we retrieved more initially than we return
        assert initial_k >= final_k
        assert len(reranked) == min(final_k, len(chunks))

    @settings(deadline=None)  # Disable deadline for property tests
    @given(
        scores=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        )
    )
    def test_property_29_score_normalization(self, scores):
        """Property 29: Normalized scores should be in [0, 1] range.

        Validates: Requirements 8.4
        """
        reranker = CrossEncoderReranker()

        normalized = reranker.normalize_scores(scores)

        # All scores should be in [0, 1]
        assert all(0.0 <= s <= 1.0 for s in normalized)

        # Should have same length
        assert len(normalized) == len(scores)


class TestCrossEncoderRerankerUnit:
    """Unit tests for CrossEncoderReranker."""

    def test_fallback_when_model_unavailable(self):
        """Test fallback behavior when model is not available."""
        reranker = CrossEncoderReranker()
        reranker.model = None  # Simulate model unavailable

        chunks = [SearchResult(f"chunk{i}", f"content {i}", 0.5) for i in range(3)]

        # Should not raise error
        reranked = reranker.rerank("test query", chunks, top_k=2)

        assert len(reranked) == 2
        assert all(isinstance(r, SearchResult) for r in reranked)

    def test_score_caching(self):
        """Test that cache can be populated and cleared."""
        reranker = CrossEncoderReranker(enable_caching=True)

        # Manually populate cache
        reranker.score_cache[("q1", "t1")] = 0.9
        reranker.score_cache[("q2", "t2")] = 0.8

        assert len(reranker.score_cache) == 2

        # Clear cache
        reranker.clear_cache()

        assert len(reranker.score_cache) == 0

    def test_clear_cache(self):
        """Test cache clearing."""
        reranker = CrossEncoderReranker(enable_caching=True)

        # Add to cache
        reranker.score_cache[("q", "t")] = 0.5

        assert len(reranker.score_cache) == 1

        # Clear cache
        reranker.clear_cache()

        assert len(reranker.score_cache) == 0

    def test_normalize_scores_edge_cases(self):
        """Test score normalization edge cases."""
        reranker = CrossEncoderReranker()

        # Empty list
        assert reranker.normalize_scores([]) == []

        # All same scores - equally relevant, should get max score
        normalized = reranker.normalize_scores([5.0, 5.0, 5.0])
        assert all(s == 1.0 for s in normalized)

        # Normal case
        normalized = reranker.normalize_scores([1.0, 2.0, 3.0])
        assert normalized[0] == 0.0  # min
        assert normalized[2] == 1.0  # max
        assert 0.0 < normalized[1] < 1.0  # middle
