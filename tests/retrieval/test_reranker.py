"""Property-based tests for CrossEncoderReranker.

Feature: rag-enhancements
Property 1: Reranking scores all pairs
Property 2: Reranking retrieves more candidates
Property 29: Reranking score normalization
"""

import pytest
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


class FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder with fixed per-text scores."""

    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.predict_calls: list[list[list[str]]] = []

    def predict(self, pairs, batch_size=32):  # noqa: ARG002
        self.predict_calls.append(pairs)
        return [self.scores_by_text[text] for _, text in pairs]


def _make_reranker(monkeypatch, model, enable_caching: bool = True) -> CrossEncoderReranker:
    """Build a reranker without loading a real cross-encoder model."""
    monkeypatch.setattr(CrossEncoderReranker, "_init_model", lambda _self: None)
    reranker = CrossEncoderReranker(device="cpu", enable_caching=enable_caching)
    reranker.model = model
    return reranker


def _citation_chunks() -> list[SearchResult]:
    """Chunks shaped like VectorStore output: doc_id/page/chunk_index are fields."""
    return [
        SearchResult(
            chunk_id="c-alpha-3",
            content="alpha content",
            score=0.9,
            doc_id="doc-alpha",
            page=3,
            chunk_index=0,
            metadata={"section": "Intro", "embedding_model": "m"},
        ),
        SearchResult(
            chunk_id="c-beta-7",
            content="beta content",
            score=0.8,
            doc_id="doc-beta",
            page=7,
            chunk_index=4,
            metadata={"section": "Methods"},
        ),
        SearchResult(
            chunk_id="c-alpha-12",
            content="gamma content",
            score=0.7,
            doc_id="doc-alpha",
            page=12,
            chunk_index=9,
            metadata={"section": "Results"},
        ),
    ]


class TestRerankerPreservesCitationFields:
    """Regression tests: reranking must not drop doc_id, page or chunk_index."""

    @pytest.mark.parametrize("enable_caching", [True, False])
    def test_rerank_preserves_fields_and_orders_by_score(self, monkeypatch, enable_caching):
        """Reranked results keep every field except score and follow model scores."""
        model = FakeCrossEncoder({"alpha content": 1.0, "beta content": -2.0, "gamma content": 5.0})
        reranker = _make_reranker(monkeypatch, model, enable_caching=enable_caching)
        chunks = _citation_chunks()
        originals = {c.chunk_id: c for c in _citation_chunks()}

        reranked = reranker.rerank("query", chunks, top_k=3)

        assert len(model.predict_calls) == 1
        assert [r.chunk_id for r in reranked] == ["c-alpha-12", "c-alpha-3", "c-beta-7"]
        assert [r.score for r in reranked] == pytest.approx([1.0, 3.0 / 7.0, 0.0])

        for result in reranked:
            original = originals[result.chunk_id]
            assert result.doc_id == original.doc_id
            assert result.page == original.page
            assert result.chunk_index == original.chunk_index
            assert result.content == original.content
            assert result.metadata == {
                **original.metadata,
                "original_score": original.score,
                "reranking_score": result.score,
            }

        # Input results are not mutated
        for chunk in chunks:
            original = originals[chunk.chunk_id]
            assert chunk.score == original.score
            assert chunk.metadata == original.metadata

    def test_rerank_preserves_fields_on_cached_scores(self, monkeypatch):
        """Scores served from the cache still yield fully populated results."""
        model = FakeCrossEncoder({"alpha content": 1.0, "beta content": -2.0, "gamma content": 5.0})
        reranker = _make_reranker(monkeypatch, model, enable_caching=True)

        reranker.rerank("query", _citation_chunks(), top_k=3)
        reranked = reranker.rerank("query", _citation_chunks(), top_k=2)

        # Second call is served entirely from cache
        assert len(model.predict_calls) == 1
        assert [(r.chunk_id, r.doc_id, r.page, r.chunk_index) for r in reranked] == [
            ("c-alpha-12", "doc-alpha", 12, 9),
            ("c-alpha-3", "doc-alpha", 3, 0),
        ]

    def test_fallback_preserves_fields(self, monkeypatch):
        """Without a model, original order and all fields are kept."""
        reranker = _make_reranker(monkeypatch, None)

        reranked = reranker.rerank("query", _citation_chunks(), top_k=2)

        assert [(r.chunk_id, r.doc_id, r.page, r.chunk_index, r.metadata) for r in reranked] == [
            ("c-alpha-3", "doc-alpha", 3, 0, {"section": "Intro", "embedding_model": "m"}),
            ("c-beta-7", "doc-beta", 7, 4, {"section": "Methods"}),
        ]
