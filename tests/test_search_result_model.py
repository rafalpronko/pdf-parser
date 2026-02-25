"""Tests for the unified SearchResult model.

Verifies that SearchResult works correctly as the single source of truth
for search results across vector store, hybrid search, and reranker.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.search import SearchResult


class TestSearchResultCreation:
    """Tests for SearchResult instantiation."""

    def test_create_with_required_fields(self):
        """Test creating SearchResult with only required fields."""
        result = SearchResult(
            chunk_id="chunk_1",
            content="Test content",
            score=0.95,
        )
        assert result.chunk_id == "chunk_1"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.doc_id == ""
        assert result.page == 0
        assert result.chunk_index == 0
        assert result.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating SearchResult with all fields."""
        metadata = {"source": "test", "section": "introduction"}
        result = SearchResult(
            chunk_id="chunk_42",
            content="Full content here",
            score=0.88,
            doc_id="doc_abc",
            page=3,
            chunk_index=7,
            metadata=metadata,
        )
        assert result.chunk_id == "chunk_42"
        assert result.doc_id == "doc_abc"
        assert result.page == 3
        assert result.chunk_index == 7
        assert result.metadata == metadata

    def test_positional_creation(self):
        """Test creating SearchResult with positional arguments."""
        result = SearchResult("id1", "content1", 0.5)
        assert result.chunk_id == "id1"
        assert result.content == "content1"
        assert result.score == 0.5


class TestSearchResultToDict:
    """Tests for SearchResult.to_dict() method."""

    def test_to_dict_contains_all_fields(self):
        """Test that to_dict() includes all fields."""
        result = SearchResult(
            chunk_id="c1",
            content="text",
            score=0.9,
            doc_id="d1",
            page=1,
            chunk_index=2,
            metadata={"key": "value"},
        )
        d = result.to_dict()

        assert d["chunk_id"] == "c1"
        assert d["content"] == "text"
        assert d["score"] == 0.9
        assert d["doc_id"] == "d1"
        assert d["page"] == 1
        assert d["chunk_index"] == 2
        assert d["metadata"] == {"key": "value"}

    def test_to_dict_returns_dict(self):
        """Test that to_dict() returns a dict type."""
        result = SearchResult("id", "content", 0.5)
        assert isinstance(result.to_dict(), dict)

    def test_to_dict_keys(self):
        """Test that to_dict() has exactly the expected keys."""
        result = SearchResult("id", "content", 0.5)
        expected_keys = {
            "chunk_id",
            "content",
            "score",
            "doc_id",
            "page",
            "chunk_index",
            "metadata",
        }
        assert set(result.to_dict().keys()) == expected_keys


class TestSearchResultMutability:
    """Tests for SearchResult field mutation (dataclass is mutable)."""

    def test_score_can_be_updated(self):
        """Test that score can be updated (needed by RRF fusion)."""
        result = SearchResult("id", "content", 0.5)
        result.score = 0.99
        assert result.score == 0.99

    def test_metadata_can_be_extended(self):
        """Test that metadata dict can be extended."""
        result = SearchResult("id", "content", 0.5, metadata={"a": 1})
        result.metadata["b"] = 2
        assert result.metadata == {"a": 1, "b": 2}


class TestSearchResultDefaultMetadata:
    """Tests for metadata default_factory isolation."""

    def test_separate_metadata_instances(self):
        """Test that each SearchResult gets its own metadata dict."""
        r1 = SearchResult("id1", "c1", 0.5)
        r2 = SearchResult("id2", "c2", 0.6)

        r1.metadata["key"] = "only_in_r1"
        assert "key" not in r2.metadata

    def test_metadata_default_is_empty_dict(self):
        """Test that default metadata is an empty dict."""
        result = SearchResult("id", "content", 0.5)
        assert result.metadata == {}
        assert isinstance(result.metadata, dict)


class TestSearchResultPropertyBased:
    """Property-based tests for SearchResult."""

    @given(
        chunk_id=st.text(min_size=1, max_size=50),
        content=st.text(min_size=0, max_size=500),
        score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        doc_id=st.text(min_size=0, max_size=50),
        page=st.integers(min_value=0, max_value=10000),
        chunk_index=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=200, deadline=None)
    def test_property_roundtrip_to_dict(self, chunk_id, content, score, doc_id, page, chunk_index):
        """Property: to_dict() preserves all field values."""
        result = SearchResult(
            chunk_id=chunk_id,
            content=content,
            score=score,
            doc_id=doc_id,
            page=page,
            chunk_index=chunk_index,
        )
        d = result.to_dict()
        assert d["chunk_id"] == chunk_id
        assert d["content"] == content
        assert d["score"] == score
        assert d["doc_id"] == doc_id
        assert d["page"] == page
        assert d["chunk_index"] == chunk_index

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_sorting_by_score(self, scores):
        """Property: SearchResults can be sorted by score descending."""
        results = [
            SearchResult(chunk_id=f"c{i}", content=f"text {i}", score=s)
            for i, s in enumerate(scores)
        ]
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].score >= sorted_results[i + 1].score
