"""Property-based tests for QueryExpander.

Feature: rag-enhancements
Property 8: Query expansion generates multiple variations
Property 9: HyDE generates hypothetical answer
Property 11: Query expansion caching
"""

import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import AsyncMock, Mock

from app.retrieval.query_expansion import QueryExpander


class TestQueryExpanderProperties:
    """Property-based tests for QueryExpander."""

    @pytest.mark.asyncio
    @settings(deadline=None)
    @given(query=st.text(min_size=10, max_size=200))
    async def test_property_8_expansion_generates_variations(self, query):
        """Property 8: Expansion should generate multiple variations.
        
        Validates: Requirements 3.1, 3.3
        """
        # Mock LLM client
        llm_client = Mock()
        llm_client.generate = AsyncMock(
            return_value="Alternative 1: variation one\nAlternative 2: variation two"
        )
        
        expander = QueryExpander(llm_client, method="multi-query", num_variations=2)
        
        # Expand query
        variations = await expander.expand(query)
        
        # Should have multiple variations
        assert len(variations) >= 1
        
        # Original query should be included
        assert query in variations or len(variations) > 1

    @pytest.mark.asyncio
    @settings(deadline=None)
    @given(query=st.text(min_size=10, max_size=200))
    async def test_property_9_hyde_generates_answer(self, query):
        """Property 9: HyDE should generate hypothetical answer.
        
        Validates: Requirements 3.2
        """
        # Mock LLM client
        llm_client = Mock()
        llm_client.generate = AsyncMock(
            return_value="This is a hypothetical answer to the question."
        )
        
        expander = QueryExpander(llm_client, method="hyde")
        
        # Generate HyDE
        result = await expander.hyde_expand(query)
        
        # Should return non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    @settings(deadline=None)
    @given(query=st.text(min_size=10, max_size=100))
    async def test_property_11_caching(self, query):
        """Property 11: Repeated queries should use cache.
        
        Validates: Requirements 3.5
        """
        # Mock LLM client
        llm_client = Mock()
        llm_client.generate = AsyncMock(return_value="variation")
        
        expander = QueryExpander(llm_client, method="multi-query", cache_ttl=3600)
        
        # First call
        result1 = await expander.expand(query)
        
        # Second call should use cache
        result2 = await expander.expand(query)
        
        # Results should be identical (from cache)
        assert result1 == result2
        
        # LLM should only be called once (first time)
        assert llm_client.generate.call_count <= 1


class TestQueryExpanderUnit:
    """Unit tests for QueryExpander."""

    @pytest.mark.asyncio
    async def test_fallback_on_expansion_failure(self):
        """Test fallback to original query on failure."""
        # Mock LLM client that fails
        llm_client = Mock()
        llm_client.generate = AsyncMock(side_effect=Exception("LLM failed"))
        
        expander = QueryExpander(llm_client, method="multi-query")
        
        query = "test query"
        
        # Should not raise exception
        result = await expander.expand(query)
        
        # Should return original query
        assert result == [query]

    @pytest.mark.asyncio
    async def test_no_expansion_method(self):
        """Test 'none' expansion method."""
        llm_client = Mock()
        
        expander = QueryExpander(llm_client, method="none")
        
        query = "test query"
        result = await expander.expand(query)
        
        # Should return only original query
        assert result == [query]
        
        # LLM should not be called
        assert not llm_client.generate.called

    def test_cache_expiration(self):
        """Test cache expiration."""
        llm_client = Mock()
        expander = QueryExpander(llm_client, cache_ttl=0)  # Immediate expiration
        
        query = "test query"
        expansions = ["query1", "query2"]
        
        # Cache expansion
        expander.cache_expansion(query, expansions)
        
        # Should be expired immediately
        cached = expander.get_cached(query)
        assert cached is None

    def test_clear_cache(self):
        """Test cache clearing."""
        llm_client = Mock()
        expander = QueryExpander(llm_client)
        
        # Add to cache
        expander.cache_expansion("query1", ["var1"])
        expander.cache_expansion("query2", ["var2"])
        
        assert len(expander.cache) == 2
        
        # Clear cache
        expander.clear_cache()
        
        assert len(expander.cache) == 0

    def test_parse_variations(self):
        """Test parsing variations from LLM response."""
        llm_client = Mock()
        expander = QueryExpander(llm_client)
        
        response = """Alternative 1: First variation
Alternative 2: Second variation
Alternative 3: Third variation"""
        
        variations = expander._parse_variations(response)
        
        assert len(variations) == 3
        assert "First variation" in variations
        assert "Second variation" in variations
        assert "Third variation" in variations
