"""Query expansion for improved retrieval coverage."""

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expands queries using HyDE or multi-query techniques.

    Supports:
    - HyDE: Generate hypothetical document for query
    - Multi-query: Generate multiple query variations
    - Caching: LRU cache with TTL
    """

    def __init__(
        self,
        llm_client: Any,
        method: str = "multi-query",
        cache_ttl: int = 3600,
        num_variations: int = 3,
    ):
        """Initialize query expander.

        Args:
            llm_client: LLM client for generation
            method: Expansion method (hyde, multi-query, none)
            cache_ttl: Cache TTL in seconds
            num_variations: Number of query variations
        """
        self.llm_client = llm_client
        self.method = method.lower()
        self.cache_ttl = cache_ttl
        self.num_variations = num_variations

        # Cache: query_hash -> (expansions, timestamp)
        self.cache: dict[str, tuple[list[str], float]] = {}

        logger.info(
            f"Initialized QueryExpander: method={method}, "
            f"cache_ttl={cache_ttl}s, num_variations={num_variations}"
        )

    async def expand(self, query: str) -> list[str]:
        """Expand query into multiple variations.

        Args:
            query: Original query

        Returns:
            List of query variations (includes original)
        """
        # Check cache first
        cached = self.get_cached(query)
        if cached is not None:
            logger.info(f"✓ Query Expansion (CACHED) - method={self.method}")
            logger.info(f"  Original query: {query}")
            logger.info(f"  Cached expansions ({len(cached)}):")
            for i, exp in enumerate(cached, 1):
                logger.info(f"    {i}. {exp}")
            return cached

        logger.info(f"→ Query Expansion START - method={self.method}")
        logger.info(f"  Original query: {query}")

        # Expand based on method
        try:
            if self.method == "hyde":
                expansions = await self.hyde_expand(query)
            elif self.method == "multi-query":
                expansions = await self.multi_query_expand(query)
            elif self.method == "hybrid":
                # Combine both HyDE and Multi-Query
                expansions = await self.hybrid_expand(query)
            else:
                # No expansion
                expansions = [query]
                logger.info("  No expansion method - using original query only")

            # Cache result
            self.cache_expansion(query, expansions)

            logger.info(f"✓ Query Expansion COMPLETE - generated {len(expansions)} variations")

            return expansions

        except Exception as e:
            logger.error(f"✗ Query expansion FAILED: {e}")
            # Fallback to original query
            return [query]

    async def hyde_expand(self, query: str) -> list[str]:
        """Generate hypothetical document for query.

        Args:
            query: Original query

        Returns:
            List containing hypothetical answer
        """
        prompt = f"""Please write a passage to answer the question:
Question: {query}
Passage:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
            )

            hypothetical_doc = response.strip()
            logger.info("  HyDE - Generated hypothetical document:")
            logger.info(
                f"    {hypothetical_doc[:200]}{'...' if len(hypothetical_doc) > 200 else ''}"
            )

            # Return as list with single hypothetical document
            return [hypothetical_doc]

        except Exception as e:
            logger.error(f"  HyDE generation FAILED: {e}")
            # Return original query as fallback
            return [query]

    async def multi_query_expand(
        self,
        query: str,
        num_variations: int | None = None,
    ) -> list[str]:
        """Generate multiple query variations.

        Args:
            query: Original query
            num_variations: Number of variations (uses default if None)

        Returns:
            List of query variations including original
        """
        if num_variations is None:
            num_variations = self.num_variations

        prompt = f"""You are an AI assistant helping to generate alternative phrasings of a question.
Generate {num_variations} different ways to ask the following question:
Question: {query}

Alternative 1:
Alternative 2:
Alternative 3:"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=300,
            )

            # Parse variations from response
            variations = self._parse_variations(response)

            # Always include original query
            if query not in variations:
                variations.insert(0, query)

            logger.info(f"  Multi-Query - Generated {len(variations)} variations:")
            for i, var in enumerate(variations[: num_variations + 1], 1):
                logger.info(f"    {i}. {var}")

            return variations[: num_variations + 1]  # +1 for original

        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}")
            return [query]

    async def hybrid_expand(
        self,
        query: str,
        num_variations: int | None = None,
    ) -> list[str]:
        """Combine HyDE and Multi-Query expansion methods.

        Generates both hypothetical document and multiple query variations
        for maximum retrieval coverage.

        Args:
            query: Original query
            num_variations: Number of multi-query variations (uses default if None)

        Returns:
            List combining HyDE hypothetical doc and query variations
        """
        logger.info("  Hybrid Mode - Combining HyDE + Multi-Query")

        # Get HyDE hypothetical document
        hyde_results = await self.hyde_expand(query)
        logger.info(f"  ✓ HyDE: Generated hypothetical document")

        # Get Multi-Query variations
        multi_results = await self.multi_query_expand(query, num_variations)
        logger.info(f"  ✓ Multi-Query: Generated {len(multi_results)} variations")

        # Combine both (HyDE first, then multi-query variations)
        combined = hyde_results + multi_results

        # Deduplicate while preserving order
        seen = set()
        unique_combined = []
        for item in combined:
            if item not in seen:
                seen.add(item)
                unique_combined.append(item)

        logger.info(
            f"  ✓ Hybrid Complete: {len(unique_combined)} total expansions "
            f"({len(hyde_results)} HyDE + {len(multi_results)} Multi-Query)"
        )

        return unique_combined

    def _parse_variations(self, response: str) -> list[str]:
        """Parse query variations from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of parsed variations
        """
        variations = []

        # Split by lines and look for alternatives
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Extract content after "Alternative N:" pattern
            if line.startswith("Alternative"):
                # Find colon and extract content after it
                colon_idx = line.find(":")
                if colon_idx != -1:
                    content = line[colon_idx + 1 :].strip()
                    if content:
                        variations.append(content)
                continue

            # Remove numbering if present
            if line[0].isdigit() and len(line) > 2 and line[1:3] in [". ", ": "]:
                line = line[3:].strip()

            if line:
                variations.append(line)

        return variations

    def get_cached(self, query: str) -> list[str] | None:
        """Get cached expansion if available and not expired.

        Args:
            query: Query to look up

        Returns:
            Cached expansions or None
        """
        query_hash = self._hash_query(query)

        if query_hash in self.cache:
            expansions, timestamp = self.cache[query_hash]

            # Check if expired
            if time.time() - timestamp < self.cache_ttl:
                return expansions
            else:
                # Remove expired entry
                del self.cache[query_hash]

        return None

    def cache_expansion(self, query: str, expansions: list[str]) -> None:
        """Cache query expansion.

        Args:
            query: Original query
            expansions: Expanded queries
        """
        query_hash = self._hash_query(query)
        self.cache[query_hash] = (expansions, time.time())

        # Simple cache size limit
        if len(self.cache) > 1000:
            # Remove oldest entries
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1][1],  # Sort by timestamp
            )
            # Keep newest 800
            self.cache = dict(sorted_items[-800:])

    def _hash_query(self, query: str) -> str:
        """Generate hash for query.

        Args:
            query: Query text

        Returns:
            Query hash
        """
        return hashlib.md5(query.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the expansion cache."""
        self.cache.clear()
        logger.info("Cleared query expansion cache")
