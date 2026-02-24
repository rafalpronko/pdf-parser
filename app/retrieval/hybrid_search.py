"""Hybrid search combining vector and keyword search with RRF fusion."""

import logging
from collections import defaultdict

from app.retrieval.bm25_index import BM25Index
from app.storage.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search combining vector and BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from
    both vector similarity and keyword-based search.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """Initialize hybrid search engine.

        Args:
            vector_store: Vector database for semantic search
            bm25_index: BM25 index for keyword search
            vector_weight: Weight for vector search scores
            keyword_weight: Weight for keyword search scores
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        logger.info(
            f"Initialized HybridSearchEngine: "
            f"vector_weight={vector_weight}, keyword_weight={keyword_weight}"
        )

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        modality_filter: str | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search with RRF fusion.

        Args:
            query: Query text for keyword search
            query_embedding: Query embedding for vector search
            top_k: Number of results to return
            modality_filter: Optional modality filter

        Returns:
            List of search results sorted by fused score
        """
        logger.info(f"→ Hybrid Search START - requesting top_{top_k}")
        logger.info(f"  Query: {query}")

        # Perform both searches in parallel (conceptually)
        vector_results = await self.vector_search(query_embedding, top_k * 2)
        keyword_results = await self.keyword_search(query, top_k * 2)

        # Fuse results using RRF
        fused_results = self.reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            k=60,  # Standard RRF constant
        )

        logger.info(
            f"✓ Hybrid Search COMPLETE: {len(vector_results)} vector + "
            f"{len(keyword_results)} keyword → {len(fused_results)} fused → top {top_k}"
        )

        return fused_results[:top_k]

    async def vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results from vector store
        """
        try:
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            logger.info(f"  Vector Search - found {len(results)} results:")
            for i, res in enumerate(results[:5], 1):  # Log top 5
                logger.info(
                    f"    {i}. score={res.relevance_score:.4f} | "
                    f"doc={res.doc_id[:8]}... | chunk={res.chunk_index} | "
                    f"content={res.content[:80]}..."
                )
            if len(results) > 5:
                logger.info(f"    ... and {len(results) - 5} more results")
            return results

        except Exception as e:
            logger.error(f"  Vector search FAILED: {e}")
            return []

    async def keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Perform BM25 keyword search.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of search results from BM25 index
        """
        try:
            # Get BM25 results (chunk_id, score)
            bm25_results = self.bm25_index.search(query, top_k=top_k)

            # Convert to SearchResult objects
            search_results = []
            for chunk_id, score in bm25_results:
                # Get metadata from BM25 index
                if chunk_id in self.bm25_index.metadata:
                    meta = self.bm25_index.metadata[chunk_id]

                    # Find content
                    idx = self.bm25_index.doc_ids.index(chunk_id)
                    content = self.bm25_index.texts[idx]

                    result = SearchResult(
                        chunk_id=chunk_id,
                        doc_id=meta.get("doc_id", ""),
                        content=content,
                        page=meta.get("page", 0),
                        chunk_index=meta.get("chunk_index", 0),
                        metadata=meta,
                        relevance_score=score,
                    )
                    search_results.append(result)

            logger.info(f"  BM25 Keyword Search - found {len(search_results)} results:")
            for i, res in enumerate(search_results[:5], 1):  # Log top 5
                logger.info(
                    f"    {i}. score={res.relevance_score:.4f} | "
                    f"doc={res.doc_id[:8]}... | chunk={res.chunk_index} | "
                    f"content={res.content[:80]}..."
                )
            if len(search_results) > 5:
                logger.info(f"    ... and {len(search_results) - 5} more results")
            return search_results

        except Exception as e:
            logger.error(f"  Keyword search FAILED: {e}")
            return []

    def reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """Fuse rankings using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank_i)) for each ranking

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            k: RRF constant (default 60)

        Returns:
            Fused and sorted results
        """
        # Calculate RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, SearchResult] = {}

        # Add vector search rankings
        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] += self.vector_weight / (k + rank + 1)
            chunk_map[result.chunk_id] = result

        # Add keyword search rankings
        for rank, result in enumerate(keyword_results):
            rrf_scores[result.chunk_id] += self.keyword_weight / (k + rank + 1)
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result

        # Create fused results with updated scores
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            result = chunk_map[chunk_id]

            # Create new result with fused score
            fused_result = SearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                content=result.content,
                page=result.page,
                chunk_index=result.chunk_index,
                metadata={
                    **result.metadata,
                    "original_score": result.relevance_score,
                    "rrf_score": rrf_score,
                },
                relevance_score=rrf_score,
            )
            fused_results.append(fused_result)

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"  RRF Fusion - combined to {len(fused_results)} unique results:")
        for i, res in enumerate(fused_results[:5], 1):  # Log top 5
            orig_score = res.metadata.get("original_score", 0.0)
            rrf_score = res.metadata.get("rrf_score", 0.0)
            logger.info(
                f"    {i}. RRF={rrf_score:.4f} (orig={orig_score:.4f}) | "
                f"doc={res.doc_id[:8]}... | chunk={res.chunk_index} | "
                f"content={res.content[:80]}..."
            )
        if len(fused_results) > 5:
            logger.info(f"    ... and {len(fused_results) - 5} more results")

        return fused_results
