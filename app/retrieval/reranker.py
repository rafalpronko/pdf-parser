"""Cross-encoder reranker for improving retrieval relevance."""

import logging
from typing import Any

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class SearchResult:
    """Search result with content and score."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        metadata: dict[str, Any] | None = None,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata or {}


class CrossEncoderReranker:
    """Reranks query-document pairs using cross-encoder model.

    Cross-encoders jointly encode query and document to produce
    a relevance score, providing better ranking than bi-encoders.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 32,
        enable_caching: bool = True,
    ):
        """Initialize reranker with specified model.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            batch_size: Batch size for scoring
            enable_caching: Whether to cache reranking scores
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.enable_caching = enable_caching

        # Score cache: (query, content) -> score
        self.score_cache: dict[tuple[str, str], float] = {}

        # Determine device
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Initialize model
        self.model = None
        self._init_model()

        logger.info(
            f"Initialized CrossEncoderReranker: model={model_name}, "
            f"device={self.device}, batch_size={batch_size}"
        )

    def _init_model(self) -> None:
        """Initialize cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None

        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        chunks: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank chunks by relevance to query.

        Args:
            query: Search query
            chunks: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of search results
        """
        if self.model is None:
            logger.warning("⚠ Reranker model not available, returning original order")
            return chunks[:top_k]

        if not chunks:
            return []

        logger.info(f"→ Cross-Encoder Reranking START - {len(chunks)} candidates → top {top_k}")
        logger.info(f"  Query: {query}")

        # Log input chunks with original scores
        logger.info("  Input candidates (before reranking):")
        for i, chunk in enumerate(chunks[:5], 1):
            logger.info(
                f"    {i}. orig_score={chunk.score:.4f} | "
                f"chunk={chunk.chunk_id[:12]}... | "
                f"content={chunk.content[:80]}..."
            )
        if len(chunks) > 5:
            logger.info(f"    ... and {len(chunks) - 5} more candidates")

        # Score all query-chunk pairs
        texts = [chunk.content for chunk in chunks]
        scores = self.score_pairs(query, texts)

        # Normalize scores to [0, 1]
        normalized_scores = self.normalize_scores(scores)

        # Update chunks with reranking scores
        reranked_chunks = []
        for chunk, score in zip(chunks, normalized_scores):
            # Create new result with reranking score
            reranked = SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=score,
                metadata={
                    **chunk.metadata,
                    "original_score": chunk.score,
                    "reranking_score": score,
                },
            )
            reranked_chunks.append(reranked)

        # Sort by reranking score descending
        reranked_chunks.sort(key=lambda x: x.score, reverse=True)

        logger.info("✓ Cross-Encoder Reranking COMPLETE:")
        logger.info("  Output (after reranking):")
        for i, chunk in enumerate(reranked_chunks[:top_k], 1):
            orig_score = chunk.metadata.get("original_score", 0.0)
            rerank_score = chunk.metadata.get("reranking_score", 0.0)
            score_change = (
                "↑" if rerank_score > orig_score else "↓" if rerank_score < orig_score else "="
            )
            logger.info(
                f"    {i}. rerank={rerank_score:.4f} (was {orig_score:.4f} {score_change}) | "
                f"chunk={chunk.chunk_id[:12]}... | "
                f"content={chunk.content[:80]}..."
            )

        return reranked_chunks[:top_k]

    def score_pairs(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """Score query-text pairs in batch.

        Args:
            query: Query text
            texts: List of document texts

        Returns:
            List of relevance scores
        """
        if self.model is None:
            # Return uniform scores if model unavailable
            return [0.5] * len(texts)

        scores = []

        # Check cache first
        if self.enable_caching:
            cached_scores = []
            uncached_indices = []
            uncached_texts = []

            for i, text in enumerate(texts):
                cache_key = (query, text)
                if cache_key in self.score_cache:
                    cached_scores.append((i, self.score_cache[cache_key]))
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            # Score uncached pairs
            if uncached_texts:
                pairs = [[query, text] for text in uncached_texts]
                new_scores = self.model.predict(pairs, batch_size=self.batch_size)

                # Cache new scores
                for text, score in zip(uncached_texts, new_scores):
                    self.score_cache[(query, text)] = float(score)

                # Combine cached and new scores
                all_scores = [0.0] * len(texts)
                for i, score in cached_scores:
                    all_scores[i] = score
                for i, score in zip(uncached_indices, new_scores):
                    all_scores[i] = float(score)

                scores = all_scores
            else:
                # All cached
                scores = [0.0] * len(texts)
                for i, score in cached_scores:
                    scores[i] = score
        else:
            # No caching - score all pairs
            pairs = [[query, text] for text in texts]
            raw_scores = self.model.predict(pairs, batch_size=self.batch_size)
            scores = [float(s) for s in raw_scores]

        return scores

    def normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to 0-1 range.

        Args:
            scores: Raw scores

        Returns:
            Normalized scores in [0, 1]
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return [0.5] * len(scores)

        # Min-max normalization
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]

        return normalized

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self.score_cache.clear()
        logger.info("Cleared reranking score cache")
