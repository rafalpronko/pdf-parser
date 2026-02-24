"""BM25 keyword search index for document chunks."""

import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 keyword search index for document chunks.

    Provides efficient keyword-based search using the BM25 algorithm.
    Supports adding, removing, and searching documents with metadata.
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25 index.

        Args:
            persist_path: Path to persist index to disk
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.persist_path = persist_path
        self.k1 = k1
        self.b = b

        # Index data structures
        self.doc_ids: list[str] = []
        self.texts: list[str] = []
        self.metadata: dict[str, dict[str, Any]] = {}
        self.bm25: BM25Okapi | None = None

        # Tokenized corpus for BM25
        self.tokenized_corpus: list[list[str]] = []

        logger.info(f"Initialized BM25Index with k1={k1}, b={b}")

    def add_documents(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add documents to BM25 index.

        Args:
            doc_ids: List of document IDs
            texts: List of document texts
            metadata: List of metadata dicts for each document
        """
        if len(doc_ids) != len(texts) or len(doc_ids) != len(metadata):
            raise ValueError("doc_ids, texts, and metadata must have same length")

        # Add to index
        for doc_id, text, meta in zip(doc_ids, texts, metadata):
            self.doc_ids.append(doc_id)
            self.texts.append(text)
            self.metadata[doc_id] = meta

            # Tokenize text
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        self._rebuild_bm25()

        logger.info(f"Added {len(doc_ids)} documents to BM25 index")

    def remove_documents(self, doc_ids: list[str]) -> None:
        """Remove documents from index.

        Args:
            doc_ids: List of document IDs to remove
        """
        removed_count = 0

        # Find indices to remove
        indices_to_remove = []
        for i, doc_id in enumerate(self.doc_ids):
            if doc_id in doc_ids:
                indices_to_remove.append(i)
                if doc_id in self.metadata:
                    del self.metadata[doc_id]
                removed_count += 1

        # Remove in reverse order to maintain indices
        for i in sorted(indices_to_remove, reverse=True):
            del self.doc_ids[i]
            del self.texts[i]
            del self.tokenized_corpus[i]

        # Rebuild BM25 index if we removed anything
        if removed_count > 0:
            self._rebuild_bm25()
            logger.info(f"Removed {removed_count} documents from BM25 index")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search index and return top-k results.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if self.bm25 is None or len(self.doc_ids) == 0:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Return (doc_id, score) pairs
        results = [
            (self.doc_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # Only return docs with non-zero scores
        ]

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Simple whitespace tokenization with lowercasing.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return text.lower().split()

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from tokenized corpus."""
        if len(self.tokenized_corpus) > 0:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
            logger.debug(f"Rebuilt BM25 index with {len(self.tokenized_corpus)} documents")
        else:
            self.bm25 = None
            logger.debug("BM25 index is empty")

    def save(self) -> None:
        """Persist index to disk.

        Saves all index data structures to disk for later loading.
        """
        if self.persist_path is None:
            logger.warning("No persist_path configured, skipping save")
            return

        # Create parent directory if needed
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Save index data
        index_data = {
            "doc_ids": self.doc_ids,
            "texts": self.texts,
            "metadata": self.metadata,
            "tokenized_corpus": self.tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
        }

        with open(self.persist_path, "wb") as f:
            pickle.dump(index_data, f)

        logger.info(f"Saved BM25 index to {self.persist_path}")

    def load(self) -> None:
        """Load index from disk.

        Loads previously saved index data and rebuilds BM25 index.
        """
        if self.persist_path is None:
            logger.warning("No persist_path configured, skipping load")
            return

        if not self.persist_path.exists():
            logger.info(f"No saved index found at {self.persist_path}")
            return

        try:
            with open(self.persist_path, "rb") as f:
                index_data = pickle.load(f)

            # Restore index data
            self.doc_ids = index_data["doc_ids"]
            self.texts = index_data["texts"]
            self.metadata = index_data["metadata"]
            self.tokenized_corpus = index_data["tokenized_corpus"]
            self.k1 = index_data.get("k1", self.k1)
            self.b = index_data.get("b", self.b)

            # Rebuild BM25 index
            self._rebuild_bm25()

            logger.info(
                f"Loaded BM25 index from {self.persist_path} with {len(self.doc_ids)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            # Reset to empty state
            self.doc_ids = []
            self.texts = []
            self.metadata = {}
            self.tokenized_corpus = []
            self.bm25 = None

    def rebuild_from_chunks(self, chunks: list[Any]) -> None:
        """Rebuild index from document chunks.

        Useful for recovering from corruption or initializing from existing data.

        Args:
            chunks: List of DocumentChunk/TextChunk objects
        """
        logger.info(f"Rebuilding BM25 index from {len(chunks)} chunks")

        # Clear existing data
        self.doc_ids = []
        self.texts = []
        self.metadata = {}
        self.tokenized_corpus = []

        # Extract data from chunks
        doc_ids = []
        texts = []
        metadata_list = []

        for chunk in chunks:
            # Handle both TextChunk and DocumentChunk
            doc_ids.append(chunk.chunk_id)
            texts.append(chunk.content)
            metadata_list.append(
                {
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }
            )

        # Add documents
        if doc_ids:
            self.add_documents(doc_ids, texts, metadata_list)
            logger.info(f"Rebuilt BM25 index with {len(doc_ids)} chunks")
        else:
            logger.warning("No chunks provided for rebuild")

    def detect_corruption(self) -> bool:
        """Detect if index is corrupted.

        Returns:
            True if corruption detected, False otherwise
        """
        try:
            # Check basic consistency
            if len(self.doc_ids) != len(self.texts):
                logger.warning("Corruption detected: doc_ids and texts length mismatch")
                return True

            if len(self.doc_ids) != len(self.tokenized_corpus):
                logger.warning("Corruption detected: doc_ids and tokenized_corpus length mismatch")
                return True

            # Check if BM25 index exists when it should
            if len(self.doc_ids) > 0 and self.bm25 is None:
                logger.warning("Corruption detected: documents exist but BM25 index is None")
                return True

            # Try a test search if index exists
            if self.bm25 is not None:
                try:
                    self.search("test", top_k=1)
                except Exception as e:
                    logger.warning(f"Corruption detected: search failed with {e}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error during corruption detection: {e}")
            return True
