"""Vector database storage using ChromaDB."""

import logging
from typing import Any

from app.models.chunk import EmbeddedChunk

logger = logging.getLogger(__name__)


class SearchResult:
    """Search result from vector database."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        page: int,
        chunk_index: int,
        metadata: dict[str, Any],
        relevance_score: float,
    ):
        """Initialize search result.

        Args:
            chunk_id: Unique chunk identifier
            doc_id: Document identifier
            content: Chunk content
            page: Page number
            chunk_index: Chunk index in document
            metadata: Additional metadata
            relevance_score: Relevance score (0-1, higher is more relevant)
        """
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content
        self.page = page
        self.chunk_index = chunk_index
        self.metadata = metadata
        self.relevance_score = relevance_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
        }


class NoOpEmbeddingFunction:
    """No-op embedding function to prevent Chroma from downloading default model."""

    def __call__(self, input: Any) -> Any:
        return []

    def name(self) -> str:
        """Return the name of this embedding function."""
        return "noop"


class VectorStore:
    """Vector database storage using ChromaDB.

    Manages embedding storage, retrieval, and deletion operations.
    Supports idempotent reprocessing and transactional consistency.
    """

    def __init__(
        self,
        persist_directory: str = "./data/vectordb",
        collection_name: str = "documents",
    ):
        """Initialize vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        # We handle embeddings manually (via OpenAI), so we use a no-op function
        # to prevent Chroma from downloading the default SentenceTransformer model.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            embedding_function=NoOpEmbeddingFunction(),
        )

        logger.info(
            f"Initialized VectorStore with collection '{collection_name}' at '{persist_directory}'"
        )

    async def add_embeddings(
        self,
        embeddings: list[EmbeddedChunk],
        doc_id: str,
    ) -> bool:
        """Store embeddings with metadata.

        Implements idempotent reprocessing: if chunks for this document
        already exist, they are replaced rather than duplicated.

        Args:
            embeddings: List of embedded chunks to store
            doc_id: Document identifier

        Returns:
            bool: True if successful

        Raises:
            ValueError: If embeddings list is empty
            Exception: If storage operation fails
        """
        if not embeddings:
            raise ValueError("Cannot add empty embeddings list")

        try:
            # First, delete any existing chunks for this document
            # This ensures idempotent reprocessing
            await self.delete_document(doc_id)

            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embedding_vectors = []

            for embedded_chunk in embeddings:
                chunk = embedded_chunk.chunk
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)

                # Prepare metadata - ChromaDB only accepts str, int, float, bool (NOT None)
                metadata = {
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                }

                # Add other metadata, converting complex types to strings
                # IMPORTANT: ChromaDB does NOT accept None values
                for key, value in chunk.metadata.items():
                    if value is None:
                        # Skip None values - ChromaDB doesn't accept them
                        continue
                    elif isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, list):
                        # Convert lists to JSON string or count
                        if key == "associated_images":
                            metadata["num_associated_images"] = len(value)
                        else:
                            metadata[key] = str(value)
                    else:
                        # Convert other types to string
                        metadata[key] = str(value)

                metadatas.append(metadata)
                embedding_vectors.append(embedded_chunk.embedding)

            # Add to collection
            self._collection.add(
                ids=ids,
                embeddings=embedding_vectors,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(embeddings)} embeddings for document '{doc_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to add embeddings for document '{doc_id}': {e}")
            raise

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[SearchResult]:
        """Perform similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            doc_id: Optional document ID to filter results

        Returns:
            list[SearchResult]: Search results ordered by relevance (descending)

        Raises:
            ValueError: If query_embedding is empty or top_k is invalid
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        try:
            # Build where clause for filtering
            where = {"doc_id": doc_id} if doc_id else None

            # Query the collection
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )

            # Parse results
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    content = results["documents"][0][i]
                    distance = results["distances"][0][i]

                    # Convert cosine distance to similarity score (0-1, higher is better)
                    # ChromaDB cosine distance = 1 - cosine_similarity
                    relevance_score = max(0.0, min(1.0, 1.0 - distance))

                    search_results.append(
                        SearchResult(
                            chunk_id=chunk_id,
                            doc_id=metadata["doc_id"],
                            content=content,
                            page=metadata["page"],
                            chunk_index=metadata["chunk_index"],
                            metadata={
                                k: v
                                for k, v in metadata.items()
                                if k not in ["doc_id", "page", "chunk_index"]
                            },
                            relevance_score=relevance_score,
                        )
                    )

            logger.info(f"Search returned {len(search_results)} results (requested top_k={top_k})")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks for a document.

        Implements cascade deletion: all chunks associated with the
        document are removed.

        Args:
            doc_id: Document identifier

        Returns:
            bool: True if successful (even if no chunks were found)
        """
        try:
            # Query for all chunks with this doc_id
            results = self._collection.get(
                where={"doc_id": doc_id},
            )

            if results["ids"]:
                # Delete all matching chunks
                self._collection.delete(
                    ids=results["ids"],
                )
                logger.info(f"Deleted {len(results['ids'])} chunks for document '{doc_id}'")
            else:
                logger.info(
                    f"No chunks found for document '{doc_id}' (already deleted or never existed)"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to delete document '{doc_id}': {e}")
            raise

    async def get_chunk(self, chunk_id: str) -> SearchResult | None:
        """Retrieve a specific chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            SearchResult | None: The chunk if found, None otherwise
        """
        try:
            results = self._collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"],
            )

            if results["ids"]:
                metadata = results["metadatas"][0]
                content = results["documents"][0]

                return SearchResult(
                    chunk_id=chunk_id,
                    doc_id=metadata["doc_id"],
                    content=content,
                    page=metadata["page"],
                    chunk_index=metadata["chunk_index"],
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ["doc_id", "page", "chunk_index"]
                    },
                    relevance_score=1.0,  # Not from search, so max score
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get chunk '{chunk_id}': {e}")
            raise

    async def count_chunks(self, doc_id: str | None = None) -> int:
        """Count chunks in the collection.

        Args:
            doc_id: Optional document ID to filter by

        Returns:
            int: Number of chunks
        """
        try:
            if doc_id:
                results = self._collection.get(
                    where={"doc_id": doc_id},
                )
                return len(results["ids"])
            else:
                return self._collection.count()

        except Exception as e:
            logger.error(f"Failed to count chunks: {e}")
            raise

    def reset(self) -> None:
        """Reset the collection (delete all data).

        WARNING: This is destructive and should only be used for testing.
        """
        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.warning(f"Reset collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
