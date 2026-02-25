"""Document processing service orchestrating the full pipeline."""

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.logging_config import get_logger, log_progress
from app.models.chunk import DocumentChunk, EmbeddedChunk
from app.models.document import DocumentInfo, DocumentMetadata, DocumentUploadResponse
from app.models.vlm.vlm_client import VLMClient
from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.chunker import SemanticChunker
from app.processing.multimodal import MultimodalChunker, MultimodalEmbedder
from app.retrieval.bm25_index import BM25Index
from app.storage.file_storage import FileStorageService, FileValidationError
from app.storage.vector_store import VectorStore

logger = get_logger(__name__)


class ProcessingStatus(str, Enum):
    """Document processing status."""

    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentProcessingError(Exception):
    """Raised when document processing fails."""

    pass


class DocumentService:
    """Service orchestrating document processing pipeline.

    This service coordinates the full document processing workflow:
    1. Upload and validate PDF file
    2. Parse PDF to extract text, images, tables
    3. Chunk text into semantic segments
    4. Generate embeddings for chunks
    5. Store embeddings in vector database

    Includes comprehensive error handling, logging, and status tracking.
    """

    def __init__(
        self,
        file_storage: FileStorageService | None = None,
        parser: RAGAnythingParser | None = None,
        chunker: SemanticChunker | None = None,
        multimodal_chunker: MultimodalChunker | None = None,
        multimodal_embedder: MultimodalEmbedder | None = None,
        vlm_client: VLMClient | None = None,
        openai_client: OpenAIClient | None = None,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ):
        """Initialize document service with multimodal support and RAG enhancements.

        Args:
            file_storage: File storage service (creates default if None)
            parser: PDF parser (creates default if None)
            chunker: Text chunker (creates default if None) - legacy
            multimodal_chunker: Multimodal chunker (creates default if None)
            multimodal_embedder: Multimodal embedder (creates default if None)
            vlm_client: VLM client for visual understanding (creates default if None)
            openai_client: OpenAI client (creates default if None)
            vector_store: Vector store (creates default if None)
            bm25_index: BM25 index for keyword search (creates default if None)
        """
        self.settings = get_settings()

        # Initialize components
        self.file_storage = file_storage or FileStorageService()
        self._parser = parser

        # Legacy chunker for backward compatibility
        self.chunker = chunker or SemanticChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # Multimodal components (Lazy initialization)
        self._multimodal_chunker = multimodal_chunker
        self._multimodal_embedder = multimodal_embedder
        self._vlm_client = vlm_client

        # We also need these params to init the lazy components later if needed
        self._chunk_size = self.settings.chunk_size
        self._chunk_overlap = self.settings.chunk_overlap

        self.openai_client = openai_client or OpenAIClient(
            api_key=(
                self.settings.openai_api_key.get_secret_value()
                if self.settings.openai_api_key
                else ""
            ),
            model=self.settings.openai_model,
            embedding_model=self.settings.openai_embedding_model,
        )
        self.vector_store = vector_store or VectorStore(
            persist_directory=self.settings.vector_db_path,
            collection_name=self.settings.text_collection,  # Use text collection as default
        )

        # BM25 index for keyword search
        bm25_persist_path = Path(self.settings.vector_db_path) / "bm25_index.json"
        self.bm25_index = bm25_index or BM25Index(
            persist_path=bm25_persist_path,
            k1=self.settings.bm25_k1,
            b=self.settings.bm25_b,
        )

        # Try to load existing BM25 index
        if self.settings.enable_hybrid_search:
            try:
                self.bm25_index.load()
                logger.info("Loaded existing BM25 index")
            except Exception as e:
                logger.info(f"No existing BM25 index found, will create new one: {e}")

        # In-memory storage for document metadata and status
        # In production, this would be a database
        self._documents: dict[str, dict[str, Any]] = {}
        self._status: dict[str, ProcessingStatus] = {}

        logger.info("DocumentService initialized with RAG enhancements")

    @property
    def multimodal_chunker(self) -> MultimodalChunker:
        """Lazy load multimodal chunker."""
        if self._multimodal_chunker is None:
            logger.info("Initializing MultimodalChunker (Lazy Load)")
            self._multimodal_chunker = MultimodalChunker(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        return self._multimodal_chunker

    @property
    def multimodal_embedder(self) -> MultimodalEmbedder:
        """Lazy load multimodal embedder."""
        if self._multimodal_embedder is None:
            logger.info("Initializing MultimodalEmbedder (Lazy Load)")
            self._multimodal_embedder = MultimodalEmbedder()
        return self._multimodal_embedder

    @property
    def parser(self) -> RAGAnythingParser:
        """Lazy load parser."""
        if self._parser is None:
            logger.info("Initializing RAGAnythingParser (Lazy Load)")
            self._parser = RAGAnythingParser()
        return self._parser

    @property
    def vlm_client(self) -> VLMClient:
        """Lazy load VLM client."""
        if self._vlm_client is None:
            logger.info("Initializing VLMClient (Lazy Load)")
            self._vlm_client = VLMClient()
        return self._vlm_client

    async def process_document(
        self,
        file_content: bytes,
        metadata: DocumentMetadata,
    ) -> DocumentUploadResponse:
        """Process uploaded PDF through complete pipeline.

        This method orchestrates the full document processing workflow:
        1. Upload → validate and save file
        2. Parse → extract text, images, tables
        3. Chunk → split into semantic segments
        4. Embed → generate embeddings
        5. Store → save to vector database

        Args:
            file_content: Binary content of the PDF file
            metadata: Document metadata (filename, tags, etc.)

        Returns:
            DocumentUploadResponse with document ID and status

        Raises:
            FileValidationError: If file validation fails
            DocumentProcessingError: If processing fails at any stage
        """
        doc_id = None
        current_status = ProcessingStatus.UPLOADED

        try:
            # Step 1: Upload and validate file
            logger.info(f"Starting document processing for '{metadata.filename}'")
            current_status = ProcessingStatus.UPLOADED

            file_metadata = await self.file_storage.save_file(
                file_content=file_content,
                filename=metadata.filename,
                content_type=metadata.content_type,
            )
            doc_id = file_metadata.file_id

            # Initialize status tracking
            self._status[doc_id] = current_status
            self._documents[doc_id] = {
                "file_metadata": file_metadata,
                "document_metadata": metadata,
                "created_at": file_metadata.created_at,
                "num_pages": 0,
                "num_chunks": 0,
            }

            logger.info(
                f"Document '{doc_id}' uploaded successfully (size: {file_metadata.file_size} bytes)"
            )

            # Step 2: Parse PDF
            current_status = ProcessingStatus.PARSING
            self._status[doc_id] = current_status
            logger.info(f"Parsing document '{doc_id}'")

            parsed_doc = self.parser.parse_pdf(file_metadata.upload_path)
            self._documents[doc_id]["num_pages"] = parsed_doc.num_pages

            logger.info(
                f"Document '{doc_id}' parsed successfully "
                f"({parsed_doc.num_pages} pages, "
                f"{len(parsed_doc.text_blocks)} text blocks, "
                f"{len(parsed_doc.images)} images, "
                f"{len(parsed_doc.tables)} tables)"
            )

            # Step 3: Chunk document
            current_status = ProcessingStatus.CHUNKING
            self._status[doc_id] = current_status
            logger.info(f"Chunking document '{doc_id}'")

            # Use structure-aware chunking for better context preservation
            chunks = self.chunker.chunk_with_structure(
                parsed_doc=parsed_doc,
                doc_id=doc_id,
            )
            self._documents[doc_id]["num_chunks"] = len(chunks)

            logger.info(f"Document '{doc_id}' chunked into {len(chunks)} segments")

            # Step 4: Generate embeddings
            current_status = ProcessingStatus.EMBEDDING
            self._status[doc_id] = current_status
            logger.info(f"Generating embeddings for document '{doc_id}'")

            embedded_chunks = await self._embed_chunks(chunks)

            logger.info(f"Generated {len(embedded_chunks)} embeddings for document '{doc_id}'")

            # Step 5: Store in vector database
            current_status = ProcessingStatus.STORING
            self._status[doc_id] = current_status
            logger.info(f"Storing embeddings for document '{doc_id}'")

            await self.vector_store.add_embeddings(
                embeddings=embedded_chunks,
                doc_id=doc_id,
            )

            logger.info(f"Document '{doc_id}' stored successfully in vector database")

            # Step 6: Add to BM25 index for hybrid search
            if self.settings.enable_hybrid_search:
                logger.info(f"Adding document '{doc_id}' to BM25 index")

                # Prepare data for BM25
                chunk_ids = [chunk.chunk.chunk_id for chunk in embedded_chunks]
                texts = [chunk.chunk.content for chunk in embedded_chunks]
                metadata_list = [
                    {
                        "doc_id": chunk.chunk.doc_id,
                        "page": chunk.chunk.page,
                        "chunk_index": chunk.chunk.chunk_index,
                        **chunk.chunk.metadata,
                    }
                    for chunk in embedded_chunks
                ]

                # Add to BM25 index
                self.bm25_index.add_documents(chunk_ids, texts, metadata_list)

                # Persist BM25 index
                try:
                    self.bm25_index.save()
                    logger.info("BM25 index saved successfully")
                except Exception as e:
                    logger.warning(f"Failed to save BM25 index: {e}")

                logger.info(f"Document '{doc_id}' added to BM25 index ({len(chunk_ids)} chunks)")

            # Mark as completed
            current_status = ProcessingStatus.COMPLETED
            self._status[doc_id] = current_status

            logger.info(f"Document '{doc_id}' processing completed successfully")

            return DocumentUploadResponse(
                doc_id=doc_id,
                filename=metadata.filename,
                status=ProcessingStatus.COMPLETED.value,
                message="Document processed successfully",
                created_at=file_metadata.created_at,
            )

        except FileValidationError as e:
            # Validation errors are user errors, not system errors
            error_msg = f"File validation failed: {str(e)}"
            logger.warning(error_msg)

            if doc_id:
                self._status[doc_id] = ProcessingStatus.FAILED
                self._log_error(doc_id, current_status, error_msg)

            raise

        except Exception as e:
            # System errors during processing
            error_msg = f"Document processing failed at stage '{current_status.value}': {str(e)}"
            logger.error(
                error_msg,
                exc_info=True,
                extra={
                    "doc_id": doc_id,
                    "stage": current_status.value,
                    "document_filename": metadata.filename,
                },
            )

            if doc_id:
                self._status[doc_id] = ProcessingStatus.FAILED
                self._log_error(doc_id, current_status, error_msg)

                # Attempt cleanup on failure to maintain consistency
                try:
                    await self._cleanup_failed_document(doc_id)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup document '{doc_id}': {cleanup_error}")

            raise DocumentProcessingError(error_msg) from e

    async def _embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        """Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            List of embedded chunks
        """
        if not chunks:
            return []

        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]

        # Log progress for large batches
        if len(texts) > 10:
            log_progress(logger, "Generating embeddings", 0, len(texts), batch_size=len(texts))

        # Generate embeddings in batch
        embeddings = await self.openai_client.embed_batch(texts)

        # Log completion
        if len(texts) > 10:
            log_progress(
                logger, "Generating embeddings", len(texts), len(texts), batch_size=len(texts)
            )

        # Combine chunks with embeddings
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embedding, modality="text")
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        return embedded_chunks

    async def _cleanup_failed_document(self, doc_id: str) -> None:
        """Clean up resources for a failed document processing.

        Ensures transactional consistency by removing partial data.

        Args:
            doc_id: Document identifier
        """
        logger.info(f"Cleaning up failed document '{doc_id}'")

        # Remove from vector store
        try:
            await self.vector_store.delete_document(doc_id)
        except Exception as e:
            logger.error(f"Failed to delete vectors for '{doc_id}': {e}")

        # Remove file from storage
        try:
            self.file_storage.delete_file(doc_id)
        except Exception as e:
            logger.error(f"Failed to delete file for '{doc_id}': {e}")

    def _log_error(
        self,
        doc_id: str,
        stage: ProcessingStatus,
        error_message: str,
    ) -> None:
        """Log error with complete context information.

        Args:
            doc_id: Document identifier
            stage: Processing stage where error occurred
            error_message: Error message
        """
        timestamp = datetime.now(UTC)

        # Store error information
        if doc_id in self._documents:
            if "errors" not in self._documents[doc_id]:
                self._documents[doc_id]["errors"] = []

            self._documents[doc_id]["errors"].append(
                {
                    "timestamp": timestamp,
                    "stage": stage.value,
                    "error": error_message,
                }
            )

        # Log with structured data
        logger.error(
            f"Error processing document '{doc_id}'",
            extra={
                "doc_id": doc_id,
                "stage": stage.value,
                "error": error_message,
                "timestamp": timestamp.isoformat(),
            },
        )

    async def get_document(self, doc_id: str) -> DocumentInfo:
        """Retrieve document information.

        Args:
            doc_id: Document identifier

        Returns:
            DocumentInfo with document details

        Raises:
            ValueError: If document not found
        """
        if doc_id not in self._documents:
            raise ValueError(f"Document '{doc_id}' not found")

        doc_data = self._documents[doc_id]
        file_metadata = doc_data["file_metadata"]
        document_metadata = doc_data["document_metadata"]

        return DocumentInfo(
            doc_id=doc_id,
            filename=file_metadata.filename,
            file_size=file_metadata.file_size,
            num_pages=doc_data["num_pages"],
            num_chunks=doc_data["num_chunks"],
            created_at=doc_data["created_at"],
            tags=document_metadata.tags,
        )

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DocumentInfo]:
        """List all processed documents.

        Args:
            skip: Number of documents to skip (for pagination)
            limit: Maximum number of documents to return

        Returns:
            List of DocumentInfo objects
        """
        # Get all document IDs sorted by creation time (newest first)
        doc_ids = sorted(
            self._documents.keys(),
            key=lambda doc_id: self._documents[doc_id]["created_at"],
            reverse=True,
        )

        # Apply pagination
        paginated_ids = doc_ids[skip : skip + limit]

        # Get document info for each ID
        documents = []
        for doc_id in paginated_ids:
            try:
                doc_info = await self.get_document(doc_id)
                documents.append(doc_info)
            except ValueError:
                # Document was deleted or corrupted, skip it
                logger.warning(f"Skipping invalid document '{doc_id}'")
                continue

        return documents

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document and associated data.

        Performs cascade deletion:
        1. Remove embeddings from vector store
        2. Delete file from storage
        3. Remove metadata from memory

        Args:
            doc_id: Document identifier

        Returns:
            bool: True if successful

        Raises:
            ValueError: If document not found
        """
        if doc_id not in self._documents:
            raise ValueError(f"Document '{doc_id}' not found")

        logger.info(f"Deleting document '{doc_id}'")

        try:
            # Remove from vector store
            await self.vector_store.delete_document(doc_id)
            logger.info(f"Deleted vectors for document '{doc_id}'")

            # Remove from BM25 index
            if self.settings.enable_hybrid_search:
                try:
                    # Get all chunk IDs for this document from BM25 metadata
                    chunk_ids_to_remove = [
                        cid
                        for cid, meta in self.bm25_index.metadata.items()
                        if meta.get("doc_id") == doc_id
                    ]

                    if chunk_ids_to_remove:
                        self.bm25_index.remove_documents(chunk_ids_to_remove)
                        self.bm25_index.save()
                        logger.info(
                            f"Deleted {len(chunk_ids_to_remove)} chunks "
                            f"from BM25 index for document '{doc_id}'"
                        )
                except Exception as e:
                    logger.warning(f"Failed to remove from BM25 index: {e}")

            # Remove file from storage
            self.file_storage.delete_file(doc_id)
            logger.info(f"Deleted file for document '{doc_id}'")

            # Remove from memory
            del self._documents[doc_id]
            if doc_id in self._status:
                del self._status[doc_id]

            logger.info(f"Document '{doc_id}' deleted successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to delete document '{doc_id}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e

    def get_processing_status(self, doc_id: str) -> ProcessingStatus:
        """Get current processing status for a document.

        Args:
            doc_id: Document identifier

        Returns:
            ProcessingStatus enum value

        Raises:
            ValueError: If document not found
        """
        if doc_id not in self._status:
            raise ValueError(f"Document '{doc_id}' not found")

        return self._status[doc_id]

    async def close(self):
        """Close all resources."""
        await self.openai_client.close()
