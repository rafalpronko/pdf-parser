"""Query service implementing RAG pipeline with enhancements."""

import time

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.logging_config import get_logger
from app.models.query import QueryRequest, QueryResponse, SourceReference
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.query_expansion import QueryExpander
from app.retrieval.reranker import CrossEncoderReranker
from app.services.document_service import DocumentService
from app.storage.vector_store import SearchResult, VectorStore

logger = get_logger(__name__)


class QueryService:
    """Service handling RAG query pipeline with enhancements.

    This service implements the enhanced Retrieval-Augmented Generation workflow:
    1. Query Expansion (optional) - Generate query variations
    2. Embed the user's query
    3. Hybrid Search - Retrieve using vector + keyword search
    4. Reranking (optional) - Rerank results with cross-encoder
    5. Construct prompt with query and retrieved context
    6. Generate answer using LLM
    7. Return response with source citations

    Handles empty results and multi-source synthesis.
    """

    def __init__(
        self,
        openai_client: OpenAIClient | None = None,
        vector_store: VectorStore | None = None,
        document_service: DocumentService | None = None,
        bm25_index: BM25Index | None = None,
        hybrid_search: HybridSearchEngine | None = None,
        query_expander: QueryExpander | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        """Initialize query service with RAG enhancements.

        Args:
            openai_client: OpenAI client for embeddings and generation
            vector_store: Vector store for similarity search
            document_service: Document service for metadata lookup
            bm25_index: BM25 index for keyword search
            hybrid_search: Hybrid search engine
            query_expander: Query expansion component
            reranker: Cross-encoder reranker
        """
        self.settings = get_settings()

        # Initialize core components
        self.openai_client = openai_client or OpenAIClient(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            embedding_model=self.settings.openai_embedding_model,
        )
        self.vector_store = vector_store or VectorStore(
            persist_directory=self.settings.vector_db_path,
            collection_name=self.settings.text_collection,
        )
        self.document_service = document_service

        # Initialize BM25 index if hybrid search is enabled
        if self.settings.enable_hybrid_search:
            from pathlib import Path

            bm25_persist_path = Path(self.settings.vector_db_path) / "bm25_index.pkl"
            self.bm25_index = bm25_index or BM25Index(
                persist_path=bm25_persist_path,
                k1=self.settings.bm25_k1,
                b=self.settings.bm25_b,
            )

            # Try to load existing index
            try:
                self.bm25_index.load()
                logger.info("Loaded BM25 index for hybrid search")
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}")

            # Initialize hybrid search engine
            self.hybrid_search = hybrid_search or HybridSearchEngine(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                vector_weight=self.settings.vector_weight,
                keyword_weight=self.settings.keyword_weight,
            )
        else:
            self.bm25_index = None
            self.hybrid_search = None

        # Initialize query expander if enabled
        if self.settings.enable_query_expansion:
            self.query_expander = query_expander or QueryExpander(
                llm_client=self.openai_client,
                method=self.settings.expansion_method,
                cache_ttl=self.settings.expansion_cache_ttl,
                num_variations=self.settings.num_query_variations,
            )
        else:
            self.query_expander = None

        # Initialize reranker if enabled
        if self.settings.enable_reranking:
            self._reranker = reranker
            # Store params for lazy init
            self._reranker_model = self.settings.reranker_model
            self._reranker_device = "auto" if self.settings.enable_gpu else "cpu"
            self._reranker_batch_size = self.settings.reranking_batch_size
            self._reranker_caching = self.settings.cache_reranking_scores
        else:
            self._reranker = None

        logger.info(
            f"QueryService initialized with enhancements: "
            f"hybrid_search={self.settings.enable_hybrid_search}, "
            f"query_expansion={self.settings.enable_query_expansion}, "
            f"reranking={self.settings.enable_reranking}"
        )

    @property
    def reranker(self) -> CrossEncoderReranker | None:
        """Lazy load reranker."""
        if not self.settings.enable_reranking:
            return None

        if self._reranker is None:
            logger.info("Initializing CrossEncoderReranker (Lazy Load)")
            self._reranker = CrossEncoderReranker(
                model_name=self._reranker_model,
                device=self._reranker_device,
                batch_size=self._reranker_batch_size,
                enable_caching=self._reranker_caching,
            )
        return self._reranker

    async def query(
        self,
        request: QueryRequest,
    ) -> QueryResponse:
        """Execute enhanced RAG query pipeline.

        Args:
            request: Query request with question and parameters

        Returns:
            QueryResponse with answer and source citations

        Raises:
            Exception: If query processing fails
        """
        start_time = time.time()

        try:
            logger.info("=" * 80)
            logger.info("🔍 RAG PIPELINE START")
            logger.info("=" * 80)
            logger.info(f"Query: '{request.question}'")
            logger.info(f"Parameters: top_k={request.top_k}, temperature={request.temperature}")
            logger.info(
                f"Enhancements: hybrid={self.settings.enable_hybrid_search}, "
                f"expansion={self.settings.enable_query_expansion}, "
                f"reranking={self.settings.enable_reranking}"
            )
            logger.info("-" * 80)

            # Step 1: Query Expansion (optional)
            queries = [request.question]
            if self.query_expander:
                queries = await self.query_expander.expand(request.question)
                # Logging is now in QueryExpander
            else:
                logger.info("→ Query Expansion DISABLED")

            # Step 2: Generate query embeddings for all variations
            logger.info(f"→ Generating embeddings for {len(queries)} query variation(s)")
            query_embeddings = []
            for i, query in enumerate(queries, 1):
                embedding = await self._embed_query(query)
                query_embeddings.append(embedding)
                logger.info(f"  {i}. Embedded: {query[:80]}...")
            logger.info(
                f"✓ Generated {len(query_embeddings)} embeddings (dim={len(query_embeddings[0]) if query_embeddings else 0})"
            )

            # Step 3: Retrieve relevant context using hybrid search or vector search
            logger.info("-" * 80)
            all_results = []
            retrieval_k = self.settings.reranking_top_k if self.reranker else request.top_k

            if self.hybrid_search and self.settings.enable_hybrid_search:
                # Use hybrid search (vector + keyword)
                logger.info(f"→ Retrieval Mode: HYBRID SEARCH (top_k={retrieval_k})")

                # For query expansion, we search with each variation and combine
                for i, (query, embedding) in enumerate(zip(queries, query_embeddings), 1):
                    logger.info(f"  Searching with variation {i}/{len(queries)}")
                    results = await self.hybrid_search.search(
                        query=query,
                        query_embedding=embedding,
                        top_k=retrieval_k,
                    )
                    all_results.extend(results)
            else:
                # Use standard vector search
                logger.info(f"→ Retrieval Mode: VECTOR SEARCH ONLY (top_k={retrieval_k})")

                for i, embedding in enumerate(query_embeddings, 1):
                    logger.info(f"  Searching with embedding {i}/{len(query_embeddings)}")
                    results = await self.retrieve_context(
                        query_embedding=embedding,
                        top_k=retrieval_k,
                    )
                    all_results.extend(results)
                    logger.info(f"    Found {len(results)} results")

            # Deduplicate results by chunk_id
            seen_chunks = set()
            unique_results = []
            for result in all_results:
                if result.chunk_id not in seen_chunks:
                    seen_chunks.add(result.chunk_id)
                    unique_results.append(result)

            search_results = unique_results
            logger.info(
                f"→ Deduplication: {len(all_results)} total → {len(search_results)} unique chunks"
            )
            logger.info("-" * 80)

            # Step 4: Reranking (optional)
            if self.reranker and search_results:
                # Convert to reranker format
                from app.retrieval.reranker import SearchResult as RerankerSearchResult

                reranker_results = [
                    RerankerSearchResult(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        score=r.relevance_score,
                        metadata={
                            "doc_id": r.doc_id,
                            "page": r.page,
                            "chunk_index": r.chunk_index,
                        },
                    )
                    for r in search_results
                ]

                # Rerank for EACH query variant if query expansion was used
                rerank_k = max(request.top_k, self.settings.final_top_k)

                if len(queries) > 1:
                    # Multiple queries from expansion - rerank for each and use RRF fusion
                    logger.info(f"→ Multi-Query Reranking: Rerankując dla {len(queries)} wariantów")

                    all_reranked = []
                    for i, query in enumerate(queries, 1):
                        reranked = self.reranker.rerank(
                            query=query,
                            chunks=reranker_results,
                            top_k=rerank_k * 2,  # Get more results for fusion
                        )
                        all_reranked.append(reranked)
                        logger.info(f"  {i}. Reranked dla wariantu {i}: {len(reranked)} chunków")

                    # Use Reciprocal Rank Fusion to combine rankings
                    logger.info("→ Reciprocal Rank Fusion: Łączenie wyników")
                    reranked = self._reciprocal_rank_fusion(all_reranked, k=60)[:rerank_k]
                    logger.info(f"  RRF zwrócił {len(reranked)} chunków")

                else:
                    # Single query - standard reranking
                    logger.info(f"→ Standard Reranking: Rerankując dla oryginalnego pytania")
                    reranked = self.reranker.rerank(
                        query=request.question,
                        chunks=reranker_results,
                        top_k=rerank_k,
                    )

                # Convert back to SearchResult format
                search_results = [
                    SearchResult(
                        chunk_id=r.chunk_id,
                        doc_id=r.metadata.get("doc_id", ""),
                        content=r.content,
                        page=r.metadata.get("page", 0),
                        chunk_index=r.metadata.get("chunk_index", 0),
                        metadata=r.metadata,
                        relevance_score=r.score,
                    )
                    for r in reranked
                ]

            else:
                # Just take top_k if no reranking
                logger.info(f"→ Reranking DISABLED - taking top {request.top_k} results")
                search_results = search_results[: request.top_k]

            logger.info("-" * 80)

            # Step 5: Handle empty results
            if not search_results:
                logger.warning("⚠ No relevant documents found for query")
                processing_time = time.time() - start_time
                logger.info("=" * 80)
                logger.info(f"✗ RAG PIPELINE COMPLETE (no results) - {processing_time:.2f}s")
                logger.info("=" * 80)

                return QueryResponse(
                    answer="I couldn't find any relevant information in the documents to answer your question.",
                    sources=[],
                    processing_time=processing_time,
                )

            # Step 6: Generate response with context
            logger.info("→ LLM Generation START")
            logger.info(f"  Context: {len(search_results)} chunks")
            total_context_chars = sum(len(r.content) for r in search_results)
            logger.info(f"  Total context size: {total_context_chars} characters")

            answer = await self.generate_response(
                question=request.question,
                context=search_results,
                temperature=request.temperature,
            )

            logger.info("✓ LLM Generation COMPLETE")
            logger.info(f"  Answer preview: {answer[:200]}...")

            # Step 7: Build source references
            sources = []
            if request.include_sources:
                sources = await self._build_source_references(search_results)

            processing_time = time.time() - start_time

            logger.info("-" * 80)
            logger.info("=" * 80)
            logger.info(f"✓ RAG PIPELINE COMPLETE - {processing_time:.2f}s")
            logger.info(f"  Sources: {len(sources)}")
            logger.info(f"  Answer length: {len(answer)} characters")
            logger.info("=" * 80)

            return QueryResponse(
                answer=answer,
                sources=sources,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Query processing failed: {str(e)}"
            logger.error(
                error_msg,
                exc_info=True,
                extra={
                    "question": request.question[:100],
                    "processing_time": processing_time,
                },
            )
            raise

    async def _embed_query(self, question: str) -> list[float]:
        """Generate embedding for query.

        Args:
            question: User's question

        Returns:
            Query embedding vector
        """
        logger.debug(f"Generating embedding for query: '{question[:50]}...'")
        embedding = await self.openai_client.embed_text(question)
        logger.debug(f"Generated embedding with dimension {len(embedding)}")
        return embedding

    async def retrieve_context(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks from vector store.

        Results are ordered by descending relevance score.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve

        Returns:
            List of search results ordered by relevance (descending)
        """
        logger.debug(f"Retrieving top {top_k} relevant chunks")

        search_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        logger.debug(
            f"Retrieved {len(search_results)} chunks "
            f"(scores: {[r.relevance_score for r in search_results]})"
        )

        return search_results

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list],
        k: int = 60,
    ) -> list:
        """Combine multiple ranked lists using Reciprocal Rank Fusion.

        RRF score for document d:
            RRF(d) = sum over all rankings of: 1 / (k + rank(d))

        where k is a constant (typically 60) to reduce impact of high ranks.

        Args:
            ranked_lists: List of ranked result lists (each from different query variant)
            k: Constant for RRF formula (default 60)

        Returns:
            Combined list sorted by RRF score (descending)
        """
        # Calculate RRF scores for all documents
        rrf_scores: dict[str, float] = {}
        doc_metadata: dict[str, any] = {}  # Store first occurrence of each doc

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc.chunk_id

                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)

                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += rrf_score
                else:
                    rrf_scores[doc_id] = rrf_score
                    doc_metadata[doc_id] = doc  # Keep first occurrence

        # Sort by RRF score (descending) and return documents
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )

        # Reconstruct result list with RRF scores
        result = []
        for doc_id in sorted_doc_ids:
            doc = doc_metadata[doc_id]
            # Update score to RRF score
            doc.score = rrf_scores[doc_id]
            result.append(doc)

        logger.info(f"  RRF combined {len(ranked_lists)} rankings into {len(result)} unique docs")

        return result

    async def generate_response(
        self,
        question: str,
        context: list[SearchResult],
        temperature: float,
    ) -> str:
        """Generate answer using LLM with retrieved context.

        Constructs a prompt containing both the question and all
        retrieved context, then generates an answer using the LLM.

        Args:
            question: User's question
            context: Retrieved search results
            temperature: Sampling temperature for generation

        Returns:
            Generated answer string
        """
        logger.debug(f"Generating response with {len(context)} context chunks")

        # Build context string from search results
        context_str = self._build_context_string(context)

        # Generate answer using OpenAI
        answer = await self.openai_client.generate_with_context(
            question=question,
            context=context_str,
            temperature=temperature,
        )

        logger.debug(f"Generated answer: '{answer[:100]}...'")

        return answer

    def _build_context_string(self, search_results: list[SearchResult]) -> str:
        """Build context string from search results.

        Formats retrieved chunks into a structured context string
        for the LLM prompt.

        Args:
            search_results: List of search results

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            # Format: [Source N] (Page X, Score: Y.YY)
            # Content...
            context_parts.append(
                f"[Source {i}] (Page {result.page}, "
                f"Relevance: {result.relevance_score:.2f})\n"
                f"{result.content}"
            )

        return "\n\n".join(context_parts)

    async def _build_source_references(
        self,
        search_results: list[SearchResult],
    ) -> list[SourceReference]:
        """Build source references from search results.

        Looks up document metadata to include filename in citations.

        Args:
            search_results: List of search results

        Returns:
            List of source references with metadata
        """
        sources = []

        for result in search_results:
            # Get filename from document service if available
            filename = result.doc_id  # Default to doc_id

            if self.document_service:
                try:
                    doc_info = await self.document_service.get_document(result.doc_id)
                    filename = doc_info.filename
                except Exception as e:
                    logger.warning(f"Could not retrieve filename for doc '{result.doc_id}': {e}")

            sources.append(
                SourceReference(
                    doc_id=result.doc_id,
                    filename=filename,
                    page=result.page,
                    chunk_content=result.content,
                    modality="text",  # Default to text modality
                    relevance_score=result.relevance_score,
                )
            )

        return sources

    async def close(self):
        """Close all resources."""
        await self.openai_client.close()
