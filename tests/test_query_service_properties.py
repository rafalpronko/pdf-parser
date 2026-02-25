"""Property-based tests for query service."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.models.query import QueryRequest, QueryResponse, SourceReference
from app.models.search import SearchResult
from app.services.query_service import QueryService


# Strategy for generating valid query requests
@st.composite
def query_request_strategy(draw):
    """Generate valid query requests."""
    question = draw(
        st.text(
            min_size=1, max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        )
    )
    top_k = draw(st.integers(min_value=1, max_value=20))
    temperature = draw(
        st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    include_sources = draw(st.booleans())

    return QueryRequest(
        question=question,
        top_k=top_k,
        temperature=temperature,
        include_sources=include_sources,
    )


# Strategy for generating search results
@st.composite
def search_result_strategy(draw, doc_id=None, min_score=0.0, max_score=1.0):
    """Generate valid search results."""
    if doc_id is None:
        doc_id = draw(
            st.text(
                min_size=1, max_size=50, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
            )
        )

    chunk_id = draw(
        st.text(
            min_size=1, max_size=50, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
        )
    )
    content = draw(
        st.text(
            min_size=1, max_size=500, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        )
    )
    page = draw(st.integers(min_value=0, max_value=1000))
    chunk_index = draw(st.integers(min_value=0, max_value=1000))
    relevance_score = draw(
        st.floats(min_value=min_score, max_value=max_score, allow_nan=False, allow_infinity=False)
    )

    return SearchResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        page=page,
        chunk_index=chunk_index,
        metadata={},
        score=relevance_score,
    )


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    client = MagicMock()
    client.embed_text = AsyncMock(return_value=[0.1] * 384)
    client.generate_with_context = AsyncMock(return_value="Generated answer")
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_document_service():
    """Create mock document service."""
    service = MagicMock()
    service.get_document = AsyncMock()
    return service


@pytest.fixture
def query_service(mock_openai_client, mock_vector_store, mock_document_service, monkeypatch):
    """Create query service with mocked dependencies."""
    # Mock settings to avoid requiring environment variables
    from app.config import Settings

    mock_settings = MagicMock(spec=Settings)
    mock_settings.openai_api_key = MagicMock()
    mock_settings.openai_api_key.get_secret_value.return_value = (
        "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
    )
    mock_settings.openai_model = "gpt-4o-mini"
    mock_settings.openai_embedding_model = "text-embedding-3-small"
    mock_settings.vector_db_path = "./data/vectordb"
    mock_settings.text_collection = "text_chunks"
    mock_settings.collection_name = "documents"
    mock_settings.enable_hybrid_search = False
    mock_settings.enable_query_expansion = False
    mock_settings.enable_reranking = False
    mock_settings.reranking_top_k = 40
    mock_settings.final_top_k = 10

    monkeypatch.setattr("app.services.query_service.get_settings", lambda: mock_settings)

    return QueryService(
        openai_client=mock_openai_client,
        vector_store=mock_vector_store,
        document_service=mock_document_service,
    )


class TestRetrievalRelevanceOrdering:
    """Property-based tests for Property 17: Retrieval relevance ordering."""

    @pytest.mark.asyncio
    @given(
        num_results=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_17_retrieval_relevance_ordering(
        self, query_service, mock_vector_store, num_results
    ):
        """Feature: pdf-rag-system, Property 17: Retrieval relevance ordering.

        For any query submitted to the system, retrieved chunks should be ordered
        by descending relevance score, and all scores should be between 0 and 1.

        Validates: Requirements 5.1
        """
        # Generate search results with random scores
        search_results = []
        for i in range(num_results):
            # Generate random score between 0 and 1
            score = (i * 0.05) % 1.0  # Vary scores
            result = SearchResult(
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i % 3}",  # Multiple docs
                content=f"Content {i}",
                page=i,
                chunk_index=i,
                metadata={},
                score=score,
            )
            search_results.append(result)

        # Sort by descending relevance (as vector store should do)
        search_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Mock vector store to return these results
        mock_vector_store.search.return_value = search_results

        # Execute query
        query_embedding = [0.1] * 384
        results = await query_service.retrieve_context(
            query_embedding=query_embedding,
            top_k=num_results,
        )

        # Property: All scores should be between 0 and 1
        for result in results:
            assert 0.0 <= result.relevance_score <= 1.0, (
                f"Relevance score {result.relevance_score} is out of range [0, 1]"
            )

        # Property: Results should be ordered by descending relevance
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score, (
                f"Results not ordered by relevance: {results[i].relevance_score} < {results[i + 1].relevance_score}"
            )

    @pytest.mark.asyncio
    @given(
        num_results=st.integers(min_value=2, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_17_scores_in_valid_range(
        self, query_service, mock_vector_store, num_results
    ):
        """Feature: pdf-rag-system, Property 17: Retrieval relevance ordering.

        For any retrieved results, all relevance scores must be in the valid range [0, 1].

        Validates: Requirements 5.1
        """
        # Generate results with scores at boundaries and in between
        search_results = []
        for i in range(num_results):
            # Create scores that test boundaries
            if i == 0:
                score = 1.0  # Maximum
            elif i == 1:
                score = 0.0  # Minimum
            else:
                score = (num_results - i) / num_results  # Descending

            result = SearchResult(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=f"Content {i}",
                page=i,
                chunk_index=i,
                metadata={},
                score=score,
            )
            search_results.append(result)

        mock_vector_store.search.return_value = search_results

        # Execute query
        query_embedding = [0.1] * 384
        results = await query_service.retrieve_context(
            query_embedding=query_embedding,
            top_k=num_results,
        )

        # Property: All scores must be in [0, 1]
        for result in results:
            assert 0.0 <= result.relevance_score <= 1.0, (
                f"Score {result.relevance_score} violates range constraint [0, 1]"
            )


class TestPromptContainsQueryAndContext:
    """Property-based tests for Property 18: Prompt contains query and context."""

    @pytest.mark.asyncio
    @given(
        question=st.text(
            min_size=1, max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
        num_chunks=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_18_prompt_contains_query_and_context(
        self,
        query_service,
        mock_openai_client,
        mock_vector_store,
        mock_document_service,
        question,
        num_chunks,
    ):
        """Feature: pdf-rag-system, Property 18: Prompt contains query and context.

        For any query with retrieved chunks, the constructed prompt sent to the LLM
        should contain both the original query text and the content from all
        retrieved chunks.

        Validates: Requirements 5.2
        """
        # Mock document service
        from datetime import UTC, datetime

        from app.models.document import DocumentInfo

        def create_doc_info(doc_id):
            return DocumentInfo(
                doc_id=doc_id,
                filename=f"{doc_id}.pdf",
                file_size=1024,
                num_pages=10,
                num_chunks=5,
                created_at=datetime.now(UTC),
                tags=[],
            )

        mock_document_service.get_document = AsyncMock(
            side_effect=lambda doc_id: create_doc_info(doc_id)
        )

        # Generate search results
        search_results = []
        chunk_contents = []
        for i in range(num_chunks):
            content = f"Chunk content {i}: This is test content for chunk {i}"
            chunk_contents.append(content)
            result = SearchResult(
                chunk_id=f"chunk_{i}",
                doc_id="test_doc",
                content=content,
                page=i,
                chunk_index=i,
                metadata={},
                score=1.0 - (i * 0.1),
            )
            search_results.append(result)

        mock_vector_store.search.return_value = search_results

        # Track what was passed to generate_with_context
        captured_args = {}

        async def capture_generate_call(question, context, temperature):
            captured_args["question"] = question
            captured_args["context"] = context
            captured_args["temperature"] = temperature
            return "Generated answer"

        mock_openai_client.generate_with_context.side_effect = capture_generate_call

        # Execute query
        request = QueryRequest(question=question, top_k=num_chunks)
        await query_service.query(request)

        # Property: Question should be passed to LLM
        assert "question" in captured_args, "Question not passed to LLM"
        assert captured_args["question"] == question, (
            f"Question mismatch: expected '{question}', got '{captured_args['question']}'"
        )

        # Property: Context should contain all chunk contents
        assert "context" in captured_args, "Context not passed to LLM"
        context = captured_args["context"]

        for i, chunk_content in enumerate(chunk_contents):
            assert chunk_content in context, (
                f"Chunk {i} content not found in context. Expected: '{chunk_content}'"
            )

    @pytest.mark.asyncio
    @given(
        question=st.text(
            min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_18_context_includes_metadata(
        self, query_service, mock_openai_client, mock_vector_store, mock_document_service, question
    ):
        """Feature: pdf-rag-system, Property 18: Prompt contains query and context.

        For any query, the context should include metadata like page numbers and
        relevance scores to help the LLM provide better answers.

        Validates: Requirements 5.2
        """
        # Mock document service
        from datetime import UTC, datetime

        from app.models.document import DocumentInfo

        def create_doc_info(doc_id):
            return DocumentInfo(
                doc_id=doc_id,
                filename=f"{doc_id}.pdf",
                file_size=1024,
                num_pages=10,
                num_chunks=5,
                created_at=datetime.now(UTC),
                tags=[],
            )

        mock_document_service.get_document = AsyncMock(
            side_effect=lambda doc_id: create_doc_info(doc_id)
        )

        # Create a search result with specific metadata
        page_num = 42
        relevance = 0.95
        result = SearchResult(
            chunk_id="chunk_1",
            doc_id="test_doc",
            content="Important content from the document",
            page=page_num,
            chunk_index=0,
            metadata={},
            score=relevance,
        )

        mock_vector_store.search.return_value = [result]

        # Track context
        captured_context = None

        async def capture_context(question, context, temperature):
            nonlocal captured_context
            captured_context = context
            return "Answer"

        mock_openai_client.generate_with_context.side_effect = capture_context

        # Execute query
        request = QueryRequest(question=question)
        await query_service.query(request)

        # Property: Context should include page number
        assert captured_context is not None
        assert str(page_num) in captured_context, f"Page number {page_num} not found in context"

        # Property: Context should include relevance score
        assert f"{relevance:.2f}" in captured_context, (
            f"Relevance score {relevance:.2f} not found in context"
        )


class TestResponseIncludesSourceCitations:
    """Property-based tests for Property 19: Response includes source citations."""

    @pytest.mark.asyncio
    @given(
        question=st.text(
            min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
        num_sources=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_19_response_includes_source_citations(
        self,
        query_service,
        mock_openai_client,
        mock_vector_store,
        mock_document_service,
        question,
        num_sources,
    ):
        """Feature: pdf-rag-system, Property 19: Response includes source citations.

        For any query response, the response should include both an answer string
        and a list of source references, where each source reference contains
        doc_id, filename, page number, and relevance score.

        Validates: Requirements 5.3
        """
        # Generate search results
        search_results = []
        for i in range(num_sources):
            result = SearchResult(
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i}",
                content=f"Content {i}",
                page=i + 1,
                chunk_index=i,
                metadata={},
                score=1.0 - (i * 0.1),
            )
            search_results.append(result)

        mock_vector_store.search.return_value = search_results

        # Mock document service to return filenames
        from datetime import UTC, datetime

        from app.models.document import DocumentInfo

        def create_doc_info(doc_id):
            return DocumentInfo(
                doc_id=doc_id,
                filename=f"{doc_id}.pdf",
                file_size=1024,
                num_pages=10,
                num_chunks=5,
                created_at=datetime.now(UTC),
                tags=[],
            )

        mock_document_service.get_document = AsyncMock(
            side_effect=lambda doc_id: create_doc_info(doc_id)
        )

        # Execute query
        request = QueryRequest(question=question, top_k=num_sources, include_sources=True)
        response = await query_service.query(request)

        # Property: Response should have an answer
        assert isinstance(response, QueryResponse)
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

        # Property: Response should have sources list
        assert isinstance(response.sources, list)
        assert len(response.sources) == num_sources, (
            f"Expected {num_sources} sources, got {len(response.sources)}"
        )

        # Property: Each source should have required fields
        for i, source in enumerate(response.sources):
            assert isinstance(source, SourceReference)

            # doc_id should be present
            assert source.doc_id == f"doc_{i}", f"Source {i}: doc_id mismatch"

            # filename should be present
            assert source.filename == f"doc_{i}.pdf", f"Source {i}: filename mismatch"

            # page should be present and non-negative
            assert source.page >= 0, f"Source {i}: page number {source.page} is negative"

            # relevance_score should be in [0, 1]
            assert 0.0 <= source.relevance_score <= 1.0, (
                f"Source {i}: relevance score {source.relevance_score} out of range"
            )

    @pytest.mark.asyncio
    @given(
        question=st.text(
            min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_19_sources_excluded_when_not_requested(
        self, query_service, mock_openai_client, mock_vector_store, question
    ):
        """Feature: pdf-rag-system, Property 19: Response includes source citations.

        For any query with include_sources=False, the response should have an
        empty sources list.

        Validates: Requirements 5.3
        """
        # Generate search results
        result = SearchResult(
            chunk_id="chunk_1",
            doc_id="doc_1",
            content="Content",
            page=1,
            chunk_index=0,
            metadata={},
            score=0.9,
        )

        mock_vector_store.search.return_value = [result]

        # Execute query with include_sources=False
        request = QueryRequest(question=question, include_sources=False)
        response = await query_service.query(request)

        # Property: Sources list should be empty
        assert isinstance(response.sources, list)
        assert len(response.sources) == 0, (
            f"Expected empty sources list, got {len(response.sources)} sources"
        )


class TestMultiSourceSynthesis:
    """Property-based tests for Property 20: Multi-source synthesis."""

    @pytest.mark.asyncio
    @given(
        question=st.text(
            min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
        num_docs=st.integers(min_value=2, max_value=5),
        chunks_per_doc=st.integers(min_value=1, max_value=3),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_20_multi_source_synthesis(
        self,
        query_service,
        mock_openai_client,
        mock_vector_store,
        mock_document_service,
        question,
        num_docs,
        chunks_per_doc,
    ):
        """Feature: pdf-rag-system, Property 20: Multi-source synthesis.

        For any query where multiple documents contain relevant information
        (top-k > 1 with similar scores), the response sources list should include
        references from multiple documents.

        Validates: Requirements 5.5
        """
        # Generate search results from multiple documents with similar scores
        search_results = []
        base_score = 0.9

        for doc_idx in range(num_docs):
            for chunk_idx in range(chunks_per_doc):
                # Similar scores across documents (within 0.1 range)
                score = base_score - (doc_idx * 0.02) - (chunk_idx * 0.01)
                result = SearchResult(
                    chunk_id=f"doc{doc_idx}_chunk{chunk_idx}",
                    doc_id=f"doc_{doc_idx}",
                    content=f"Content from doc {doc_idx}, chunk {chunk_idx}",
                    page=chunk_idx + 1,
                    chunk_index=chunk_idx,
                    metadata={},
                    score=max(0.0, score),
                )
                search_results.append(result)

        # Sort by relevance (descending)
        search_results.sort(key=lambda r: r.relevance_score, reverse=True)

        mock_vector_store.search.return_value = search_results

        # Mock document service
        from datetime import UTC, datetime

        from app.models.document import DocumentInfo

        def create_doc_info(doc_id):
            return DocumentInfo(
                doc_id=doc_id,
                filename=f"{doc_id}.pdf",
                file_size=1024,
                num_pages=10,
                num_chunks=5,
                created_at=datetime.now(UTC),
                tags=[],
            )

        mock_document_service.get_document = AsyncMock(
            side_effect=lambda doc_id: create_doc_info(doc_id)
        )

        # Execute query
        total_chunks = num_docs * chunks_per_doc
        request = QueryRequest(question=question, top_k=total_chunks, include_sources=True)
        response = await query_service.query(request)

        # Property: Response should have sources from multiple documents
        unique_doc_ids = set(source.doc_id for source in response.sources)

        assert len(unique_doc_ids) >= 2, (
            f"Expected sources from at least 2 documents, got {len(unique_doc_ids)} unique doc_ids"
        )

        # Property: Sources should include references from multiple documents
        # (at least 2 different documents when we have multiple docs with similar scores)
        assert len(unique_doc_ids) == num_docs, (
            f"Expected sources from {num_docs} documents, got {len(unique_doc_ids)}"
        )

    @pytest.mark.asyncio
    @given(
        question=st.text(
            min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
        ),
        num_docs=st.integers(min_value=2, max_value=5),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    async def test_property_20_context_from_multiple_sources(
        self,
        query_service,
        mock_openai_client,
        mock_vector_store,
        mock_document_service,
        question,
        num_docs,
    ):
        """Feature: pdf-rag-system, Property 20: Multi-source synthesis.

        For any query with results from multiple documents, the context passed to
        the LLM should include content from all documents.

        Validates: Requirements 5.5
        """
        # Mock document service
        from datetime import UTC, datetime

        from app.models.document import DocumentInfo

        def create_doc_info(doc_id):
            return DocumentInfo(
                doc_id=doc_id,
                filename=f"{doc_id}.pdf",
                file_size=1024,
                num_pages=10,
                num_chunks=5,
                created_at=datetime.now(UTC),
                tags=[],
            )

        mock_document_service.get_document = AsyncMock(
            side_effect=lambda doc_id: create_doc_info(doc_id)
        )

        # Generate one result per document
        search_results = []
        doc_contents = []

        for doc_idx in range(num_docs):
            content = f"Unique content from document {doc_idx}"
            doc_contents.append(content)
            result = SearchResult(
                chunk_id=f"chunk_{doc_idx}",
                doc_id=f"doc_{doc_idx}",
                content=content,
                page=1,
                chunk_index=0,
                metadata={},
                score=0.9 - (doc_idx * 0.05),
            )
            search_results.append(result)

        mock_vector_store.search.return_value = search_results

        # Track context
        captured_context = None

        async def capture_context(question, context, temperature):
            nonlocal captured_context
            captured_context = context
            return "Synthesized answer"

        mock_openai_client.generate_with_context.side_effect = capture_context

        # Execute query
        request = QueryRequest(question=question, top_k=num_docs)
        await query_service.query(request)

        # Property: Context should include content from all documents
        assert captured_context is not None

        for doc_idx, content in enumerate(doc_contents):
            assert content in captured_context, (
                f"Content from document {doc_idx} not found in context: '{content}'"
            )
