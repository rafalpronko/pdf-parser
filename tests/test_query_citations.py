"""End-to-end citation tests for QueryService with reranking enabled.

Regression coverage for citations losing doc_id/filename/page after the
cross-encoder reranking step (every source ended up as doc_id="", page=0 and
the LLM context said "Page 0" for every chunk).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import Settings
from app.models.query import QueryRequest
from app.models.search import SearchResult
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.services.query_service import QueryService

# (chunk_id, doc_id, page, chunk_index, content, vector score)
CHUNKS = [
    (
        "chunk-alpha-3",
        "doc-alpha",
        3,
        0,
        "Transformers use self-attention to model interactions between tokens.",
        0.9,
    ),
    (
        "chunk-beta-7",
        "doc-beta",
        7,
        4,
        "Residual networks add skip connections so gradients flow through deep layers.",
        0.8,
    ),
    (
        "chunk-alpha-12",
        "doc-alpha",
        12,
        9,
        "Batch normalization stabilizes training of deep residual networks.",
        0.7,
    ),
]
FILENAMES = {"doc-alpha": "alpha-paper.pdf", "doc-beta": "beta-paper.pdf"}

# Cross-encoder scores chosen so the reranked order differs from retrieval order
RERANK_SCORES = {CHUNKS[0][4]: 1.0, CHUNKS[1][4]: -2.0, CHUNKS[2][4]: 5.0}
EXPECTED_ORDER = ["chunk-alpha-12", "chunk-alpha-3", "chunk-beta-7"]

QUESTION = "How do residual networks and self-attention work?"


class FakeVectorStore:
    """Returns results shaped like VectorStore.search (citation fields on the dataclass)."""

    def __init__(self):
        self.calls = 0

    async def search(self, query_embedding, top_k=5, doc_id=None):  # noqa: ARG002
        self.calls += 1
        return [
            SearchResult(
                chunk_id=chunk_id,
                content=content,
                score=score,
                doc_id=chunk_doc_id,
                page=page,
                chunk_index=chunk_index,
                metadata={"embedding_model": "text-embedding-3-small"},
            )
            for chunk_id, chunk_doc_id, page, chunk_index, content, score in CHUNKS
        ][:top_k]


class FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder."""

    def __init__(self):
        self.predict_calls = 0

    def predict(self, pairs, batch_size=32):  # noqa: ARG002
        self.predict_calls += 1
        return [RERANK_SCORES[text] for _, text in pairs]


class FakeQueryExpander:
    """Multi-query expander returning the original question plus two variants."""

    async def expand(self, query):
        return [query, "What are skip connections?", "Explain self-attention in transformers"]


def _build_bm25_index() -> BM25Index:
    """In-memory BM25 index populated the same way DocumentService does it."""
    index = BM25Index(persist_path=None)
    index.add_documents(
        [chunk_id for chunk_id, *_ in CHUNKS],
        [content for *_, content, _ in CHUNKS],
        [
            {"doc_id": doc_id, "page": page, "chunk_index": chunk_index}
            for _, doc_id, page, chunk_index, _, _ in CHUNKS
        ],
    )
    return index


@pytest.fixture
def make_service(tmp_path, monkeypatch):
    """Factory building a QueryService with reranking enabled and fake collaborators."""

    def _make(*, hybrid: bool, expansion: bool):
        settings = Settings(
            _env_file=None,
            openai_api_key=None,
            upload_dir=tmp_path / "uploads",
            vector_db_path=str(tmp_path / "vectordb"),
            enable_hybrid_search=hybrid,
            enable_query_expansion=expansion,
            enable_reranking=True,
            reranking_top_k=10,
            final_top_k=3,
            enable_gpu=False,
        )
        monkeypatch.setattr("app.config._settings", settings)
        monkeypatch.setattr("app.services.query_service.get_settings", lambda: settings)

        openai_client = MagicMock()
        openai_client.embed_text = AsyncMock(return_value=[0.1] * 8)
        openai_client.generate_with_context = AsyncMock(return_value="Generated answer")
        openai_client.close = AsyncMock()

        document_service = MagicMock()
        document_service.get_document = AsyncMock(
            side_effect=lambda doc_id: SimpleNamespace(filename=FILENAMES[doc_id])
        )

        # Build the reranker without downloading a real cross-encoder model
        monkeypatch.setattr(CrossEncoderReranker, "_init_model", lambda _self: None)
        reranker = CrossEncoderReranker(device="cpu")
        reranker.model = FakeCrossEncoder()

        vector_store = FakeVectorStore()
        bm25_index = _build_bm25_index()
        hybrid_search = HybridSearchEngine(
            vector_store=vector_store,
            bm25_index=bm25_index,
            vector_weight=0.7,
            keyword_weight=0.3,
        )

        service = QueryService(
            openai_client=openai_client,
            vector_store=vector_store,
            document_service=document_service,
            bm25_index=bm25_index,
            hybrid_search=hybrid_search,
            query_expander=FakeQueryExpander() if expansion else None,
            reranker=reranker,
        )
        return service, openai_client, document_service, reranker.model, vector_store

    return _make


@pytest.mark.parametrize("hybrid", [True, False], ids=["hybrid", "vector-only"])
@pytest.mark.parametrize("expansion", [False, True], ids=["single-query", "multi-query"])
async def test_reranked_sources_keep_citation_fields(make_service, hybrid, expansion):
    """Sources and LLM context carry the right doc_id, filename and page after reranking."""
    service, openai_client, document_service, model, vector_store = make_service(
        hybrid=hybrid, expansion=expansion
    )

    response = await service.query(QueryRequest(question=QUESTION, top_k=3))

    # The request really went through retrieval and the cross-encoder
    num_queries = 3 if expansion else 1
    assert vector_store.calls == num_queries
    assert model.predict_calls >= 1

    by_content = {content: (doc_id, page) for _, doc_id, page, _, content, _ in CHUNKS}
    ids_by_content = {content: chunk_id for chunk_id, *_, content, _ in CHUNKS}

    # Sources follow the reranked order and carry correct citation fields
    assert [ids_by_content[s.chunk_content] for s in response.sources] == EXPECTED_ORDER
    for source in response.sources:
        expected_doc_id, expected_page = by_content[source.chunk_content]
        assert source.doc_id == expected_doc_id
        assert source.filename == FILENAMES[expected_doc_id]
        assert source.page == expected_page

    looked_up = {call.args[0] for call in document_service.get_document.await_args_list}
    assert looked_up == {"doc-alpha", "doc-beta"}

    # Context passed to the LLM cites the real page numbers
    context = openai_client.generate_with_context.await_args.kwargs["context"]
    assert "[Source 1] (Page 12," in context
    assert "[Source 2] (Page 3," in context
    assert "[Source 3] (Page 7," in context
    assert "Page 0" not in context
