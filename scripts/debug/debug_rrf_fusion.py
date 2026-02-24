"""Debug RRF fusion szczegółowo."""

import asyncio
import os

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.retrieval.query_expansion import QueryExpander
from app.storage.vector_store import VectorStore
from app.retrieval.reranker import CrossEncoderReranker, SearchResult as RerankerSearchResult

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"
os.environ["EXPANSION_METHOD"] = "hybrid"


async def debug_rrf_fusion():
    """Sprawdź ranking dla każdego query wariantu osobno."""
    settings = get_settings()
    openai_client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )

    question = "Co jest przedmiotem ubezpieczenia?"

    print("=" * 80)
    print("DEBUG: RRF FUSION - Reranking dla każdego wariantu")
    print("=" * 80)

    # Query expansion
    expander = QueryExpander(
        llm_client=openai_client,
        method="hybrid",
        num_variations=3,
    )
    queries = await expander.expand(question)

    print(f"\nWygenerowano {len(queries)} wariantów\n")

    # Get embeddings and search
    vector_store = VectorStore(
        persist_directory=settings.vector_db_path,
        collection_name=settings.text_collection,
    )

    all_results = []
    for query in queries:
        emb = await openai_client.embed_text(query)
        results = await vector_store.search(query_embedding=emb, top_k=40)
        all_results.extend(results)

    # Deduplicate
    seen = set()
    unique_results = []
    for r in all_results:
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            unique_results.append(r)

    print(f"Po deduplikacji: {len(unique_results)} chunków\n")

    # Setup reranker
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        batch_size=settings.reranking_batch_size,
        device="auto",
    )

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
        for r in unique_results
    ]

    # Rerank dla każdego wariantu
    print("=" * 80)
    print("RERANKING DLA KAŻDEGO QUERY WARIANTU")
    print("=" * 80)

    all_reranked = []

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query[:80]}...")
        print("-" * 80)

        reranked = reranker.rerank(
            query=query,
            chunks=reranker_results,
            top_k=20,
        )
        all_reranked.append(reranked)

        # Znajdź chunk z "pojazd (silnikowy"
        target_rank = None
        for rank, doc in enumerate(reranked, 1):
            if "pojazd (silnikowy" in doc.content.lower():
                target_rank = rank
                print(f"✓ Chunk 'Pojazd (silnikowy...' na pozycji #{rank}, score: {doc.score:.4f}")
                break

        if not target_rank:
            print(f"✗ Chunk 'Pojazd (silnikowy...' NIE w top-20!")

        # Pokaż top 5 dla tego wariantu
        print("\nTop 5:")
        for rank, doc in enumerate(reranked[:5], 1):
            is_target = "pojazd (silnikowy" in doc.content.lower()
            marker = " ← TARGET" if is_target else ""
            print(f"  {rank}. Score: {doc.score:.4f}{marker}")
            print(f"     {doc.content[:80]}...")

    # RRF Fusion
    print("\n" + "=" * 80)
    print("RRF FUSION")
    print("=" * 80)

    # Oblicz RRF scores
    rrf_scores = {}
    for ranked_list in all_reranked:
        for rank, doc in enumerate(ranked_list, start=1):
            rrf_score = 1.0 / (60 + rank)
            if doc.chunk_id in rrf_scores:
                rrf_scores[doc.chunk_id] += rrf_score
            else:
                rrf_scores[doc.chunk_id] = rrf_score

    # Znajdź target chunk w RRF
    target_chunk_id = None
    for doc in reranker_results:
        if "pojazd (silnikowy" in doc.content.lower():
            target_chunk_id = doc.chunk_id
            break

    if target_chunk_id:
        if target_chunk_id in rrf_scores:
            target_rrf_score = rrf_scores[target_chunk_id]
            # Pozycja w rankingu RRF
            sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            target_rrf_rank = None
            for rank, (chunk_id, score) in enumerate(sorted_by_rrf, 1):
                if chunk_id == target_chunk_id:
                    target_rrf_rank = rank
                    break

            print(f"\n✓ Chunk 'Pojazd (silnikowy...' w RRF:")
            print(f"  RRF Score: {target_rrf_score:.4f}")
            print(f"  RRF Rank: #{target_rrf_rank}")

            # Dekompozycja RRF score
            print(f"\n  Dekompozycja RRF score:")
            for i, ranked_list in enumerate(all_reranked, 1):
                for rank, doc in enumerate(ranked_list, start=1):
                    if doc.chunk_id == target_chunk_id:
                        contribution = 1.0 / (60 + rank)
                        print(
                            f"    Query {i}: rank #{rank:2d} → contribution {contribution:.4f}"
                        )
                        break
        else:
            print(f"\n✗ Chunk 'Pojazd (silnikowy...' NIE w RRF scores!")

    # Pokaż top 10 RRF
    print("\n" + "=" * 80)
    print("TOP 10 PO RRF FUSION")
    print("=" * 80)

    sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    for rank, (chunk_id, score) in enumerate(sorted_by_rrf, 1):
        # Znajdź dokument
        doc = None
        for d in reranker_results:
            if d.chunk_id == chunk_id:
                doc = d
                break

        is_target = "pojazd (silnikowy" in doc.content.lower() if doc else False
        marker = " ← TARGET" if is_target else ""

        print(f"\n{rank}. RRF Score: {score:.4f}{marker}")
        if doc:
            print(f"   {doc.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(debug_rrf_fusion())
