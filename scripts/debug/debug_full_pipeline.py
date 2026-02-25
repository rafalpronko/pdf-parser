"""Debug całego pipeline'u krok po kroku."""

import asyncio
import os

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.retrieval.query_expansion import QueryExpander
from app.retrieval.reranker import CrossEncoderReranker
from app.storage.vector_store import VectorStore

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"
os.environ["EXPANSION_METHOD"] = "hybrid"


async def debug_full_pipeline():
    """Śledź każdy krok pipeline'u."""
    settings = get_settings()
    openai_client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )

    question = "Co jest przedmiotem ubezpieczenia?"

    print("=" * 80)
    print("DEBUG: PEŁNY PIPELINE RAG")
    print("=" * 80)
    print(f"\nPytanie: {question}\n")

    # =========================================================================
    # KROK 1: Query Expansion
    # =========================================================================
    print("=" * 80)
    print("KROK 1: QUERY EXPANSION (Hybrid)")
    print("=" * 80)

    expander = QueryExpander(
        llm_client=openai_client,
        method="hybrid",
        num_variations=3,
    )

    queries = await expander.expand(question)

    print(f"\nWygenerowano {len(queries)} wariantów:\n")
    for i, q in enumerate(queries, 1):
        query_type = "HyDE" if i == 1 else f"Multi-{i - 1}"
        print(f"{i}. [{query_type}] {q[:100]}...")

    # =========================================================================
    # KROK 2: Embeddingi
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 2: GENEROWANIE EMBEDDINGÓW")
    print("=" * 80)

    embeddings = []
    for i, query in enumerate(queries, 1):
        emb = await openai_client.embed_text(query)
        embeddings.append(emb)
        print(f"{i}. Embedding dla wariantu {i}: dim={len(emb)}")

    # =========================================================================
    # KROK 3: Vector Search dla KAŻDEGO wariantu
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 3: VECTOR SEARCH (dla każdego wariantu osobno)")
    print("=" * 80)

    vector_store = VectorStore(
        persist_directory=settings.vector_db_path,
        collection_name=settings.text_collection,
    )

    all_results = []
    target_chunk_ranks = []  # Gdzie chunk #7 jest w każdym wyszukaniu

    for i, (query, embedding) in enumerate(zip(queries, embeddings), 1):
        results = await vector_store.search(
            query_embedding=embedding,
            top_k=40,
        )
        all_results.extend(results)

        # Znajdź gdzie jest chunk z "pojazd (silnikowy"
        target_rank = None
        for rank, result in enumerate(results, 1):
            if "pojazd (silnikowy" in result.content.lower():
                target_rank = rank
                break

        query_type = "HyDE" if i == 1 else f"Multi-{i - 1}"
        print(f"\n{i}. [{query_type}]")
        print(f"   Znaleziono: {len(results)} chunków")
        if target_rank:
            print(f"   ✓ Chunk 'Pojazd (silnikowy...' na pozycji #{target_rank}")
            target_chunk_ranks.append((query_type, target_rank))
        else:
            print("   ✗ Chunk 'Pojazd (silnikowy...' NIE ZNALEZIONY w top-40")

    # =========================================================================
    # KROK 4: Deduplication
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 4: DEDUPLICATION")
    print("=" * 80)

    print(f"\nPrzed deduplikacją: {len(all_results)} chunków")

    seen_chunks = set()
    unique_results = []
    target_in_unique = False

    for result in all_results:
        if result.chunk_id not in seen_chunks:
            seen_chunks.add(result.chunk_id)
            unique_results.append(result)

            if "pojazd (silnikowy" in result.content.lower():
                target_in_unique = True

    print(f"Po deduplikacji: {len(unique_results)} unikalnych chunków")

    if target_in_unique:
        print("✓ Chunk 'Pojazd (silnikowy...' ZACHOWANY po deduplikacji")
    else:
        print("✗ Chunk 'Pojazd (silnikowy...' UTRACONY podczas deduplikacji")

    # =========================================================================
    # KROK 5: Reranking z ORYGINALNYM pytaniem
    # =========================================================================
    print("\n" + "=" * 80)
    print("KROK 5: RERANKING (używa TYLKO oryginalnego pytania!)")
    print("=" * 80)

    print(f"\nReranking query: '{question}'")
    print(f"Liczba chunków do rerankingu: {len(unique_results)}")

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        batch_size=settings.reranking_batch_size,
        device="auto",
    )

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
        for r in unique_results
    ]

    reranked = reranker.rerank(
        query=question,  # ← TUTAJ! Używa tylko oryginalnego pytania
        chunks=reranker_results,
        top_k=10,
    )

    print("\nTop 10 po rerankingu:\n")

    target_in_reranked = False
    for i, result in enumerate(reranked, 1):
        is_target = "pojazd (silnikowy" in result.content.lower()
        marker = " ← SZUKANY CHUNK!" if is_target else ""

        if is_target:
            target_in_reranked = True

        print(f"{i:2d}. Score: {result.score:.4f}{marker}")
        print(f"    {result.content[:100]}...")

    # =========================================================================
    # PODSUMOWANIE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)

    print("\n1. Query Expansion:")
    print(f"   Wygenerowano {len(queries)} wariantów (1 HyDE + {len(queries) - 1} Multi-Query)")

    print("\n2. Vector Search:")
    if target_chunk_ranks:
        print(f"   ✓ Chunk znaleziony w {len(target_chunk_ranks)}/{len(queries)} wyszukań:")
        for query_type, rank in target_chunk_ranks:
            print(f"     - {query_type}: pozycja #{rank}")
    else:
        print("   ✗ Chunk NIE ZNALEZIONY w żadnym wyszukaniu")

    print("\n3. Deduplication:")
    print(f"   {len(all_results)} → {len(unique_results)} chunków")
    if target_in_unique:
        print("   ✓ Chunk ZACHOWANY")
    else:
        print("   ✗ Chunk UTRACONY")

    print("\n4. Reranking:")
    print("   Query: TYLKO oryginalne pytanie (nie warianty!)")
    if target_in_reranked:
        print("   ✓ Chunk ZACHOWANY w top-10")
    else:
        print("   ✗ Chunk USUNIĘTY z top-10")

    print("\n" + "=" * 80)
    print("WNIOSEK")
    print("=" * 80)

    if target_chunk_ranks and target_in_unique and not target_in_reranked:
        print("""
✗ PROBLEM ZIDENTYFIKOWANY:

1. Vector search ZNAJDUJE właściwy chunk (dzięki HyDE/Multi-Query)
2. Deduplication ZACHOWUJE właściwy chunk
3. Reranking USUWA właściwy chunk!

PRZYCZYNA:
- Reranking używa TYLKO oryginalnego pytania
- Cross-encoder MS-MARCO preferuje "Suma ubezpieczenia..."
  nad "Pojazd (silnikowy..." dla tego pytania
- HyDE/Multi-Query pomogły znaleźć chunk, ale reranking to zepsuł

ROZWIĄZANIE:
- Wyłączyć reranking (ENABLE_RERANKING=false)
- LUB rerankować dla każdego query wariantu osobno
- LUB użyć lepszego reranker model
""")


if __name__ == "__main__":
    asyncio.run(debug_full_pipeline())
