"""Debug reranking to see where the correct chunk goes."""

import asyncio
import os

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.storage.vector_store import VectorStore

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def debug_reranking():
    """Debug where the chunk with 'Pojazd (silnikowy...' goes during reranking."""
    settings = get_settings()
    openai_client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )

    question = "Co jest przedmiotem ubezpieczenia?"

    print("=" * 80)
    print("DEBUG: Reranking Analysis")
    print("=" * 80)
    print(f"\nPytanie: {question}\n")

    # Get embedding for question
    embedding = await openai_client.embed_text(question)

    # Search vector store directly
    vector_store = VectorStore(
        persist_directory=settings.vector_db_path,
        collection_name=settings.text_collection,
    )

    # Get top 20 results BEFORE reranking
    print("=" * 80)
    print("STEP 1: Vector Search (top 20 PRZED rerankingiem)")
    print("=" * 80)

    results = await vector_store.search(
        query_embedding=embedding,
        top_k=20,
    )

    print(f"\nZnaleziono {len(results)} wyników:\n")

    # Find the chunk with "Pojazd (silnikowy"
    target_chunk_position = None

    for i, result in enumerate(results, 1):
        # Check if this is the chunk we're looking for
        is_target = "pojazd (silnikowy" in result.content.lower()

        marker = " ← SZUKANY CHUNK!" if is_target else ""

        if is_target:
            target_chunk_position = i

        print(f"{i:2d}. Page {result.page}, Score: {result.relevance_score:.4f}{marker}")
        print(f"    Chunk ID: {result.chunk_id}")
        print(f"    Content: {result.content[:120]}...")

        if is_target:
            print("    PEŁNA TREŚĆ:")
            print(f"    {result.content[:300]}...")

        print()

    if target_chunk_position:
        print("=" * 80)
        print(f"✓ ZNALEZIONO chunk z 'Pojazd (silnikowy' na pozycji {target_chunk_position}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("✗ NIE ZNALEZIONO chunka z 'Pojazd (silnikowy' w top-20!")
        print("=" * 80)

    # Now test with reranking
    print("\n" + "=" * 80)
    print("STEP 2: Reranking Analysis")
    print("=" * 80)

    from app.retrieval.reranker import CrossEncoderReranker
    from app.retrieval.reranker import SearchResult as RerankerSearchResult

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        batch_size=settings.reranking_batch_size,
        device="auto" if settings.enable_gpu else "cpu",
    )

    # Convert to reranker format
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
        for r in results
    ]

    # Rerank top 10
    reranked = reranker.rerank(
        query=question,
        chunks=reranker_results,
        top_k=10,
    )

    print("\nTop 10 PO rerankingu:\n")

    reranked_target_position = None

    for i, result in enumerate(reranked, 1):
        is_target = "pojazd (silnikowy" in result.content.lower()

        marker = " ← SZUKANY CHUNK!" if is_target else ""

        if is_target:
            reranked_target_position = i

        print(f"{i:2d}. Score: {result.score:.4f}{marker}")
        print(f"    Content: {result.content[:120]}...")
        print()

    print("=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)

    if target_chunk_position:
        print(
            f"\n✓ Chunk z 'Pojazd (silnikowy' był na pozycji {target_chunk_position} przed rerankingiem"
        )

        if reranked_target_position:
            print(f"✓ Po rerankingu przesunął się na pozycję {reranked_target_position}")
            if reranked_target_position > 5:
                print("  ⚠ To wciąż poza top-5 używanym domyślnie!")
        else:
            print("✗ Po rerankingu WYPADŁ z top-10!")
            print("  Problem: Cross-encoder uznał inne chunki za bardziej relevantne")
    else:
        print("✗ Chunk z 'Pojazd (silnikowy' NIE JEST w top-20 vector search!")
        print("  Problem: Query expansion nie generuje odpowiednich zapytań")


if __name__ == "__main__":
    asyncio.run(debug_reranking())
