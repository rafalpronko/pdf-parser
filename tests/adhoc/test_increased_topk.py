"""Test with increased reranking_top_k and final_top_k."""

import asyncio
import os

from app.config import reload_settings
from app.models.query import QueryRequest
from app.services.query_service import QueryService

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def test_increased_topk():
    """Test with new settings: reranking_top_k=40, final_top_k=10."""
    os.environ["EXPANSION_METHOD"] = "hybrid"
    os.environ["ENABLE_RERANKING"] = "true"
    reload_settings()

    query_service = QueryService()

    question = "Co jest przedmiotem ubezpieczenia?"

    print("=" * 80)
    print("TEST: ZWIĘKSZONE TOP_K")
    print("=" * 80)
    print(f"\nPytanie: {question}")
    print("Ustawienia:")
    print("  - reranking_top_k: 40 (było 20)")
    print("  - final_top_k: 10 (było 5)")
    print("  - Reranking: WŁĄCZONY\n")

    result = await query_service.query(
        QueryRequest(
            question=question,
            top_k=10,
            include_sources=True,
        )
    )

    print("=" * 80)
    print("WYNIKI")
    print("=" * 80)

    print("\n📝 Odpowiedź:")
    print(f"{result.answer}\n")

    print("📊 Statystyki:")
    print(f"  Czas: {result.processing_time:.2f}s")
    print(f"  Liczba źródeł: {len(result.sources)}")

    if result.sources:
        print("\n📚 Top 10 Źródeł (PO rerankingu z większą pulą):")
        for i, source in enumerate(result.sources, 1):
            is_target = "pojazd (silnikowy" in source.chunk_content.lower()
            marker = " ← SZUKANY CHUNK!" if is_target else ""

            print(f"\n  {i}. Strona {source.page}, Relevance: {source.relevance_score:.4f}{marker}")
            print(f"     Chunk: {source.chunk_content[:150]}...")

    # Sprawdź czy odpowiedź zawiera kluczowe frazy
    key_phrases = [
        "pojazd",
        "silnikowy",
        "przyczepa",
        "wyposażenie",
        "suma ubezpieczenia",
    ]

    print("\n" + "=" * 80)
    print("ANALIZA JAKOŚCI")
    print("=" * 80)

    found = sum(1 for phrase in key_phrases if phrase.lower() in result.answer.lower())
    print(f"\n✓ Kluczowe frazy: {found}/{len(key_phrases)}")

    for phrase in key_phrases:
        status = "✓" if phrase.lower() in result.answer.lower() else "✗"
        print(f"  {status} {phrase}")

    if "pojazd" in result.answer.lower() and "silnikowy" in result.answer.lower():
        print("\n✓ SUKCES: Odpowiedź zawiera pełną informację o przedmiocie!")
        print("  Rozwiązanie: Zwiększenie reranking_top_k i final_top_k pomogło!")
    else:
        print("\n✗ PROBLEM: Odpowiedź nadal niekompletna")
        print("  Rozwiązanie: Trzeba zastosować inne podejście (RRF lub wyłączyć reranking)")


if __name__ == "__main__":
    asyncio.run(test_increased_topk())
