"""Test nowego multi-query reranking z RRF fusion."""

import asyncio
import os

from app.config import reload_settings
from app.models.query import QueryRequest
from app.services.query_service import QueryService

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def test_rrf_reranking():
    """Test nowej implementacji: reranking dla każdego wariantu + RRF."""
    os.environ["EXPANSION_METHOD"] = "hybrid"
    os.environ["ENABLE_RERANKING"] = "true"
    reload_settings()

    query_service = QueryService()

    question = "Co jest przedmiotem ubezpieczenia?"

    print("=" * 80)
    print("TEST: NOWY RERANKING Z RRF FUSION")
    print("=" * 80)
    print(f"\nPytanie: {question}")
    print("Metoda: Hybrid (HyDE + Multi-Query)")
    print("Reranking: WŁĄCZONY (dla każdego wariantu osobno + RRF)\n")

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
        print("\n📚 Top 10 Źródeł (po RRF fusion):")
        for i, source in enumerate(result.sources, 1):
            is_target = "pojazd (silnikowy" in source.chunk_content.lower()
            marker = " ← SZUKANY CHUNK!" if is_target else ""

            print(f"\n  {i}. Strona {source.page}, RRF Score: {source.relevance_score:.4f}{marker}")
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

    print("\n" + "=" * 80)
    print("PORÓWNANIE Z POPRZEDNIMI METODAMI")
    print("=" * 80)

    print("\nMetoda                           | Frazy | Status")
    print("-" * 60)
    print("Reranking (tylko oryginalne)     | 2/5   | ✗ Niepełna")
    print("BEZ rerankingu                   | 4/5   | ✓ Pełna")
    print("Reranking + RRF (NOWA METODA)    | ?/5   | ?")

    if "pojazd" in result.answer.lower() and "silnikowy" in result.answer.lower():
        print("\n" + "=" * 80)
        print("✓✓✓ SUKCES! ✓✓✓")
        print("=" * 80)
        print("\nNowa metoda działa!")
        print("- Reranking dla każdego query wariantu")
        print("- RRF fusion łączy najlepsze wyniki")
        print("- Odpowiedź zawiera pełną informację o przedmiocie ubezpieczenia")
    else:
        print("\n✗ Odpowiedź nadal niekompletna - wymaga dalszej analizy")


if __name__ == "__main__":
    asyncio.run(test_rrf_reranking())
