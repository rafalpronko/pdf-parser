"""Test BGE-reranker-v2-m3 model."""

import asyncio
import os

from app.models.query import QueryRequest
from app.services.query_service import QueryService
from app.config import reload_settings

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def test_bge_reranker():
    """Test z BGE-reranker-v2-m3 (lepszy multilingual model)."""
    os.environ["EXPANSION_METHOD"] = "hybrid"
    os.environ["ENABLE_RERANKING"] = "true"
    os.environ["RERANKER_MODEL"] = "BAAI/bge-reranker-v2-m3"  # NOWY MODEL
    reload_settings()

    query_service = QueryService()

    question = "Co jest przedmiotem ubezpieczenia?"

    print("=" * 80)
    print("TEST: BGE-RERANKER-V2-M3 (Multilingual Model)")
    print("=" * 80)
    print(f"\nPytanie: {question}")
    print(f"Expansion: Hybrid (HyDE + Multi-Query)")
    print(f"Reranker: BAAI/bge-reranker-v2-m3")
    print(f"  - Multilingual (lepszy dla polskiego)")
    print(f"  - Lepiej obsługuje listy i struktury")
    print(f"  - State-of-the-art quality\n")

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

    print(f"\n📝 Odpowiedź:")
    print(f"{result.answer}\n")

    print(f"📊 Statystyki:")
    print(f"  Czas: {result.processing_time:.2f}s")
    print(f"  Liczba źródeł: {len(result.sources)}")

    if result.sources:
        print(f"\n📚 Top 10 Źródeł (BGE-reranker-v2-m3 + RRF):")
        for i, source in enumerate(result.sources, 1):
            is_target = "pojazd (silnikowy" in source.chunk_content.lower()
            marker = " ← SZUKANY CHUNK!" if is_target else ""

            print(
                f"\n  {i}. Strona {source.page}, Score: {source.relevance_score:.4f}{marker}"
            )
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
    print("PORÓWNANIE WSZYSTKICH METOD")
    print("=" * 80)

    print("\nMetoda                                  | Frazy | Status")
    print("-" * 70)
    print("MS-MARCO (tylko oryginalne pytanie)     | 2/5   | ✗ Niepełna")
    print("MS-MARCO + RRF (każdy wariant osobno)   | 3/5   | ~ Lepsza")
    print("BEZ rerankingu                          | 4/5   | ✓ Pełna")
    print(f"BGE-reranker-v2-m3 + RRF (NOWA)         | {found}/5   | ?")

    if found >= 4:
        print("\n" + "=" * 80)
        print("✓✓✓ SUKCES! ✓✓✓")
        print("=" * 80)
        print("\nBGE-reranker-v2-m3 działa świetnie!")
        print("- Multilingual model lepiej rozumie polski")
        print("- Lepiej radzi sobie z listami punktowanymi")
        print("- Zachowuje precision z rerankingu + recall z expansion")
    elif "pojazd" in result.answer.lower() and "silnikowy" in result.answer.lower():
        print("\n✓ SUKCES! Odpowiedź zawiera kluczowe informacje")
    else:
        print("\n~ Poprawa, ale nadal nie idealna")


if __name__ == "__main__":
    asyncio.run(test_bge_reranker())
