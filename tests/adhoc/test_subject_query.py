"""Test query: 'Co jest przedmiotem ubezpieczenia?'"""

import asyncio
import os

from app.config import reload_settings
from app.models.query import QueryRequest
from app.services.query_service import QueryService

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def test_with_method(method_name: str, question: str):
    """Test query with specific expansion method."""
    os.environ["EXPANSION_METHOD"] = method_name
    reload_settings()

    query_service = QueryService()

    result = await query_service.query(
        QueryRequest(
            question=question,
            top_k=5,
            include_sources=True,
        )
    )

    return result


async def main():
    question = "Co jest przedmiotem ubezpieczenia?"

    print("=" * 80)
    print("TEST PYTANIA: Co jest przedmiotem ubezpieczenia?")
    print("=" * 80)
    print(f"\nPytanie: {question}\n")

    methods = ["multi-query", "hyde", "hybrid"]
    results = {}

    for method in methods:
        print("=" * 80)
        print(f"METODA: {method.upper()}")
        print("=" * 80)

        result = await test_with_method(method, question)
        results[method] = result

        print("\n📝 Odpowiedź:")
        print(f"{result.answer}\n")

        print("📊 Statystyki:")
        print(f"  Czas: {result.processing_time:.2f}s")
        print(f"  Liczba źródeł: {len(result.sources)}")

        if result.sources:
            print("\n📚 Top 3 Źródła:")
            for i, source in enumerate(result.sources[:3], 1):
                print(f"  {i}. Strona {source.page}, Relevance: {source.relevance_score:.4f}")
                print(f"     {source.chunk_content[:100]}...")

        print()

    # Porównanie
    print("=" * 80)
    print("PORÓWNANIE METOD")
    print("=" * 80)

    print("\n📊 Czasy przetwarzania:")
    for method, result in results.items():
        print(f"  {method:12s}: {result.processing_time:5.2f}s")

    print("\n📏 Długość odpowiedzi:")
    for method, result in results.items():
        print(f"  {method:12s}: {len(result.answer):4d} znaków")

    # Sprawdź kluczowe frazy dla tego pytania
    key_terms = [
        "pojazd",
        "autocasco",
        "ubezpieczenie",
        "szkoda",
        "ochrona",
    ]

    print("\n🔍 Obecność kluczowych terminów:")
    for method, result in results.items():
        found = sum(1 for term in key_terms if term.lower() in result.answer.lower())
        print(f"  {method:12s}: {found}/{len(key_terms)} terminów")

    # Która metoda najlepsza?
    print("\n" + "=" * 80)
    print("REKOMENDACJA")
    print("=" * 80)

    best_method = max(
        results.items(),
        key=lambda x: sum(1 for term in key_terms if term.lower() in x[1].answer.lower()),
    )

    print(f"\n✓ Najlepsza metoda dla tego pytania: {best_method[0].upper()}")
    print("  Znalazła najwięcej kluczowych terminów")

    # Pokaż różnice w odpowiedziach
    print("\n" + "=" * 80)
    print("SZCZEGÓŁOWE ODPOWIEDZI")
    print("=" * 80)

    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print("-" * 80)
        print(result.answer)
        print()


if __name__ == "__main__":
    asyncio.run(main())
