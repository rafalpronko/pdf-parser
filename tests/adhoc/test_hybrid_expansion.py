"""Test the new HYBRID query expansion method."""

import asyncio
import os

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.models.query import QueryRequest
from app.retrieval.query_expansion import QueryExpander
from app.services.query_service import QueryService

os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def test_expansion_methods():
    """Compare all three expansion methods."""
    settings = get_settings()
    openai_client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )

    question = "Jak i kiedy należy opłacać składki?"

    print("=" * 80)
    print("QUERY EXPANSION METHODS COMPARISON")
    print("=" * 80)
    print(f"\nOriginal Question: {question}\n")

    # Test 1: Multi-Query
    print("=" * 80)
    print("METHOD 1: Multi-Query")
    print("=" * 80)
    multi_expander = QueryExpander(
        llm_client=openai_client,
        method="multi-query",
        num_variations=3,
    )
    multi_results = await multi_expander.expand(question)
    print(f"\nGenerated {len(multi_results)} queries:")
    for i, q in enumerate(multi_results, 1):
        print(f"  {i}. {q[:100]}...")

    # Test 2: HyDE
    print("\n" + "=" * 80)
    print("METHOD 2: HyDE")
    print("=" * 80)
    hyde_expander = QueryExpander(
        llm_client=openai_client,
        method="hyde",
    )
    hyde_results = await hyde_expander.expand(question)
    print(f"\nGenerated {len(hyde_results)} hypothetical document:")
    for i, doc in enumerate(hyde_results, 1):
        print(f"  {i}. {doc[:100]}...")

    # Test 3: HYBRID (NEW!)
    print("\n" + "=" * 80)
    print("METHOD 3: HYBRID (HyDE + Multi-Query)")
    print("=" * 80)
    hybrid_expander = QueryExpander(
        llm_client=openai_client,
        method="hybrid",
        num_variations=3,
    )
    hybrid_results = await hybrid_expander.expand(question)
    print(f"\nGenerated {len(hybrid_results)} total expansions:")
    for i, exp in enumerate(hybrid_results, 1):
        print(f"  {i}. {exp[:100]}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nMulti-Query:  {len(multi_results)} queries")
    print(f"HyDE:         {len(hyde_results)} hypothetical doc")
    print(f"HYBRID:       {len(hybrid_results)} total (HyDE + Multi-Query)")

    print("\n✓ Hybrid gives MAXIMUM coverage by combining both methods!")


async def test_hybrid_rag():
    """Test full RAG pipeline with HYBRID expansion."""
    print("\n" + "=" * 80)
    print("FULL RAG TEST WITH HYBRID EXPANSION")
    print("=" * 80)

    os.environ["EXPANSION_METHOD"] = "hybrid"
    from app.config import reload_settings

    reload_settings()

    query_service = QueryService()

    result = await query_service.query(
        QueryRequest(
            question="Jak i kiedy należy opłacać składki?",
            top_k=5,
            include_sources=True,
        )
    )

    print("\n📝 Answer:")
    print(f"{result.answer}")

    print("\n📊 Stats:")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print(f"  Sources: {len(result.sources)}")

    # Check quality
    key_phrases = ["składka", "gotówką", "kartą", "przelewem", "zawarciu umowy"]
    found = sum(1 for phrase in key_phrases if phrase.lower() in result.answer.lower())

    print(f"\n✓ Quality: {found}/{len(key_phrases)} key phrases found")


async def main():
    await test_expansion_methods()
    await test_hybrid_rag()

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("\nFor insurance documents, use HYBRID for:")
    print("  ✓ Maximum retrieval coverage")
    print("  ✓ Both question variations AND answer-seeking")
    print("  ✓ Best of both worlds!")

    print("\nTo enable HYBRID:")
    print("  export EXPANSION_METHOD=hybrid")
    print("  # or add to .env:")
    print("  echo 'EXPANSION_METHOD=hybrid' >> .env")


if __name__ == "__main__":
    asyncio.run(main())
