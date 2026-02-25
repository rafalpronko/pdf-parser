"""Compare HyDE vs Multi-Query expansion methods."""

import asyncio
import os

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.retrieval.query_expansion import QueryExpander

# Set credentials
os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def main():
    settings = get_settings()
    openai_client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )

    question = "Jak i kiedy należy opłacać składki?"

    print("=" * 80)
    print("QUERY EXPANSION: HYDE vs MULTI-QUERY")
    print("=" * 80)
    print(f"\nOriginal Question: {question}\n")

    # Test Multi-Query
    print("=" * 80)
    print("METHOD 1: Multi-Query (Multiple Question Variations)")
    print("=" * 80)

    multi_query_expander = QueryExpander(
        llm_client=openai_client,
        method="multi-query",
        num_variations=3,
    )

    multi_variations = await multi_query_expander.expand(question)

    print(f"\nGenerated {len(multi_variations)} variations:")
    for i, var in enumerate(multi_variations, 1):
        print(f"  {i}. {var}")

    # Test HyDE
    print("\n" + "=" * 80)
    print("METHOD 2: HyDE (Hypothetical Document)")
    print("=" * 80)

    hyde_expander = QueryExpander(
        llm_client=openai_client,
        method="hyde",
    )

    hyde_result = await hyde_expander.expand(question)

    print("\nGenerated hypothetical document:")
    print(f"  {hyde_result[0]}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print("\nMulti-Query:")
    print("  ✓ Generates multiple question variations")
    print("  ✓ Good for capturing different query intents")
    print("  ✓ Each variation searched independently")
    print(f"  - Generated {len(multi_variations)} variations")

    print("\nHyDE:")
    print("  ✓ Generates hypothetical answer")
    print("  ✓ Searches for documents similar to the answer")
    print("  ✓ Works better for 'answer-seeking' queries")
    print("  - Generated 1 hypothetical document")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print("\nFor your insurance document queries:")
    print("  → HyDE may work BETTER because:")
    print("    - Questions seek specific factual answers")
    print("    - Documents contain answer-like text (not questions)")
    print("    - HyDE aligns query embedding space with answer space")

    print("\nTo enable HyDE:")
    print("  export EXPANSION_METHOD=hyde")
    print("  # or add to .env file")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
