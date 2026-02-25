#!/usr/bin/env python3
"""Debug retrieval to see what chunks are being retrieved."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.query_service import QueryService


async def main():
    """Test retrieval."""

    print("🔍 Testing Retrieval")
    print("=" * 80)

    # Initialize with all enhancements disabled for debugging
    query_service = QueryService()

    # Test query
    question = "Co jest przedmiotem ubezpieczenia?"

    print(f"\n📝 Query: {question}")
    print("-" * 80)

    # Get embedding
    embedding = await query_service._embed_query(question)
    print(f"✅ Generated embedding (dim: {len(embedding)})")

    # Try vector search
    print("\n🔍 Vector Search Results:")
    vector_results = await query_service.retrieve_context(query_embedding=embedding, top_k=10)

    for i, result in enumerate(vector_results[:5], 1):
        print(f"\n{i}. Page {result.page}, Score: {result.relevance_score:.4f}")
        print("   Content (first 150 chars):")
        print(f"   {result.content[:150]}...")

    # Try hybrid search if available
    if query_service.hybrid_search:
        print("\n\n🔀 Hybrid Search Results:")
        hybrid_results = await query_service.hybrid_search.search(
            query=question, query_embedding=embedding, top_k=10
        )

        for i, result in enumerate(hybrid_results[:5], 1):
            print(f"\n{i}. Page {result.page}, Score: {result.relevance_score:.4f}")
            print("   Content (first 150 chars):")
            print(f"   {result.content[:150]}...")

    await query_service.close()


if __name__ == "__main__":
    asyncio.run(main())
