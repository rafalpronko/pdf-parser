#!/usr/bin/env python3
"""Check what's actually in the vector store."""

import asyncio

import httpx


async def check_vector_store():
    """Check vector store content."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Search for the expected content
        expected_terms = [
            "gotówką kartą płatniczą przelewem bankowym",
            "przy zawarciu umowy",
            "punkt sprzedaży",
            "dostępność formy płatności",
            "pierwsza rata zapłacona",
        ]

        print("=== Checking if expected content exists in vector store ===")

        for term in expected_terms:
            print(f"\nSearching for: '{term}'")
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={"question": term, "top_k": 3, "temperature": 0.1, "include_sources": True},
            )

            if response.status_code == 200:
                result = response.json()

                if result["sources"]:
                    print(f"✓ Found {len(result['sources'])} sources")
                    for source in result["sources"][:1]:
                        print(f"  Content: {source['chunk_content'][:150]}...")
                else:
                    print("✗ No sources found")

        print("\n" + "=" * 60)
        print("=== What IS actually in the vector store? ===")

        # Get a broad sample of what's actually there
        broad_terms = ["składka", "płatność", "ubezpieczenie", "WARTA"]

        all_unique_chunks = set()

        for term in broad_terms:
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={"question": term, "top_k": 10, "temperature": 0.1, "include_sources": True},
            )

            if response.status_code == 200:
                result = response.json()
                for source in result["sources"]:
                    all_unique_chunks.add(source["chunk_content"])

        print(f"\nFound {len(all_unique_chunks)} unique chunks in vector store:")

        for i, chunk in enumerate(sorted(all_unique_chunks), 1):
            print(f"\n{i}. {chunk[:200]}...")
            if i >= 10:  # Limit output
                print(f"\n... and {len(all_unique_chunks) - 10} more chunks")
                break


if __name__ == "__main__":
    asyncio.run(check_vector_store())
