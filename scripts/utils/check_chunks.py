#!/usr/bin/env python3
"""Check what chunks actually exist in the system."""

import asyncio

import httpx


async def check_chunks():
    """Check unique chunks in the system."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Search for different terms to see variety of chunks
        search_terms = ["składki", "płatność", "ubezpieczenie", "WARTA", "polisa"]

        all_chunks = set()

        for term in search_terms:
            print(f"\n=== Searching for '{term}' ===")
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={"question": term, "top_k": 10, "temperature": 0.1, "include_sources": True},
            )

            if response.status_code == 200:
                result = response.json()

                for source in result["sources"]:
                    chunk_content = source["chunk_content"]
                    all_chunks.add(chunk_content[:100])  # First 100 chars as identifier

        print("\n=== UNIQUE CHUNKS FOUND ===")
        print(f"Total unique chunks: {len(all_chunks)}")

        for i, chunk in enumerate(sorted(all_chunks), 1):
            print(f"\n{i}. {chunk}...")


if __name__ == "__main__":
    asyncio.run(check_chunks())
