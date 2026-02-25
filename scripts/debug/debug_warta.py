#!/usr/bin/env python3
"""Debug script to check what's happening with Warta document."""

import asyncio

import httpx


async def debug_warta():
    """Debug Warta document processing."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Get document info
        doc_id = "99d52d6b-4b33-4160-8b6a-d3627a21da0d"

        print("=== Document Info ===")
        response = await client.get(f"http://localhost:8000/api/v1/documents/{doc_id}")
        if response.status_code == 200:
            doc_info = response.json()
            print(f"Filename: {doc_info['filename']}")
            print(f"Pages: {doc_info['num_pages']}")
            print(f"Chunks: {doc_info['num_chunks']}")

        # Search for chunks containing "składki"
        print("\n=== Searching for 'składki' ===")
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={"question": "składki", "top_k": 10, "temperature": 0.1, "include_sources": True},
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result['sources'])} sources")

            for i, source in enumerate(result["sources"][:5], 1):
                print(f"\n--- Source {i} ---")
                print(f"Page: {source['page']}")
                print(f"Score: {source['relevance_score']:.3f}")
                print(f"Content: {source['chunk_content'][:200]}...")

        # Search for chunks containing "opłacać"
        print("\n=== Searching for 'opłacać' ===")
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={"question": "opłacać", "top_k": 10, "temperature": 0.1, "include_sources": True},
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result['sources'])} sources")

            for i, source in enumerate(result["sources"][:5], 1):
                print(f"\n--- Source {i} ---")
                print(f"Page: {source['page']}")
                print(f"Score: {source['relevance_score']:.3f}")
                print(f"Content: {source['chunk_content'][:200]}...")


if __name__ == "__main__":
    asyncio.run(debug_warta())
