#!/usr/bin/env python3
"""Search for specific payment method information."""

import asyncio

import httpx


async def search_payment_methods():
    """Search for payment method information."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Search for specific payment terms
        search_terms = [
            "gotówka",
            "karta płatnicza",
            "przy zawarciu umowy",
            "pierwsza rata",
            "punkt sprzedaży",
            "forma płatności",
            "dostępność",
            "zawarcie umowy",
        ]

        for term in search_terms:
            print(f"\n=== Searching for '{term}' ===")
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={"question": term, "top_k": 5, "temperature": 0.1, "include_sources": True},
            )

            if response.status_code == 200:
                result = response.json()

                if result["sources"]:
                    print(f"Found {len(result['sources'])} sources")
                    for i, source in enumerate(result["sources"][:2], 1):
                        print(f"\n--- Source {i} ---")
                        print(f"Content: {source['chunk_content'][:200]}...")
                else:
                    print("No sources found")


if __name__ == "__main__":
    asyncio.run(search_payment_methods())
