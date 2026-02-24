#!/usr/bin/env python3
"""Find chunks with actual answers about payment."""

import asyncio
import httpx

async def find_answers():
    """Find chunks with payment information."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # Search for different payment-related terms
        search_terms = [
            "płatność",
            "wpłata", 
            "przelew",
            "bankowy",
            "termin zapłacenia",
            "rata składki",
            "przekaz pocztowy"
        ]
        
        for term in search_terms:
            print(f"\n=== Searching for '{term}' ===")
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={
                    "question": term,
                    "top_k": 3,
                    "temperature": 0.1,
                    "include_sources": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Found {len(result['sources'])} sources")
                
                for i, source in enumerate(result['sources'][:2], 1):
                    print(f"\n--- Source {i} ---")
                    print(f"Page: {source['page']}")
                    print(f"Score: {source['relevance_score']:.3f}")
                    print(f"Content: {source['chunk_content'][:300]}...")

if __name__ == "__main__":
    asyncio.run(find_answers())