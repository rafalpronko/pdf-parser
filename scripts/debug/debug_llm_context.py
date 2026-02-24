#!/usr/bin/env python3
"""Debug script to see what context LLM receives."""

import asyncio
import httpx

async def debug_llm_context():
    """Debug what context LLM receives."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Query with detailed logging
        print("=== Full Query Response ===")
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={
                "question": "Jak i kiedy należy opłacać składki?",
                "top_k": 3,
                "temperature": 0.1,
                "include_sources": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Answer: {result['answer']}")
            print(f"\nNumber of sources: {len(result['sources'])}")
            
            print("\n=== FULL CONTEXT THAT LLM SEES ===")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n[Source {i}] (Page {source['page']}, Relevance: {source['relevance_score']:.2f})")
                print(f"Full content: {source['chunk_content']}")
                print("-" * 80)

if __name__ == "__main__":
    asyncio.run(debug_llm_context())