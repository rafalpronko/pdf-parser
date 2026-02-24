#!/usr/bin/env python3
"""Debug the specific problematic query."""

import asyncio
import httpx

async def debug_specific_query():
    """Debug why the specific query fails."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # Test different variations of the question
        questions = [
            "Jak i kiedy należy opłacać składki?",
            "Jak opłacać składki?",
            "Kiedy opłacać składki?",
            "terminy płatności składki",
            "sposób płacenia składki",
            "raty składki",
            "płacenie składki"
        ]
        
        for question in questions:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print('='*60)
            
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={
                    "question": question,
                    "top_k": 3,
                    "temperature": 0.1,
                    "include_sources": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Answer: {result['answer'][:100]}...")
                
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. Score: {source['relevance_score']:.3f}")
                    print(f"   Content: {source['chunk_content'][:80]}...")

if __name__ == "__main__":
    asyncio.run(debug_specific_query())