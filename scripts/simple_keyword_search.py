#!/usr/bin/env python3
"""Simple keyword-based search as fallback for semantic search."""

import asyncio
from pathlib import Path

from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.chunker import SemanticChunker


async def keyword_search(question: str, chunks: list, top_k: int = 5):
    """Simple keyword search in chunks.

    Args:
        question: Search question
        chunks: List of chunks to search
        top_k: Number of results to return

    Returns:
        List of (chunk, score) tuples
    """
    # Extract keywords from question
    keywords = question.lower().split()
    keywords = [k for k in keywords if len(k) > 3]  # Filter short words

    results = []
    for chunk in chunks:
        content_lower = chunk.content.lower()

        # Count keyword matches
        score = sum(1 for kw in keywords if kw in content_lower)

        # Bonus for exact phrase match
        if question.lower() in content_lower:
            score += 10

        if score > 0:
            results.append((chunk, score))

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


async def main():
    """Test keyword search."""
    pdf_path = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    print("Parsing and chunking...")
    parser = RAGAnythingParser()
    parsed_doc = parser.parse_pdf(pdf_path)

    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_with_structure(parsed_doc, doc_id="test")

    print(f"Created {len(chunks)} chunks\n")

    # Test keyword search
    question = "Co jest przedmiotem ubezpieczenia?"
    print(f"Question: {question}\n")
    print("=" * 80)
    print("KEYWORD SEARCH RESULTS:")
    print("=" * 80)

    results = await keyword_search(question, chunks, top_k=5)

    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score}")
        print(f"   Page: {chunk.page}, Index: {chunk.chunk_index}")
        print("   Content preview:")
        print(f"   {chunk.content[:200]}...")

        # Check if this has the answer
        has_answer = (
            "uszkodzenia" in chunk.content.lower()
            and "kradzież" in chunk.content.lower()
            and "all risks" in chunk.content.lower()
        )
        if has_answer:
            print("\n   ⭐ THIS CHUNK HAS THE FULL ANSWER!")


if __name__ == "__main__":
    asyncio.run(main())
