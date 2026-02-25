#!/usr/bin/env python3
"""Debug vector search to see what chunks are actually retrieved."""

import asyncio
from pathlib import Path

from app.clients.openai_client import OpenAIClient
from app.config import get_settings
from app.models.chunk import EmbeddedChunk
from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.chunker import SemanticChunker
from app.storage.vector_store import VectorStore


async def main():
    """Debug vector search."""
    settings = get_settings()

    # Parse and chunk
    print("1. Parsing PDF...")
    pdf_path = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )
    parser = RAGAnythingParser()
    parsed_doc = parser.parse_pdf(pdf_path)

    print("2. Chunking...")
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(parsed_doc, doc_id="debug-doc")
    print(f"   Created {len(chunks)} chunks")

    # Generate embeddings
    print("\n3. Generating embeddings...")
    openai_client = OpenAIClient(api_key=settings.openai_api_key)
    texts = [chunk.content for chunk in chunks]
    embeddings = await openai_client.embed_batch(texts)

    embedded_chunks = [
        EmbeddedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]
    print(f"   Generated {len(embedded_chunks)} embeddings")

    # Store in vector DB
    print("\n4. Storing in vector DB...")
    vector_store = VectorStore(collection_name="debug_collection")
    await vector_store.add_embeddings(embedded_chunks, doc_id="debug-doc")
    print("   Stored successfully")

    # Query
    print("\n5. Querying...")
    question = "Co jest przedmiotem ubezpieczenia?"
    query_embedding = await openai_client.embed_text(question)

    results = await vector_store.search(query_embedding, top_k=5)

    print(f"\n{'=' * 80}")
    print(f"SEARCH RESULTS FOR: '{question}'")
    print(f"{'=' * 80}\n")

    for i, result in enumerate(results, 1):
        print(f"\n{'─' * 80}")
        print(f"Result {i}")
        print(f"Relevance: {result.relevance_score:.2%}")
        print(f"Page: {result.page}")
        print(f"Chunk ID: {result.chunk_id[:8]}...")
        print(f"{'─' * 80}")
        print(result.content[:500])
        print("...")

        # Check if this is the chunk with the answer
        has_answer = (
            "pojazd" in result.content.lower()
            and "uszkodzenia" in result.content.lower()
            and "kradzież" in result.content.lower()
            and "all risks" in result.content.lower()
        )

        if has_answer:
            print("\n⭐ THIS IS THE CHUNK WITH THE ANSWER!")

    # Find the chunk with the answer
    print(f"\n\n{'=' * 80}")
    print("FINDING THE CHUNK WITH THE ANSWER")
    print(f"{'=' * 80}\n")

    answer_chunk = None
    for i, chunk in enumerate(chunks):
        if (
            "pojazd" in chunk.content.lower()
            and "uszkodzenia" in chunk.content.lower()
            and "kradzież" in chunk.content.lower()
            and "all risks" in chunk.content.lower()
        ):
            answer_chunk = (i, chunk)
            break

    if answer_chunk:
        idx, chunk = answer_chunk
        print(f"Found at index {idx} (chunk_id: {chunk.chunk_id[:8]}...)")
        print(f"Page: {chunk.page}")
        print(f"Length: {len(chunk.content)} chars")
        print("\nContent:")
        print(chunk.content)

        # Check if it's in the results
        found_in_results = any(r.chunk_id == chunk.chunk_id for r in results)
        if found_in_results:
            rank = next(i for i, r in enumerate(results, 1) if r.chunk_id == chunk.chunk_id)
            print(f"\n✅ This chunk IS in the search results at position {rank}")
        else:
            print("\n❌ This chunk is NOT in the top 5 search results!")
            print("   This is why the answer is incomplete.")

    await openai_client.close()


if __name__ == "__main__":
    asyncio.run(main())
