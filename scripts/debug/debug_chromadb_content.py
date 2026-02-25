"""Debug what's stored in ChromaDB."""

import asyncio

from app.storage.vector_store import VectorStore


async def main():
    vector_store = VectorStore()

    print("Querying ChromaDB for chunks containing 'składk'...")

    # Get all chunks
    results = vector_store._collection.get(limit=1000, include=["documents", "metadatas"])

    total_chunks = len(results["documents"])
    print(f"\nTotal chunks in DB: {total_chunks}")

    # Filter for chunks containing "składk"
    skladka_chunks = []
    for i, doc in enumerate(results["documents"]):
        if "składk" in doc.lower():
            skladka_chunks.append({"content": doc, "metadata": results["metadatas"][i]})

    print(f"Chunks containing 'składk': {len(skladka_chunks)}\n")

    print("=" * 80)
    print("FIRST 15 CHUNKS CONTAINING 'składk':")
    print("=" * 80)

    for i, chunk in enumerate(skladka_chunks[:15], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content: {chunk['content'][:500]}...")


if __name__ == "__main__":
    asyncio.run(main())
