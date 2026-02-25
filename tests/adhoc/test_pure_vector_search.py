"""Test pure vector search without hybrid search or reranking."""

import asyncio
import os
from pathlib import Path

from app.clients.openai_client import OpenAIClient
from app.models.document import DocumentMetadata
from app.parsers.adobe_pdf_parser import AdobePDFParser
from app.services.document_service import DocumentService
from app.storage.vector_store import VectorStore

# Set Adobe credentials
os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def main():
    # Test PDF path
    test_pdf = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    print("=" * 80)
    print("PURE VECTOR SEARCH TEST (NO HYBRID, NO RERANKING)")
    print("=" * 80)

    # Initialize Adobe PDF Parser
    print("\n1. Initialize Adobe PDF Parser")
    adobe_parser = AdobePDFParser(
        client_id=os.getenv("PDF_SERVICES_CLIENT_ID"),
        client_secret=os.getenv("PDF_SERVICES_CLIENT_SECRET"),
    )
    print("✓ Adobe PDF Parser initialized")

    # Clear database FIRST (before initializing DocumentService)
    print("\n2. Clear Vector Store")
    from app.config import get_settings

    settings_for_vs = get_settings()
    vector_store = VectorStore(
        persist_directory=settings_for_vs.vector_db_path,
        collection_name=settings_for_vs.text_collection,  # Use same collection as DocumentService
    )
    vector_store.reset()
    print(f"✓ Vector store cleared (collection: {settings_for_vs.text_collection})")

    # Initialize Document Service AFTER clearing database
    print("\n3. Initialize Document Service")
    doc_service = DocumentService(parser=adobe_parser)
    print("✓ Document Service initialized")

    # Process document
    print("\n4. Process and Index Document")
    with open(test_pdf, "rb") as f:
        file_content = f.read()

    metadata = DocumentMetadata(
        filename=test_pdf.name,
        content_type="application/pdf",
        tags=["insurance", "warta", "autocasco"],
    )

    upload_response = await doc_service.process_document(file_content, metadata)
    print(f"✓ Document indexed with ID: {upload_response.doc_id}")

    # Initialize OpenAI client for embeddings
    print("\n5. Initialize OpenAI Client")
    openai_client = OpenAIClient(
        api_key=settings_for_vs.openai_api_key,
        model=settings_for_vs.openai_model,
        embedding_model=settings_for_vs.openai_embedding_model,
    )
    print("✓ OpenAI Client initialized")

    # Test query
    question = "Jak i kiedy należy opłacać składki?"
    print(f"\n6. Query: '{question}'")

    # Generate query embedding
    print("   Generating query embedding...")
    query_embedding = await openai_client.embed_text(question)
    print(f"   ✓ Embedding generated (dim={len(query_embedding)})")

    # Search with PURE vector search
    print("\n7. Pure Vector Search (top_k=10)")
    results = await vector_store.search(
        query_embedding=query_embedding,
        top_k=10,
    )
    print(f"   ✓ Found {len(results)} results")

    # Display results
    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Page: {result.page}")
        print(f"Chunk Index: {result.chunk_index}")
        print(f"Relevance Score: {result.relevance_score:.4f}")
        print(f"Content Preview: {result.content[:300]}...")

    # Check if correct answer chunks are in results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Count how many chunks contain the answer keywords
    answer_keywords = ["gotówką", "kartą płatniczą", "przelewem bankowym"]
    toc_keyword = "Co należy do obowiązków Ubezpieczonego?"

    answer_chunks = []
    toc_chunks = []

    for i, result in enumerate(results, 1):
        content_lower = result.content.lower()
        if any(kw.lower() in content_lower for kw in answer_keywords):
            answer_chunks.append(i)
        if toc_keyword.lower() in content_lower:
            toc_chunks.append(i)

    print(f"\nChunks with ANSWER keywords: {answer_chunks}")
    print(f"Chunks with TOC text: {toc_chunks}")

    if answer_chunks:
        print(f"\n✓ SUCCESS: Found {len(answer_chunks)} chunk(s) with answer content")
    else:
        print("\n✗ FAILURE: No chunks with answer content found!")

    if toc_chunks and not answer_chunks:
        print("⚠ PROBLEM: Vector search is ranking TOC higher than actual answers")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
