"""Test RAG with enhancements disabled to isolate the problem."""

import asyncio
import os
from pathlib import Path

from app.models.document import DocumentMetadata
from app.models.query import QueryRequest
from app.parsers.adobe_pdf_parser import AdobePDFParser
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.storage.vector_store import VectorStore

# Set Adobe credentials
os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"

# DISABLE ALL ENHANCEMENTS
os.environ["ENABLE_HYBRID_SEARCH"] = "false"
os.environ["ENABLE_QUERY_EXPANSION"] = "false"
os.environ["ENABLE_RERANKING"] = "false"


async def main():
    # Test PDF path
    test_pdf = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    print("=" * 80)
    print("RAG TEST WITH ALL ENHANCEMENTS DISABLED")
    print("=" * 80)
    print("\nSettings:")
    print("  - Hybrid Search: DISABLED")
    print("  - Query Expansion: DISABLED")
    print("  - Reranking: DISABLED")
    print("  - Mode: Pure Vector Search Only")

    # Initialize Adobe PDF Parser
    print("\n" + "=" * 80)
    print("STEP 1: Initialize Adobe PDF Parser")
    print("=" * 80)
    adobe_parser = AdobePDFParser(
        client_id=os.getenv("PDF_SERVICES_CLIENT_ID"),
        client_secret=os.getenv("PDF_SERVICES_CLIENT_SECRET"),
    )
    print("✓ Adobe PDF Parser initialized")

    # Clear database
    print("\n" + "=" * 80)
    print("STEP 2: Clear Vector Store")
    print("=" * 80)
    from app.config import reload_settings

    settings = reload_settings()  # Reload to pick up env changes

    vector_store = VectorStore(
        persist_directory=settings.vector_db_path,
        collection_name=settings.text_collection,
    )
    vector_store.reset()
    print(f"✓ Vector store cleared (collection: {settings.text_collection})")

    # Initialize Document Service
    print("\n" + "=" * 80)
    print("STEP 3: Initialize Document Service")
    print("=" * 80)
    doc_service = DocumentService(parser=adobe_parser)
    print("✓ Document Service initialized")

    # Process document
    print("\n" + "=" * 80)
    print("STEP 4: Process and Index Document")
    print("=" * 80)
    with open(test_pdf, "rb") as f:
        file_content = f.read()

    metadata = DocumentMetadata(
        filename=test_pdf.name,
        content_type="application/pdf",
        tags=["insurance", "warta", "autocasco"],
    )

    upload_response = await doc_service.process_document(file_content, metadata)
    print(f"✓ Document indexed with ID: {upload_response.doc_id}")

    # Initialize Query Service
    print("\n" + "=" * 80)
    print("STEP 5: Initialize Query Service")
    print("=" * 80)
    query_service = QueryService()
    print("✓ Query Service initialized")
    print(f"   - Hybrid search: {settings.enable_hybrid_search}")
    print(f"   - Query expansion: {settings.enable_query_expansion}")
    print(f"   - Reranking: {settings.enable_reranking}")

    # Execute query
    print("\n" + "=" * 80)
    print("STEP 6: Execute RAG Query")
    print("=" * 80)

    question = "Jak i kiedy należy opłacać składki?"
    print(f"\nQuery: {question}")
    print("Processing...\n")

    query_request = QueryRequest(
        question=question,
        top_k=5,
        include_sources=True,
        include_visual=False,
    )

    result = await query_service.query(query_request)

    # Display results
    print("\n" + "=" * 80)
    print("STEP 7: Results")
    print("=" * 80)

    print("\n📝 Answer:")
    print(f"{result.answer}")

    print("\n📊 Metadata:")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    print(f"  - Number of sources: {len(result.sources)}")

    print("\n📚 Source References:")
    for i, source in enumerate(result.sources, 1):
        print(f"\n  Source {i}:")
        print(f"    Filename: {source.filename}")
        print(f"    Page: {source.page}")
        print(f"    Modality: {source.modality}")
        print(f"    Relevance: {source.relevance_score:.4f}")
        print(f"    Content: {source.chunk_content[:200]}...")

    # Verification
    print("\n" + "=" * 80)
    print("STEP 8: Verification")
    print("=" * 80)

    expected_answer = """Składka może być płatna gotówką, kartą płatniczą lub przelewem bankowym, w zależności od dostępności danej formy płatności w punkcie sprzedaży.
Składka lub jej pierwsza rata powinna być zapłacona przy zawarciu umowy lub później, zgodnie z ustaleniami w umowie ubezpieczenia. Wysokość rat składki i terminy płatności są określone w polisie."""

    # Check if key phrases are present
    key_phrases = [
        "składka",
        "gotówką",
        "kartą płatniczą",
        "przelewem bankowym",
        "zawarciu umowy",
        "polisie",
    ]

    print("\n📋 Key phrases check:")
    found_count = 0
    for phrase in key_phrases:
        present = phrase.lower() in result.answer.lower()
        symbol = "✓" if present else "✗"
        print(f"  {symbol} '{phrase}': {'found' if present else 'NOT FOUND'}")
        if present:
            found_count += 1

    print(
        f"\n{'✓ SUCCESS' if found_count >= 4 else '✗ FAILURE'}: Found {found_count}/{len(key_phrases)} key phrases"
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
