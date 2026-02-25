#!/usr/bin/env python3
"""Test integration with Warta document."""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.document import DocumentMetadata
from app.models.query import QueryRequest
from app.services.document_service import DocumentService
from app.services.query_service import QueryService


async def main():
    """Test Warta document processing and querying."""

    print("=" * 80)
    print("🧪 Testing RAG System with Warta Document")
    print("=" * 80)

    # Initialize services
    print("\n📦 Initializing services...")
    doc_service = DocumentService()
    query_service = QueryService(document_service=doc_service)

    # Path to Warta document
    warta_path = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    if not warta_path.exists():
        print(f"❌ Document not found: {warta_path}")
        return

    print(f"✅ Found document: {warta_path.name}")

    # Read document
    print("\n📄 Reading document...")
    with open(warta_path, "rb") as f:
        file_content = f.read()

    print(f"✅ Read {len(file_content):,} bytes")

    # Process document
    print("\n⚙️  Processing document (parsing, chunking, embedding, indexing)...")
    print("   This may take a minute...")

    metadata = DocumentMetadata(
        filename=warta_path.name,
        content_type="application/pdf",
        tags=["insurance", "warta", "autocasco"],
        description="Warta AutoCasco Standard insurance document",
    )

    try:
        result = await doc_service.process_document(file_content=file_content, metadata=metadata)

        print("\n✅ Document processed successfully!")
        print(f"   Doc ID: {result.doc_id}")
        print(f"   Status: {result.status}")
        print(f"   Created: {result.created_at}")

        # Get document info
        doc_info = await doc_service.get_document(result.doc_id)
        print("\n📊 Document Statistics:")
        print(f"   Pages: {doc_info.num_pages}")
        print(f"   Chunks: {doc_info.num_chunks}")
        print(f"   Size: {doc_info.file_size:,} bytes")

    except Exception as e:
        print(f"\n❌ Error processing document: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test queries
    print("\n" + "=" * 80)
    print("🔍 Testing Queries")
    print("=" * 80)

    queries = [
        "Jakiego rodzaju jest to ubezpieczenie?",
        "Co jest przedmiotem ubezpieczenia?",
        "Czego nie obejmuje ubezpieczenie?",
    ]

    for i, question in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {question}")
        print("-" * 80)

        try:
            request = QueryRequest(
                question=question, top_k=5, temperature=0.7, include_sources=True
            )

            response = await query_service.query(request)

            print("\n💬 Answer:")
            print(f"   {response.answer}")

            print(f"\n📚 Sources ({len(response.sources)}):")
            for j, source in enumerate(response.sources[:3], 1):
                print(f"   {j}. Page {source.page} (score: {source.relevance_score:.3f})")
                print(f"      {source.chunk_content[:100]}...")

            print(f"\n⏱️  Processing time: {response.processing_time:.2f}s")

        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback

            traceback.print_exc()

    # Cleanup
    print("\n" + "=" * 80)
    print("🧹 Cleanup")
    print("=" * 80)

    await doc_service.close()
    await query_service.close()

    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
