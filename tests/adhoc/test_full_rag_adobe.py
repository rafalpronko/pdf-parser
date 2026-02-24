"""Test full RAG pipeline with Adobe PDF Extract API."""

import asyncio
import os
from pathlib import Path

from app.models.document import DocumentMetadata
from app.models.query import QueryRequest
from app.parsers.adobe_pdf_parser import AdobePDFParser
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

# Set Adobe credentials
os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def main():
    # Test PDF path
    test_pdf = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    print("=" * 80)
    print("FULL RAG PIPELINE TEST WITH ADOBE PDF EXTRACT API")
    print("=" * 80)
    print(f"\nTest file: {test_pdf.name}")
    print(f"File exists: {test_pdf.exists()}")

    # Initialize Adobe PDF Parser
    print("\n" + "=" * 80)
    print("STEP 1: Initialize Adobe PDF Parser")
    print("=" * 80)
    adobe_parser = AdobePDFParser(
        client_id=os.getenv("PDF_SERVICES_CLIENT_ID"),
        client_secret=os.getenv("PDF_SERVICES_CLIENT_SECRET"),
    )
    print("✓ Adobe PDF Parser initialized")

    # Initialize Document Service with Adobe parser
    print("\n" + "=" * 80)
    print("STEP 2: Initialize Document Service")
    print("=" * 80)
    doc_service = DocumentService(parser=adobe_parser)
    print("✓ Document Service initialized with Adobe PDF Parser")

    # Process document
    print("\n" + "=" * 80)
    print("STEP 3: Process and Index Document")
    print("=" * 80)

    with open(test_pdf, "rb") as f:
        file_content = f.read()

    metadata = DocumentMetadata(
        filename=test_pdf.name,
        content_type="application/pdf",
        tags=["insurance", "warta", "autocasco"],
    )

    print(f"Processing document: {test_pdf.name}")
    upload_response = await doc_service.process_document(file_content, metadata)
    print(f"\n✓ Document processed successfully!")
    print(f"  Document ID: {upload_response.doc_id}")
    print(f"  Status: {upload_response.status}")
    print(f"  Message: {upload_response.message}")

    # Initialize Query Service with DocumentService
    print("\n" + "=" * 80)
    print("STEP 4: Initialize Query Service")
    print("=" * 80)
    query_service = QueryService(document_service=doc_service)
    print("✓ Query Service initialized with DocumentService")

    # Execute query
    print("\n" + "=" * 80)
    print("STEP 5: Execute RAG Query")
    print("=" * 80)

    question = "Jak i kiedy należy opłacać składki?"
    print(f"\nQuery: {question}")
    print("\nProcessing...")

    query_request = QueryRequest(
        question=question,
        top_k=10,
        include_sources=True,
        include_visual=False,
    )

    result = await query_service.query(query_request)

    # Display results
    print("\n" + "=" * 80)
    print("STEP 6: Results")
    print("=" * 80)

    print(f"\n📝 Answer:")
    print(f"{result.answer}")

    print(f"\n📊 Metadata:")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    print(f"  - Modalities used: {result.modalities_used}")
    print(f"  - Number of sources: {len(result.sources)}")
    print(f"  - Number of visual sources: {len(result.visual_sources)}")

    print(f"\n📚 Source References:")
    for i, source in enumerate(result.sources[:5], 1):
        print(f"\n  Source {i}:")
        print(f"    Filename: {source.filename}")
        print(f"    Page: {source.page}")
        print(f"    Modality: {source.modality}")
        print(f"    Relevance: {source.relevance_score:.4f}")
        print(f"    Content: {source.chunk_content[:200]}...")

    # Compare with expected answer
    print("\n" + "=" * 80)
    print("STEP 7: Verification")
    print("=" * 80)

    expected_answer = """Składka może być płatna gotówką, kartą płatniczą lub przelewem bankowym, w zależności od dostępności danej formy płatności w punkcie sprzedaży.
Składka lub jej pierwsza rata powinna być zapłacona przy zawarciu umowy lub później, zgodnie z ustaleniami w umowie ubezpieczenia. Wysokość rat składki i terminy płatności są określone w polisie."""

    print(f"\n Expected answer:")
    print(f"{expected_answer}")

    print(f"\n✓ Generated answer:")
    print(f"{result.answer}")

    # Check if key phrases are present
    key_phrases = [
        "składka",
        "gotówką",
        "kartą płatniczą",
        "przelewem bankowym",
        "zawarciu umowy",
        "polisie",
    ]

    print(f"\n📋 Key phrases check:")
    for phrase in key_phrases:
        present = phrase.lower() in result.answer.lower()
        symbol = "✓" if present else "✗"
        print(f"  {symbol} '{phrase}': {'found' if present else 'NOT FOUND'}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
