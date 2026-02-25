#!/usr/bin/env python3
"""Diagnostic script to analyze PDF parsing quality."""

import sys
from pathlib import Path

from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.chunker import SemanticChunker


def diagnose_pdf(pdf_path: Path):
    """Diagnose PDF parsing quality.

    Args:
        pdf_path: Path to PDF file
    """
    print(f"\n{'=' * 80}")
    print(f"Diagnosing: {pdf_path.name}")
    print(f"{'=' * 80}\n")

    # Parse PDF
    parser = RAGAnythingParser()
    print("Parsing PDF...")
    parsed_doc = parser.parse_pdf(pdf_path)

    # Print statistics
    print("\n📊 Parsing Statistics:")
    print(f"  Pages: {parsed_doc.num_pages}")
    print(f"  Text blocks: {len(parsed_doc.text_blocks)}")
    print(f"  Images: {len(parsed_doc.images)}")
    print(f"  Tables: {len(parsed_doc.tables)}")

    # Show text blocks
    print("\n📝 Text Blocks (first 10):")
    for i, block in enumerate(parsed_doc.text_blocks[:10], 1):
        content_preview = block.content[:100].replace("\n", " ")
        print(f"  {i}. Page {block.page}: {content_preview}...")
        if block.font_size:
            print(f"     Font size: {block.font_size}")

    if len(parsed_doc.text_blocks) > 10:
        print(f"  ... and {len(parsed_doc.text_blocks) - 10} more blocks")

    # Show full text from first page
    print("\n📄 Full text from page 0:")
    print("-" * 80)
    page_0_blocks = [b for b in parsed_doc.text_blocks if b.page == 0]
    for block in page_0_blocks:
        print(block.content)
    print("-" * 80)

    # Show images
    if parsed_doc.images:
        print("\n🖼️  Images:")
        for i, img in enumerate(parsed_doc.images[:5], 1):
            print(f"  {i}. Page {img.page}, Format: {img.format}, BBox: {img.bbox}")

    # Show tables
    if parsed_doc.tables:
        print("\n📊 Tables:")
        for i, table in enumerate(parsed_doc.tables[:3], 1):
            print(
                f"  {i}. Page {table.page}, Rows: {len(table.rows)}, Cols: {len(table.rows[0]) if table.rows else 0}"
            )
            if table.rows:
                print(f"     First row: {table.rows[0][:3]}...")

    # Test chunking
    print("\n✂️  Chunking Analysis:")
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(parsed_doc, doc_id="test")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average chunk size: {sum(len(c.content) for c in chunks) / len(chunks):.0f} chars")

    # Show first few chunks
    print("\n  First 3 chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        content_preview = chunk.content[:150].replace("\n", " ")
        print(f"  {i}. Page {chunk.page}, Length: {len(chunk.content)}")
        print(f"     {content_preview}...")

    # Search for specific content
    print("\n🔍 Searching for 'przedmiotem ubezpieczenia':")
    found = False
    for i, block in enumerate(parsed_doc.text_blocks):
        if "przedmiotem ubezpieczenia" in block.content.lower():
            print(f"  ✓ Found in block {i + 1} (page {block.page}):")
            print(f"    {block.content[:200]}...")
            found = True

    if not found:
        print("  ✗ Not found in text blocks")
        print("\n  Searching in chunks:")
        for i, chunk in enumerate(chunks):
            if "przedmiotem ubezpieczenia" in chunk.content.lower():
                print(f"  ✓ Found in chunk {i + 1} (page {chunk.page}):")
                print(f"    {chunk.content[:200]}...")
                found = True
                break

    if not found:
        print("  ✗ Not found anywhere - possible OCR or parsing issue")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_pdf.py <pdf_file>")
        print("\nExample:")
        print(
            "  python scripts/diagnose_pdf.py 'data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf'"
        )
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    diagnose_pdf(pdf_path)


if __name__ == "__main__":
    main()
