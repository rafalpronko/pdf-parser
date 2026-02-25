#!/usr/bin/env python3
"""Inspect what's actually in the retrieved chunks."""

import asyncio
from pathlib import Path

from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.chunker import SemanticChunker


async def main():
    """Inspect chunks from WARTA document."""
    pdf_path = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    print("Parsing PDF...")
    parser = RAGAnythingParser()
    parsed_doc = parser.parse_pdf(pdf_path)

    print("Chunking...")
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(parsed_doc, doc_id="test")

    print(f"\nTotal chunks: {len(chunks)}")
    print("\n" + "=" * 80)
    print("CHUNKS FROM PAGE 0 (where the answer should be):")
    print("=" * 80)

    page_0_chunks = [c for c in chunks if c.page == 0]
    print(f"\nFound {len(page_0_chunks)} chunks from page 0\n")

    for i, chunk in enumerate(page_0_chunks, 1):
        print(f"\n{'─' * 80}")
        print(f"Chunk {i} (ID: {chunk.chunk_id[:8]}..., Index: {chunk.chunk_index})")
        print(f"Length: {len(chunk.content)} chars")
        print(f"{'─' * 80}")
        print(chunk.content)

        # Check if it contains key information
        has_pojazd = "pojazd" in chunk.content.lower()
        has_uszkodzenia = (
            "uszkodzenia" in chunk.content.lower() or "uszkodzenie" in chunk.content.lower()
        )
        has_kradzież = "kradzież" in chunk.content.lower()
        has_all_risks = "all risks" in chunk.content.lower()

        print("\n🔍 Contains:")
        if has_pojazd:
            print("  ✓ pojazd")
        if has_uszkodzenia:
            print("  ✓ uszkodzenia")
        if has_kradzież:
            print("  ✓ kradzież")
        if has_all_risks:
            print("  ✓ all risks")

        if has_pojazd and has_uszkodzenia and has_kradzież:
            print("\n  ⭐ THIS CHUNK HAS THE ANSWER!")


if __name__ == "__main__":
    asyncio.run(main())
