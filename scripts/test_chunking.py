#!/usr/bin/env python3
"""Test multimodal chunking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.multimodal import MultimodalChunker


def main():
    """Test chunking."""
    print("Testing Multimodal Chunking...")

    # Parse PDF
    test_pdf = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )
    parser = RAGAnythingParser()
    parsed = parser.parse_pdf(test_pdf)

    print("\nParsed document:")
    print(f"  - Text blocks: {len(parsed.text_blocks)}")
    print(f"  - Images: {len(parsed.images)}")
    print(f"  - Charts: {len(parsed.charts)}")

    # Chunk document (limit to first 2 blocks for speed)
    from app.models.parsing import ParsedDocument

    limited = ParsedDocument(
        text_blocks=parsed.text_blocks[:2],
        images=parsed.images,
        charts=parsed.charts,
        tables=[],
        num_pages=parsed.num_pages,
        metadata=parsed.metadata,
    )

    chunker = MultimodalChunker(chunk_size=512, chunk_overlap=50)
    text_chunks, visual_chunks, multimodal_chunks = chunker.chunk_document(
        limited, doc_id="test_doc"
    )

    print("\nChunking results:")
    print(f"  - Text chunks: {len(text_chunks)}")
    print(f"  - Visual chunks: {len(visual_chunks)}")
    print(f"  - Multimodal chunks: {len(multimodal_chunks)}")

    if text_chunks:
        print("\nFirst text chunk:")
        print(f"  - ID: {text_chunks[0].chunk_id[:16]}...")
        print(f"  - Page: {text_chunks[0].page}")
        print(f"  - Length: {len(text_chunks[0].content)} chars")
        print(f"  - Preview: {text_chunks[0].content[:100]}...")

    print("\n✓ Chunking test complete!")


if __name__ == "__main__":
    main()
