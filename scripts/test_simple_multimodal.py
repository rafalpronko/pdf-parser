#!/usr/bin/env python3
"""Simple test of multimodal components."""

import asyncio
from pathlib import Path

from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.multimodal import MultimodalChunker


def test_parser():
    """Test PDF parsing."""
    print("=" * 60)
    print("TEST: PDF Parsing")
    print("=" * 60)
    
    parser = RAGAnythingParser()
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if not test_pdf.exists():
        print(f"❌ Test PDF not found")
        return None
    
    result = parser.parse_pdf(test_pdf)
    
    print(f"✅ Parsed successfully:")
    print(f"   Pages: {result.num_pages}")
    print(f"   Text blocks: {len(result.text_blocks)}")
    print(f"   Images: {len(result.images)}")
    print(f"   Charts: {len(result.charts)}")
    print(f"   Tables: {len(result.tables)}")
    
    return result


def test_chunker_simple():
    """Test chunker with simple data."""
    print("\n" + "=" * 60)
    print("TEST: Multimodal Chunker")
    print("=" * 60)
    
    from app.models.parsing import ParsedDocument, TextBlock
    
    # Create simple test document
    doc = ParsedDocument(
        text_blocks=[
            TextBlock(content="This is test content.", page=0, bbox=(0, 0, 100, 100)),
            TextBlock(content="More test content here.", page=0, bbox=(0, 100, 100, 200)),
        ],
        images=[],
        charts=[],
        tables=[],
        num_pages=1,
        metadata={"test": True}
    )
    
    chunker = MultimodalChunker(chunk_size=512, chunk_overlap=50)
    text_chunks, visual_chunks, multimodal_chunks = chunker.chunk_document(doc, "test_doc")
    
    print(f"✅ Chunked successfully:")
    print(f"   Text chunks: {len(text_chunks)}")
    print(f"   Visual chunks: {len(visual_chunks)}")
    print(f"   Multimodal chunks: {len(multimodal_chunks)}")
    
    if text_chunks:
        print(f"\n   Sample chunk content: {text_chunks[0].content}")


def main():
    """Run tests."""
    print("\n🚀 Simple Multimodal Test\n")
    
    # Test parser
    parsed = test_parser()
    
    # Test chunker with simple data
    test_chunker_simple()
    
    print("\n" + "=" * 60)
    print("✅ Tests completed!")
    print("=" * 60)
    print("\n💡 To test API:")
    print("   1. Run: uv run uvicorn app.main:app --reload")
    print("   2. Visit: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
