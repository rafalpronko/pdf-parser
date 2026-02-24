#!/usr/bin/env python3
"""Quick demo of multimodal RAG components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.parsers.rag_anything_parser import RAGAnythingParser


def main():
    """Run quick demo."""
    print("=" * 80)
    print("  RAG-Anything Multimodal System - Quick Demo")
    print("=" * 80)
    
    # Configuration
    print("\n1. Configuration:")
    settings = get_settings()
    print(f"   ✓ API Title: {settings.api_title}")
    print(f"   ✓ VLM Enabled: {settings.enable_vlm}")
    print(f"   ✓ Multimodal Chunking: {settings.enable_multimodal_chunking}")
    
    # Parser
    print("\n2. PDF Parser:")
    parser = RAGAnythingParser()
    print(f"   ✓ Parser initialized")
    print(f"   ✓ MinerU available: {parser.mineru_available}")
    
    # Parse test PDF
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if test_pdf.exists():
        print(f"\n3. Parsing PDF: {test_pdf.name}")
        parsed = parser.parse_pdf(test_pdf)
        print(f"   ✓ Pages: {parsed.num_pages}")
        print(f"   ✓ Text blocks: {len(parsed.text_blocks)}")
        print(f"   ✓ Images: {len(parsed.images)}")
        print(f"   ✓ Charts: {len(parsed.charts)}")
        print(f"   ✓ Tables: {len(parsed.tables)}")
        
        if parsed.text_blocks:
            print(f"\n4. Sample text:")
            print(f"   {parsed.text_blocks[0].content[:200]}...")
    else:
        print(f"\n   ✗ Test PDF not found")
    
    # Components available
    print("\n5. Available components:")
    try:
        from app.processing.multimodal import MultimodalChunker
        print(f"   ✓ MultimodalChunker")
    except:
        print(f"   ✗ MultimodalChunker")
    
    try:
        from app.processing.multimodal import MultimodalEmbedder
        print(f"   ✓ MultimodalEmbedder")
    except:
        print(f"   ✗ MultimodalEmbedder")
    
    try:
        from app.models.vlm import VLMClient
        print(f"   ✓ VLMClient")
    except:
        print(f"   ✗ VLMClient")
    
    print("\n" + "=" * 80)
    print("  Demo complete! ✓")
    print("=" * 80)
    print("\nKey features implemented:")
    print("  • Multimodal PDF parsing (text, images, charts, tables)")
    print("  • Multimodal chunking (text, visual, multimodal)")
    print("  • VLM client for image understanding")
    print("  • Multimodal embeddings (text + vision)")
    print("\nTo use the full system:")
    print("  1. Set OPENAI_API_KEY in .env for VLM and embeddings")
    print("  2. Install magic-pdf for MinerU support")
    print("  3. Run: uv run python scripts/demo_multimodal_rag.py")


if __name__ == "__main__":
    main()
