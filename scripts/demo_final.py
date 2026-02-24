#!/usr/bin/env python3
"""Final demo showing all RAG-Anything multimodal components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_parsing():
    """Demo PDF parsing."""
    print("\n" + "=" * 70)
    print("  1. PDF PARSING (MinerU/pypdf)")
    print("=" * 70)
    
    from app.parsers.rag_anything_parser import RAGAnythingParser
    
    parser = RAGAnythingParser()
    print(f"✓ Parser initialized (MinerU: {parser.mineru_available})")
    
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    if test_pdf.exists():
        parsed = parser.parse_pdf(test_pdf)
        print(f"✓ Parsed: {parsed.num_pages} pages, {len(parsed.text_blocks)} text blocks")
        print(f"  Images: {len(parsed.images)}, Charts: {len(parsed.charts)}, Tables: {len(parsed.tables)}")
        return parsed
    else:
        print("✗ Test PDF not found")
        return None


def demo_chunking(parsed_doc):
    """Demo multimodal chunking."""
    print("\n" + "=" * 70)
    print("  2. MULTIMODAL CHUNKING")
    print("=" * 70)
    
    from app.models.parsing import ParsedDocument
    from app.processing.multimodal import MultimodalChunker
    
    # Use only first text block for demo
    limited = ParsedDocument(
        text_blocks=parsed_doc.text_blocks[:1],
        images=parsed_doc.images[:1] if parsed_doc.images else [],
        charts=parsed_doc.charts[:1] if parsed_doc.charts else [],
        tables=[],
        num_pages=parsed_doc.num_pages,
        metadata=parsed_doc.metadata
    )
    
    chunker = MultimodalChunker(chunk_size=512, chunk_overlap=50)
    text_chunks, visual_chunks, multimodal_chunks = chunker.chunk_document(
        limited, doc_id="demo"
    )
    
    print(f"✓ Created chunks:")
    print(f"  Text: {len(text_chunks)}, Visual: {len(visual_chunks)}, Multimodal: {len(multimodal_chunks)}")
    
    if text_chunks:
        print(f"\n  Sample text chunk (ID: {text_chunks[0].chunk_id[:12]}...):")
        print(f"  '{text_chunks[0].content[:100]}...'")
    
    return text_chunks, visual_chunks


def demo_models():
    """Demo Pydantic models."""
    print("\n" + "=" * 70)
    print("  3. PYDANTIC MODELS")
    print("=" * 70)
    
    from app.models import (
        TextChunk, VisualChunk, MultimodalChunk,
        ChartBlock, QueryRequest, MultimodalQueryResponse
    )
    
    print("✓ Available models:")
    print("  - TextChunk, VisualChunk, MultimodalChunk")
    print("  - ChartBlock, ImageBlock, TableBlock")
    print("  - QueryRequest, MultimodalQueryResponse")
    
    # Create sample query
    query = QueryRequest(
        question="What is covered by insurance?",
        top_k=5,
        include_visual=True,
        modality_filter="text"
    )
    print(f"\n  Sample query: {query.question}")
    print(f"  Modality filter: {query.modality_filter}")


def demo_vlm():
    """Demo VLM client."""
    print("\n" + "=" * 70)
    print("  4. VISION-LANGUAGE MODEL (VLM)")
    print("=" * 70)
    
    from app.models.vlm import VLMClient
    from app.config import get_settings
    
    settings = get_settings()
    vlm = VLMClient()
    
    print(f"✓ VLM Client initialized")
    print(f"  Provider: {vlm.provider}")
    print(f"  Model: {vlm.model}")
    print(f"  Enabled: {vlm.enabled}")
    
    if not vlm.enabled:
        print("\n  ⚠️  VLM requires OPENAI_API_KEY in .env")
    else:
        print("\n  ✓ VLM ready for:")
        print("    - describe_image()")
        print("    - analyze_chart()")
        print("    - extract_table_from_image()")
        print("    - answer_visual_question()")


def demo_embedder():
    """Demo multimodal embedder."""
    print("\n" + "=" * 70)
    print("  5. MULTIMODAL EMBEDDINGS")
    print("=" * 70)
    
    from app.processing.multimodal import MultimodalEmbedder
    
    embedder = MultimodalEmbedder()
    
    print(f"✓ Embedder initialized")
    print(f"  Text embedder (OpenAI): {embedder.text_enabled}")
    print(f"  Vision embedder (CLIP): {embedder.vision_enabled}")
    
    if not embedder.text_enabled:
        print("\n  ⚠️  Text embeddings require OPENAI_API_KEY")
    if not embedder.vision_enabled:
        print("  ⚠️  Vision embeddings require CLIP model")
    
    if embedder.text_enabled or embedder.vision_enabled:
        print("\n  ✓ Ready for:")
        if embedder.text_enabled:
            print("    - embed_text_chunks()")
        if embedder.vision_enabled:
            print("    - embed_visual_chunks()")
        if embedder.text_enabled and embedder.vision_enabled:
            print("    - embed_multimodal_chunks()")


def main():
    """Run complete demo."""
    print("\n" + "=" * 70)
    print("  🚀 RAG-ANYTHING MULTIMODAL SYSTEM DEMO")
    print("=" * 70)
    
    # Configuration
    from app.config import get_settings
    settings = get_settings()
    print(f"\nConfiguration:")
    print(f"  API: {settings.api_title} v{settings.api_version}")
    print(f"  VLM: {settings.vlm_provider} ({settings.vlm_model})")
    print(f"  Vision Encoder: {settings.vision_encoder}")
    
    # Run demos
    parsed_doc = demo_parsing()
    
    if parsed_doc:
        demo_chunking(parsed_doc)
    
    demo_models()
    demo_vlm()
    demo_embedder()
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ DEMO COMPLETE")
    print("=" * 70)
    print("\n🎉 RAG-Anything Multimodal System is ready!")
    print("\nImplemented components:")
    print("  ✓ Multimodal PDF parsing (text, images, charts, tables)")
    print("  ✓ Multimodal chunking (text, visual, multimodal)")
    print("  ✓ VLM client for visual understanding")
    print("  ✓ Multimodal embeddings (text + vision)")
    print("  ✓ Pydantic models for all data types")
    print("\nNext steps:")
    print("  1. Set OPENAI_API_KEY in .env for full functionality")
    print("  2. Implement vector store for multimodal embeddings")
    print("  3. Implement cross-modal retrieval")
    print("  4. Add FastAPI endpoints")
    print("\n")


if __name__ == "__main__":
    main()
