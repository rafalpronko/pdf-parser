#!/usr/bin/env python3
"""Test script for multimodal RAG-Anything features."""

import asyncio
from pathlib import Path

from app.models.vlm.vlm_client import VLMClient
from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.multimodal import MultimodalChunker


async def test_parser():
    """Test PDF parsing with multimodal support."""
    print("\n" + "=" * 80)
    print("TEST 1: Multimodal PDF Parsing")
    print("=" * 80)
    
    parser = RAGAnythingParser()
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if not test_pdf.exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        return
    
    print(f"📄 Parsing: {test_pdf.name}")
    result = parser.parse_pdf(test_pdf)
    
    print(f"\n✅ Parsing Results:")
    print(f"   Pages: {result.num_pages}")
    print(f"   Text blocks: {len(result.text_blocks)}")
    print(f"   Images: {len(result.images)}")
    print(f"   Charts: {len(result.charts)}")
    print(f"   Tables: {len(result.tables)}")
    print(f"   Parser: {result.metadata.get('parser', 'unknown')}")
    
    return result


async def test_chunker(parsed_doc):
    """Test multimodal chunking."""
    print("\n" + "=" * 80)
    print("TEST 2: Multimodal Chunking")
    print("=" * 80)
    
    chunker = MultimodalChunker(chunk_size=512, chunk_overlap=50)
    
    print(f"📦 Chunking document...")
    text_chunks, visual_chunks, multimodal_chunks = chunker.chunk_document(
        parsed_doc, doc_id="test_doc"
    )
    
    print(f"\n✅ Chunking Results:")
    print(f"   Text chunks: {len(text_chunks)}")
    print(f"   Visual chunks: {len(visual_chunks)}")
    print(f"   Multimodal chunks: {len(multimodal_chunks)}")
    
    if text_chunks:
        print(f"\n📝 Sample text chunk:")
        sample = text_chunks[0]
        print(f"   ID: {sample.chunk_id[:8]}...")
        print(f"   Page: {sample.page}")
        print(f"   Content: {sample.content[:100]}...")
    
    if visual_chunks:
        print(f"\n🖼️  Sample visual chunk:")
        sample = visual_chunks[0]
        print(f"   ID: {sample.chunk_id[:8]}...")
        print(f"   Page: {sample.page}")
        print(f"   Type: {sample.visual_type}")
        print(f"   Image size: {len(sample.image_data)} bytes")
    
    if multimodal_chunks:
        print(f"\n🎨 Sample multimodal chunk:")
        sample = multimodal_chunks[0]
        print(f"   ID: {sample.chunk_id[:8]}...")
        print(f"   Page: {sample.page}")
        print(f"   Text: {sample.text_content[:100]}...")
        print(f"   Visual elements: {len(sample.visual_elements)}")
    
    return text_chunks, visual_chunks, multimodal_chunks


async def test_vlm(parsed_doc):
    """Test VLM client (if enabled)."""
    print("\n" + "=" * 80)
    print("TEST 3: Vision-Language Model (VLM)")
    print("=" * 80)
    
    vlm = VLMClient()
    
    if not vlm.enabled:
        print("⚠️  VLM not enabled (OpenAI API key not set or VLM disabled)")
        print("   Set OPENAI_API_KEY and ENABLE_VLM=true to test VLM features")
        return
    
    print(f"✅ VLM enabled: {vlm.provider} / {vlm.model}")
    
    # Test with first image if available
    if parsed_doc.images:
        print(f"\n🖼️  Testing image description...")
        image = parsed_doc.images[0]
        
        try:
            description = await vlm.describe_image(image)
            print(f"   Description: {description[:200]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   No images found in document")
    
    # Test with first chart if available
    if parsed_doc.charts:
        print(f"\n📊 Testing chart analysis...")
        chart = parsed_doc.charts[0]
        
        try:
            analysis = await vlm.analyze_chart(chart)
            print(f"   Chart type: {analysis.get('chart_type', 'unknown')}")
            print(f"   Description: {analysis.get('description', 'N/A')[:200]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   No charts found in document")


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🚀 RAG-Anything Multimodal Features Test")
    print("=" * 80)
    
    # Test 1: Parser
    parsed_doc = await test_parser()
    if not parsed_doc:
        return
    
    # Test 2: Chunker
    await test_chunker(parsed_doc)
    
    # Test 3: VLM (optional)
    await test_vlm(parsed_doc)
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. Set OPENAI_API_KEY to enable VLM features")
    print("   2. Run: uv run uvicorn app.main:app --reload")
    print("   3. Visit: http://localhost:8000/docs")
    print("   4. Test multimodal endpoints")
    print()


if __name__ == "__main__":
    asyncio.run(main())
