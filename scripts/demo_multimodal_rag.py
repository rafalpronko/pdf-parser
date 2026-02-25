#!/usr/bin/env python3
"""Demo script showing multimodal RAG-Anything capabilities."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.vlm.vlm_client import VLMClient
from app.parsers.rag_anything_parser import RAGAnythingParser
from app.processing.multimodal import MultimodalChunker, MultimodalEmbedder


async def main():
    """Demonstrate multimodal RAG capabilities."""
    print("=" * 80)
    print("RAG-Anything Multimodal System Demo")
    print("=" * 80)

    # 1. Parse PDF with multimodal extraction
    print("\n1. Parsing PDF with RAG-Anything MinerU...")
    parser = RAGAnythingParser()

    test_pdf = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    if not test_pdf.exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        return

    parsed_doc = parser.parse_pdf(test_pdf)
    print("✅ Parsed document:")
    print(f"   - Pages: {parsed_doc.num_pages}")
    print(f"   - Text blocks: {len(parsed_doc.text_blocks)}")
    print(f"   - Images: {len(parsed_doc.images)}")
    print(f"   - Charts: {len(parsed_doc.charts)}")
    print(f"   - Tables: {len(parsed_doc.tables)}")

    # 2. Create multimodal chunks
    print("\n2. Creating multimodal chunks...")
    chunker = MultimodalChunker(chunk_size=512, chunk_overlap=50)

    text_chunks, visual_chunks, multimodal_chunks = chunker.chunk_document(
        parsed_doc, doc_id="demo_doc"
    )

    print("✅ Created chunks:")
    print(f"   - Text chunks: {len(text_chunks)}")
    print(f"   - Visual chunks: {len(visual_chunks)}")
    print(f"   - Multimodal chunks: {len(multimodal_chunks)}")

    # 3. Show sample text chunk
    if text_chunks:
        print("\n📝 Sample text chunk:")
        sample = text_chunks[0]
        print(f"   - ID: {sample.chunk_id[:8]}...")
        print(f"   - Page: {sample.page}")
        print(f"   - Content preview: {sample.content[:100]}...")

    # 4. Show sample visual chunk
    if visual_chunks:
        print("\n🖼️  Sample visual chunk:")
        sample = visual_chunks[0]
        print(f"   - ID: {sample.chunk_id[:8]}...")
        print(f"   - Page: {sample.page}")
        print(f"   - Type: {sample.visual_type}")
        print(f"   - Image size: {len(sample.image_data)} bytes")

    # 5. Show sample multimodal chunk
    if multimodal_chunks:
        print("\n🎨 Sample multimodal chunk:")
        sample = multimodal_chunks[0]
        print(f"   - ID: {sample.chunk_id[:8]}...")
        print(f"   - Page: {sample.page}")
        print(f"   - Text preview: {sample.text_content[:100]}...")
        print(f"   - Visual elements: {len(sample.visual_elements)}")

    # 6. Initialize VLM client
    print("\n3. Initializing Vision-Language Model...")
    vlm_client = VLMClient()

    if vlm_client.enabled:
        print("✅ VLM initialized:")
        print(f"   - Provider: {vlm_client.provider}")
        print(f"   - Model: {vlm_client.model}")

        # Try to describe an image if available
        if parsed_doc.images:
            print("\n4. Testing VLM image description...")
            try:
                description = await vlm_client.describe_image(parsed_doc.images[0])
                print("✅ Image description:")
                print(f"   {description[:200]}...")
            except Exception as e:
                print(f"⚠️  VLM description failed: {e}")
    else:
        print("⚠️  VLM not enabled (API key not set or provider unavailable)")

    # 7. Initialize embedder
    print("\n5. Initializing Multimodal Embedder...")
    embedder = MultimodalEmbedder()

    print("✅ Embedder initialized:")
    print(f"   - Text embedding: {'enabled' if embedder.text_enabled else 'disabled'}")
    print(f"   - Vision embedding: {'enabled' if embedder.vision_enabled else 'disabled'}")

    # Try to embed a text chunk
    if embedder.text_enabled and text_chunks:
        print("\n6. Testing text embedding...")
        try:
            embedded = await embedder.embed_text_chunks([text_chunks[0]])
            if embedded:
                print("✅ Text embedding created:")
                print(f"   - Embedding dimension: {len(embedded[0].embedding)}")
                print(f"   - Modality: {embedded[0].modality}")
        except Exception as e:
            print(f"⚠️  Text embedding failed: {e}")

    print("\n" + "=" * 80)
    print("Demo completed! Multimodal RAG-Anything system is operational.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
