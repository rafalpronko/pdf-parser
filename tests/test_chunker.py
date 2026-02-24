"""Tests for the SemanticChunker."""

import pytest

from app.models.parsing import ImageBlock, ParsedDocument, TextBlock
from app.processing.chunker import SemanticChunker


class TestSemanticChunker:
    """Test the SemanticChunker class."""

    def test_chunker_initialization(self):
        """Test chunker can be initialized with valid parameters."""
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50

    def test_chunker_rejects_invalid_parameters(self):
        """Test chunker rejects invalid initialization parameters."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(chunk_size=0, chunk_overlap=50)

        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            SemanticChunker(chunk_size=512, chunk_overlap=-1)

        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            SemanticChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_document_basic(self):
        """Test basic document chunking."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        # Create a simple parsed document
        text_blocks = [
            TextBlock(
                content="This is the first sentence. This is the second sentence.",
                page=0,
                bbox=(0.0, 0.0, 100.0, 20.0),
            ),
            TextBlock(
                content="This is the third sentence. This is the fourth sentence.",
                page=0,
                bbox=(0.0, 20.0, 100.0, 40.0),
            ),
        ]

        parsed_doc = ParsedDocument(
            text_blocks=text_blocks,
            images=[],
            tables=[],
            num_pages=1,
            metadata={},
        )

        chunks = chunker.chunk_document(parsed_doc, doc_id="test-doc-1")

        # Should create at least one chunk
        assert len(chunks) > 0

        # All chunks should have required fields
        for chunk in chunks:
            assert chunk.doc_id == "test-doc-1"
            assert chunk.chunk_id is not None
            assert len(chunk.content) > 0
            assert chunk.page >= 0
            assert chunk.chunk_index >= 0

    def test_chunk_document_preserves_content(self):
        """Test that chunking preserves document content."""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)

        text_content = "Short text that fits in one chunk."
        text_blocks = [
            TextBlock(
                content=text_content,
                page=0,
                bbox=(0.0, 0.0, 100.0, 20.0),
            ),
        ]

        parsed_doc = ParsedDocument(
            text_blocks=text_blocks,
            images=[],
            tables=[],
            num_pages=1,
            metadata={},
        )

        chunks = chunker.chunk_document(parsed_doc, doc_id="test-doc-2")

        # For short text, should create one chunk
        assert len(chunks) >= 1

        # Content should be preserved (accounting for whitespace normalization)
        combined_content = " ".join(chunk.content for chunk in chunks)
        assert text_content.strip() in combined_content

    def test_chunk_document_with_images(self):
        """Test that images are associated with chunks."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        text_blocks = [
            TextBlock(
                content="This is text on page 0.",
                page=0,
                bbox=(0.0, 0.0, 100.0, 20.0),
            ),
        ]

        images = [
            ImageBlock(
                image_data=b"fake_image_data",
                page=0,
                bbox=(0.0, 30.0, 50.0, 80.0),
                format="png",
            ),
        ]

        parsed_doc = ParsedDocument(
            text_blocks=text_blocks,
            images=images,
            tables=[],
            num_pages=1,
            metadata={},
        )

        chunks = chunker.chunk_document(parsed_doc, doc_id="test-doc-3")

        # Should have at least one chunk
        assert len(chunks) > 0

        # At least one chunk should have associated images
        has_images = any(
            len(chunk.metadata.get("associated_images", [])) > 0
            for chunk in chunks
        )
        assert has_images

    def test_chunk_with_structure(self):
        """Test structure-aware chunking."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        # Create text blocks that look like headings and content
        text_blocks = [
            TextBlock(
                content="INTRODUCTION",  # Looks like a heading
                page=0,
                bbox=(0.0, 0.0, 100.0, 20.0),
                font_size=16.0,
            ),
            TextBlock(
                content="This is the introduction text with some content.",
                page=0,
                bbox=(0.0, 20.0, 100.0, 40.0),
                font_size=12.0,
            ),
            TextBlock(
                content="METHODS",  # Another heading
                page=0,
                bbox=(0.0, 40.0, 100.0, 60.0),
                font_size=16.0,
            ),
            TextBlock(
                content="This is the methods section with details.",
                page=0,
                bbox=(0.0, 60.0, 100.0, 80.0),
                font_size=12.0,
            ),
        ]

        parsed_doc = ParsedDocument(
            text_blocks=text_blocks,
            images=[],
            tables=[],
            num_pages=1,
            metadata={},
        )

        chunks = chunker.chunk_with_structure(parsed_doc, doc_id="test-doc-4")

        # Should create multiple chunks
        assert len(chunks) > 0

        # All chunks should have required fields
        for chunk in chunks:
            assert chunk.doc_id == "test-doc-4"
            assert len(chunk.content) > 0

    def test_empty_document(self):
        """Test chunking an empty document."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        parsed_doc = ParsedDocument(
            text_blocks=[],
            images=[],
            tables=[],
            num_pages=0,
            metadata={},
        )

        chunks = chunker.chunk_document(parsed_doc, doc_id="test-doc-5")

        # Empty document should produce no chunks
        assert len(chunks) == 0

    def test_chunk_indices_are_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)

        # Create text that will produce multiple chunks
        long_text = " ".join([f"Sentence number {i}." for i in range(20)])
        text_blocks = [
            TextBlock(
                content=long_text,
                page=0,
                bbox=(0.0, 0.0, 100.0, 20.0),
            ),
        ]

        parsed_doc = ParsedDocument(
            text_blocks=text_blocks,
            images=[],
            tables=[],
            num_pages=1,
            metadata={},
        )

        chunks = chunker.chunk_document(parsed_doc, doc_id="test-doc-6")

        # Check that indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_respect_max_size(self):
        """Test that chunks respect the maximum size constraint."""
        chunk_size = 100
        chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=20)

        # Create long text
        long_text = " ".join([f"Word{i}" for i in range(100)])
        text_blocks = [
            TextBlock(
                content=long_text,
                page=0,
                bbox=(0.0, 0.0, 100.0, 20.0),
            ),
        ]

        parsed_doc = ParsedDocument(
            text_blocks=text_blocks,
            images=[],
            tables=[],
            num_pages=1,
            metadata={},
        )

        chunks = chunker.chunk_document(parsed_doc, doc_id="test-doc-7")

        # All chunks should respect max size (with some tolerance for sentence boundaries)
        for chunk in chunks:
            # Allow some tolerance for sentence boundary preservation
            assert len(chunk.content) <= chunk_size * 1.5
