"""Multimodal chunking for text and visual content."""

import logging
import uuid

from app.models.chunk import MultimodalChunk, TextChunk, VisualChunk
from app.models.parsing import ChartBlock, ImageBlock, ParsedDocument, TextBlock

logger = logging.getLogger(__name__)


class MultimodalChunker:
    """Chunker for multimodal content combining text and visual elements."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize multimodal chunker.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized MultimodalChunker: size={chunk_size}, overlap={chunk_overlap}")

    def chunk_document(
        self, parsed_doc: ParsedDocument, doc_id: str
    ) -> tuple[list[TextChunk], list[VisualChunk], list[MultimodalChunk]]:
        """Create multimodal chunks from parsed document.

        Args:
            parsed_doc: Parsed document with text and visual content
            doc_id: Document identifier

        Returns:
            Tuple of (text_chunks, visual_chunks, multimodal_chunks)
        """
        logger.info(f"Chunking document {doc_id}")

        # Create text-only chunks
        text_chunks = self.create_text_chunks(parsed_doc.text_blocks, doc_id)

        # Create visual-only chunks
        visual_chunks = self.create_visual_chunks(parsed_doc.images, parsed_doc.charts, doc_id)

        # Create multimodal chunks (text + associated visuals)
        multimodal_chunks = self.create_multimodal_chunks(text_chunks, visual_chunks, parsed_doc)

        logger.info(
            f"Created {len(text_chunks)} text, {len(visual_chunks)} visual, "
            f"{len(multimodal_chunks)} multimodal chunks"
        )

        return text_chunks, visual_chunks, multimodal_chunks

    def create_text_chunks(self, text_blocks: list[TextBlock], doc_id: str) -> list[TextChunk]:
        """Create text-only chunks with semantic boundaries.

        Args:
            text_blocks: List of text blocks from document
            doc_id: Document identifier

        Returns:
            List of text chunks
        """
        chunks = []
        chunk_index = 0

        for block in text_blocks:
            # If block is small enough, create single chunk
            if len(block.content) <= self.chunk_size:
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=block.content,
                    page=block.page,
                    chunk_index=chunk_index,
                    metadata={
                        "bbox": block.bbox,
                        "font_size": block.font_size,
                        "layout_type": block.layout_type,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large blocks with overlap
                sub_chunks = self._split_text_with_overlap(block.content)
                for sub_content in sub_chunks:
                    chunk = TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        content=sub_content,
                        page=block.page,
                        chunk_index=chunk_index,
                        metadata={
                            "bbox": block.bbox,
                            "font_size": block.font_size,
                            "layout_type": block.layout_type,
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        return chunks

    def _split_text_with_overlap(self, text: str) -> list[str]:
        """Split text into chunks with overlap, preserving sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [". ", ".\n", "! ", "?\n", "? "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def create_visual_chunks(
        self,
        images: list[ImageBlock],
        charts: list[ChartBlock],
        doc_id: str,
    ) -> list[VisualChunk]:
        """Create visual-only chunks for images and charts.

        Args:
            images: List of image blocks
            charts: List of chart blocks
            doc_id: Document identifier

        Returns:
            List of visual chunks
        """
        chunks = []
        chunk_index = 0

        # Process images
        for img in images:
            chunk = VisualChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                image_data=img.image_data,
                page=img.page,
                chunk_index=chunk_index,
                visual_type="image",
                caption=None,
                metadata={
                    "bbox": img.bbox,
                    "format": img.format,
                    "visual_features": img.visual_features,
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        # Process charts
        for chart in charts:
            chunk = VisualChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                image_data=chart.image_data,
                page=chart.page,
                chunk_index=chunk_index,
                visual_type="chart",
                caption=None,
                metadata={
                    "bbox": chart.bbox,
                    "chart_type": chart.chart_type,
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def create_multimodal_chunks(
        self,
        text_chunks: list[TextChunk],
        visual_chunks: list[VisualChunk],
        parsed_doc: ParsedDocument,
    ) -> list[MultimodalChunk]:
        """Create multimodal chunks combining text and visual elements.

        Args:
            text_chunks: List of text chunks
            visual_chunks: List of visual chunks
            parsed_doc: Original parsed document

        Returns:
            List of multimodal chunks
        """
        chunks = []
        chunk_index = 0

        # Group text chunks by page
        text_by_page: dict[int, list[TextChunk]] = {}
        for chunk in text_chunks:
            if chunk.page not in text_by_page:
                text_by_page[chunk.page] = []
            text_by_page[chunk.page].append(chunk)

        # Group visual chunks by page
        visual_by_page: dict[int, list[VisualChunk]] = {}
        for chunk in visual_chunks:
            if chunk.page not in visual_by_page:
                visual_by_page[chunk.page] = []
            visual_by_page[chunk.page].append(chunk)

        # Create multimodal chunks for pages with both text and visuals
        for page in sorted(set(text_by_page.keys()) & set(visual_by_page.keys())):
            text_list = text_by_page[page]
            visual_list = visual_by_page[page]

            # For each text chunk on the page, associate nearby visuals
            for text_chunk in text_list:
                # Find visuals on same page
                associated_visuals = self._find_associated_visuals(text_chunk, visual_list)

                if associated_visuals:
                    multimodal_chunk = MultimodalChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=text_chunk.doc_id,
                        text_content=text_chunk.content,
                        visual_elements=associated_visuals,
                        page=page,
                        chunk_index=chunk_index,
                        metadata={
                            "text_chunk_id": text_chunk.chunk_id,
                            "num_visuals": len(associated_visuals),
                        },
                    )
                    chunks.append(multimodal_chunk)
                    chunk_index += 1

        return chunks

    def _find_associated_visuals(
        self, text_chunk: TextChunk, visual_chunks: list[VisualChunk]
    ) -> list[VisualChunk]:
        """Find visual chunks associated with a text chunk.

        Args:
            text_chunk: Text chunk to find visuals for
            visual_chunks: List of visual chunks on same page

        Returns:
            List of associated visual chunks
        """
        # Simple heuristic: associate all visuals on the same page
        # In a more sophisticated implementation, we could use spatial proximity
        # based on bounding boxes

        # For now, return all visuals on the same page
        # This ensures multimodal chunks capture the full context
        return visual_chunks
