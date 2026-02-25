"""Text chunking logic for document processing."""

import re
import uuid
from typing import Any

from app.models.chunk import DocumentChunk
from app.models.parsing import ImageBlock, ParsedDocument


class SemanticChunker:
    """Chunks documents into semantic segments with configurable size and overlap.

    This chunker preserves sentence integrity, respects document structure,
    and associates images with relevant text chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize the semantic chunker.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks

        Raises:
            ValueError: If chunk_overlap >= chunk_size or if values are negative
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        doc_id: str,
    ) -> list[DocumentChunk]:
        """Create semantic chunks from a parsed document.

        This method combines text blocks, splits them into chunks with overlap,
        and associates images with relevant chunks based on page proximity.

        Args:
            parsed_doc: The parsed document to chunk
            doc_id: Document identifier

        Returns:
            List of document chunks with metadata
        """
        # Extract all text content organized by page
        page_texts: dict[int, str] = {}
        for text_block in parsed_doc.text_blocks:
            page = text_block.page
            if page not in page_texts:
                page_texts[page] = ""
            page_texts[page] += text_block.content + "\n"

        # Combine all text in page order
        full_text = ""
        page_boundaries: list[tuple[int, int, int]] = []  # (page, start_pos, end_pos)

        for page in sorted(page_texts.keys()):
            start_pos = len(full_text)
            page_text = page_texts[page]
            full_text += page_text
            end_pos = len(full_text)
            page_boundaries.append((page, start_pos, end_pos))

        # Split into chunks
        chunks = self._split_text_with_overlap(full_text)

        # Create DocumentChunk objects with metadata
        document_chunks: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(chunks):
            # Find which page this chunk belongs to
            chunk_start = self._find_chunk_position(full_text, chunk_text, idx, chunks)
            chunk_page = self._find_page_for_position(chunk_start, page_boundaries)

            # Associate images from the same page or adjacent pages
            associated_images = self._find_associated_images(chunk_page, parsed_doc.images)

            chunk_id = str(uuid.uuid4())
            metadata: dict[str, Any] = {
                "chunk_length": len(chunk_text),
                "associated_images": associated_images,
            }

            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk_text.strip(),
                    page=chunk_page,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )

        return document_chunks

    def chunk_with_structure(
        self,
        parsed_doc: ParsedDocument,
        doc_id: str,
    ) -> list[DocumentChunk]:
        """Chunk document using structural information like headings.

        This method attempts to identify section boundaries and align
        chunk boundaries with document structure when possible.

        Args:
            parsed_doc: The parsed document to chunk
            doc_id: Document identifier

        Returns:
            List of document chunks aligned with document structure
        """
        # Extract text blocks with structure detection
        sections: list[tuple[str, int, bool]] = []  # (text, page, is_heading)

        for text_block in parsed_doc.text_blocks:
            text = text_block.content
            page = text_block.page

            # Detect if this is likely a heading based on:
            # 1. Font size (if available)
            # 2. Short length
            # 3. Capitalization patterns
            is_heading = self._is_likely_heading(text, text_block.font_size)
            sections.append((text, page, is_heading))

        # Group text by sections, keeping track of headings
        current_section = ""
        current_page = 0
        current_heading = ""
        section_chunks: list[tuple[str, int, str]] = []  # (text, page, heading)

        for text, page, is_heading in sections:
            if is_heading and current_section:
                # Save current section and start new one
                section_chunks.append((current_section, current_page, current_heading))
                current_heading = text
                current_section = text + "\n"
                current_page = page
            else:
                if not current_section:
                    current_page = page
                if is_heading:
                    current_heading = text
                current_section += text + "\n"

        # Add final section
        if current_section:
            section_chunks.append((current_section, current_page, current_heading))

        # Now chunk each section, respecting section boundaries
        document_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section_text, section_page, section_heading in section_chunks:
            # If section is small enough, keep it as one chunk
            if len(section_text) <= self.chunk_size:
                chunk_id = str(uuid.uuid4())
                associated_images = self._find_associated_images(section_page, parsed_doc.images)

                document_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        content=section_text.strip(),
                        page=section_page,
                        chunk_index=chunk_index,
                        metadata={
                            "chunk_length": len(section_text),
                            "associated_images": associated_images,
                            "section_boundary": True,
                            "section_heading": section_heading if section_heading else None,
                        },
                    )
                )
                chunk_index += 1
            else:
                # Split large sections with overlap
                section_sub_chunks = self._split_text_with_overlap(section_text)
                for sub_idx, sub_chunk_text in enumerate(section_sub_chunks):
                    chunk_id = str(uuid.uuid4())
                    associated_images = self._find_associated_images(
                        section_page, parsed_doc.images
                    )

                    # Mark the first chunk of a split section as a section boundary
                    is_section_boundary = sub_idx == 0

                    # Add section heading as context prefix for better retrieval
                    chunk_content = sub_chunk_text.strip()
                    if section_heading and section_heading not in chunk_content:
                        chunk_content = f"[Context: {section_heading}]\n\n{chunk_content}"

                    document_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            content=chunk_content,
                            page=section_page,
                            chunk_index=chunk_index,
                            metadata={
                                "chunk_length": len(sub_chunk_text),
                                "associated_images": associated_images,
                                "section_boundary": is_section_boundary,
                                "section_heading": section_heading if section_heading else None,
                            },
                        )
                    )
                    chunk_index += 1

        return document_chunks

    def _split_text_with_overlap(self, text: str) -> list[str]:
        """Split text into chunks with overlap, preserving sentence integrity.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Split into sentences using common sentence boundaries
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        chunks: list[str] = []
        current_chunk = ""
        overlap_buffer = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk)

                # Start new chunk with overlap from end of previous chunk
                current_chunk = overlap_buffer + sentence
                overlap_buffer = self._get_overlap_text(current_chunk)
            else:
                current_chunk += sentence
                overlap_buffer = self._get_overlap_text(current_chunk)

            # Safety check: if a single sentence is too long, split it by words
            if len(current_chunk) > self.chunk_size * 2:
                # Split by words and create chunks
                word_chunks = self._split_long_text_by_words(current_chunk)
                chunks.extend(word_chunks[:-1])  # Add all but last
                current_chunk = word_chunks[-1] if word_chunks else ""
                overlap_buffer = self._get_overlap_text(current_chunk)

        # Add final chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences preserving integrity.

        Uses regex to identify sentence boundaries while handling
        common abbreviations and edge cases.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Pattern matches sentence-ending punctuation followed by whitespace
        # and a capital letter or end of string
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$"

        sentences = re.split(sentence_pattern, text)

        # Filter out empty sentences and preserve whitespace
        result = []
        for sentence in sentences:
            if sentence.strip():
                # Ensure sentence ends with space for natural joining
                if not sentence.endswith((" ", "\n", "\t")):
                    sentence += " "
                result.append(sentence)

        # If no sentences found (no punctuation), treat whole text as one sentence
        if not result and text.strip():
            result = [text]

        return result

    def _split_long_text_by_words(self, text: str) -> list[str]:
        """Split long text by words when it exceeds chunk size.

        This is a fallback for text without sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of text chunks split by words
        """
        words = text.split()
        chunks: list[str] = []
        current_chunk = ""

        for word in words:
            # Check if adding this word would exceed chunk size
            test_chunk = current_chunk + (" " if current_chunk else "") + word
            if len(test_chunk) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk)
                current_chunk = word
            else:
                current_chunk = test_chunk

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _get_overlap_text(self, text: str) -> str:
        """Get the last chunk_overlap characters from text.

        Tries to break at sentence boundaries when possible.

        Args:
            text: Text to extract overlap from

        Returns:
            Overlap text
        """
        if len(text) <= self.chunk_overlap:
            return text

        # Get last chunk_overlap characters
        overlap = text[-self.chunk_overlap :]

        # Try to start at a sentence boundary
        sentence_start = max(
            overlap.find(". "),
            overlap.find("! "),
            overlap.find("? "),
        )

        if sentence_start > 0:
            return overlap[sentence_start + 2 :]  # Skip punctuation and space

        # Try to start at a word boundary
        space_pos = overlap.find(" ")
        if space_pos > 0:
            return overlap[space_pos + 1 :]

        return overlap

    def _is_likely_heading(self, text: str, font_size: float | None) -> bool:
        """Determine if text is likely a heading.

        Args:
            text: Text to check
            font_size: Font size if available

        Returns:
            True if text appears to be a heading
        """
        # Short text (less than 100 chars) might be a heading
        if len(text) > 100:
            return False

        # Check for common heading patterns
        # All caps
        if text.isupper() and len(text) > 3:
            return True

        # Starts with number (e.g., "1. Introduction")
        if re.match(r"^\d+\.?\s+[A-Z]", text):
            return True

        # Title case (most words capitalized)
        words = text.split()
        if len(words) > 0:
            capitalized = sum(1 for w in words if w and w[0].isupper())
            if capitalized / len(words) > 0.6:
                return True

        return False

    def _find_chunk_position(
        self,
        full_text: str,
        chunk_text: str,
        chunk_idx: int,
        all_chunks: list[str],
    ) -> int:
        """Find the starting position of a chunk in the full text.

        Args:
            full_text: Complete document text
            chunk_text: The chunk to find
            chunk_idx: Index of this chunk
            all_chunks: All chunks for context

        Returns:
            Starting position in full_text
        """
        # For first chunk, position is 0
        if chunk_idx == 0:
            return 0

        # Estimate position based on previous chunks
        # Account for overlap
        estimated_pos = 0
        for i in range(chunk_idx):
            estimated_pos += len(all_chunks[i]) - self.chunk_overlap

        # Search around estimated position
        search_start = max(0, estimated_pos - 100)
        search_end = min(len(full_text), estimated_pos + 100)

        # Find the chunk in the search window
        chunk_start = chunk_text[:50].strip()  # Use first 50 chars for matching
        pos = full_text.find(chunk_start, search_start, search_end)

        if pos >= 0:
            return pos

        # Fallback to estimated position
        return estimated_pos

    def _find_page_for_position(
        self,
        position: int,
        page_boundaries: list[tuple[int, int, int]],
    ) -> int:
        """Find which page a text position belongs to.

        Args:
            position: Character position in full text
            page_boundaries: List of (page, start_pos, end_pos) tuples

        Returns:
            Page number
        """
        for page, start_pos, end_pos in page_boundaries:
            if start_pos <= position < end_pos:
                return page

        # Default to first page if not found
        if page_boundaries:
            return page_boundaries[0][0]
        return 0

    def _find_associated_images(
        self,
        page: int,
        images: list[ImageBlock],
    ) -> list[dict[str, Any]]:
        """Find images associated with a chunk based on page proximity.

        Images on the same page or adjacent pages are associated.

        Args:
            page: Page number of the chunk
            images: All images in the document

        Returns:
            List of image metadata dictionaries
        """
        associated = []

        for img in images:
            # Associate images from same page or adjacent pages
            if abs(img.page - page) <= 1:
                associated.append(
                    {
                        "page": img.page,
                        "bbox": img.bbox,
                        "format": img.format,
                    }
                )

        return associated
