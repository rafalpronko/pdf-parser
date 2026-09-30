"""RAG-Anything PDF parser implementation using MinerU."""

import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

from app.models.parsing import (
    ImageBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)

logger = logging.getLogger(__name__)


class _BlockType:
    """Para block types in magic-pdf 0.6.1 middle JSON (magic_pdf.libs.ocr_content_type)."""

    TEXT = "text"
    TITLE = "title"
    INTERLINE_EQUATION = "interline_equation"
    IMAGE = "image"
    IMAGE_BODY = "image_body"
    IMAGE_CAPTION = "image_caption"
    TABLE = "table"
    TABLE_BODY = "table_body"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"


class _SpanType:
    """Span types in magic-pdf 0.6.1 middle JSON (magic_pdf.libs.ocr_content_type)."""

    TEXT = "text"
    INLINE_EQUATION = "inline_equation"
    INTERLINE_EQUATION = "interline_equation"
    IMAGE = "image"
    TABLE = "table"


# TextBlock.layout_type for text-bearing para blocks. Titles become separate
# "heading" blocks so the structure-aware chunker can start sections at them.
_TEXT_LAYOUT_TYPES = {
    _BlockType.TEXT: "paragraph",
    _BlockType.TITLE: "heading",
    _BlockType.INTERLINE_EQUATION: "equation",
}


class RAGAnythingParser:
    """PDF parser using MinerU (magic-pdf) for advanced PDF extraction.

    Extracts text, images, tables, and formulas from PDF documents using
    the UNIPipe API from MinerU. Provides structured content with positional
    metadata and multi-modal support.
    """

    def __init__(self, output_dir: str | None = None) -> None:
        """Initialize the MinerU parser.

        Args:
            output_dir: Directory for storing extracted images/tables.
                       If None, uses a temporary directory.
        """
        self.logger = logger
        self.output_dir = Path(output_dir) if output_dir else None

        # Set environment variable to use local models
        os.environ["MINERU_MODEL_SOURCE"] = "local"

        # Import MinerU modules and configure to use local models
        try:
            import magic_pdf.model as model_config
            from magic_pdf.pipe.UNIPipe import UNIPipe
            from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

            # Enable using inside (local) models
            model_config.__use_inside_model__ = True
            # Use lite mode (Paddle) instead of full mode (PEK) - no detectron2 required
            model_config.__model_mode__ = "lite"

            self.UNIPipe = UNIPipe
            self.DiskReaderWriter = DiskReaderWriter
            logger.info("MinerU initialized successfully with local models (lite mode)")
        except ImportError as e:
            error_msg = (
                f"MinerU is required but not available: {e}\n"
                "Install MinerU (magic-pdf 0.6.1, Python < 3.13) manually with: "
                'uv pip install "magic-pdf[cpu]==0.6.1" "numpy<2" setuptools'
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e

    def parse_pdf(self, file_path: Path) -> ParsedDocument:
        """Extract text, images, tables from PDF using MinerU UNIPipe.

        Args:
            file_path: Path to the PDF file

        Returns:
            ParsedDocument containing all extracted content

        Raises:
            ValueError: If file doesn't exist or isn't a valid PDF
        """
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        logger.info(f"→ Parsing PDF with MinerU: {file_path.name}")
        return self._parse_with_mineru_unipipe(file_path)

    def _parse_with_mineru_unipipe(self, file_path: Path) -> ParsedDocument:
        """Parse PDF using MinerU UNIPipe API.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with extracted content (text, images, tables)
        """
        try:
            # Read PDF bytes
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            # Setup output directory for images
            if self.output_dir:
                image_output_dir = self.output_dir / "images" / file_path.stem
            else:
                temp_dir = tempfile.mkdtemp(prefix="mineru_")
                image_output_dir = Path(temp_dir) / "images"

            image_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Images will be saved to: {image_output_dir}")

            # Initialize DiskReaderWriter for image storage
            image_writer = self.DiskReaderWriter(str(image_output_dir))

            # Prepare jso_useful_key (required by UNIPipe)
            jso_useful_key = {"_pdf_type": "", "model_list": []}

            # Initialize UNIPipe
            logger.info("  Initializing UNIPipe...")
            pipe = self.UNIPipe(pdf_bytes, jso_useful_key, image_writer)

            # Execute pipeline stages
            logger.info("  Stage 1/3: Classification...")
            pipe.pipe_classify()

            logger.info("  Stage 2/3: Analysis...")
            pipe.pipe_analyze()

            logger.info("  Stage 3/3: Parsing...")
            pipe.pipe_parse()

            # Use the per-page middle JSON produced by pipe_parse(). The content list
            # from pipe_mk_uni_format() is flat and carries no page numbers.
            pdf_mid_data = pipe.pdf_mid_data
            if not pdf_mid_data or "pdf_info" not in pdf_mid_data:
                raise RuntimeError("MinerU produced no parse result (pdf_mid_data is empty)")

            # Convert to ParsedDocument
            logger.info("  Extracting content...")
            parsed_doc = self._convert_to_parsed_document(pdf_mid_data, file_path, image_output_dir)

            logger.info(
                f"✓ MinerU parsing complete: "
                f"{len(parsed_doc.text_blocks)} text blocks, "
                f"{len(parsed_doc.images)} images, "
                f"{len(parsed_doc.tables)} tables"
            )

            return parsed_doc

        except SystemExit as e:
            # magic-pdf 0.6.1 calls exit(1) when a model dependency (e.g. paddleocr) fails
            # to import; don't let that terminate the API server.
            error_msg = (
                f"MinerU aborted (exit code {e.code}); check the 'mineru' extra installation"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error(f"MinerU UNIPipe parsing failed: {e}", exc_info=True)
            raise

    def _read_image_as_bytes(self, image_path: Path) -> tuple[bytes, str]:
        """Read image file and return bytes with format.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (image_data as bytes, format string)
        """
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Detect format from extension
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if mime_type and mime_type.startswith("image/"):
                format_str = mime_type.split("/")[1]  # e.g., "jpeg", "png"
            else:
                format_str = image_path.suffix.lstrip(".")  # fallback to extension

            return image_data, format_str
        except Exception as e:
            logger.warning(f"Failed to read image {image_path}: {e}")
            return b"", "unknown"

    def _parse_html_table(self, html_content: str) -> list[list[str]]:
        """Parse HTML table content into rows and columns.

        Args:
            html_content: HTML table string

        Returns:
            List of rows, where each row is a list of cell contents
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")
            table = soup.find("table")

            if not table:
                return [[]]

            rows = []
            for tr in table.find_all("tr"):
                cells = []
                for cell in tr.find_all(["td", "th"]):
                    cells.append(cell.get_text(strip=True))
                if cells:  # Only add non-empty rows
                    rows.append(cells)

            return rows if rows else [[]]

        except ImportError:
            logger.warning("BeautifulSoup not available, cannot parse HTML tables")
            return [[html_content]]  # Return raw HTML as single cell
        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            return [[html_content]]  # Return raw HTML as single cell

    def _convert_to_parsed_document(
        self, pdf_mid_data: dict[str, Any], file_path: Path, image_dir: Path
    ) -> ParsedDocument:
        """Convert MinerU middle JSON (``pipe.pdf_mid_data``) to ParsedDocument.

        magic-pdf 0.6.1 stores one dict per page in ``pdf_mid_data["pdf_info"]`` with a
        0-based ``page_idx`` and ``para_blocks`` in reading order. Text lives in
        ``lines`` -> ``spans``; image and table blocks nest ``*_body``, ``*_caption`` and
        ``table_footnote`` sub-blocks, and the body span's ``image_path`` is relative to
        the image writer root. Pages flagged with ``need_drop`` are kept (as with
        ``drop_mode="none"``).

        Args:
            pdf_mid_data: Middle JSON produced by ``UNIPipe.pipe_parse()``
            file_path: Original PDF file path
            image_dir: Root directory of the image writer holding the cropped images

        Returns:
            ParsedDocument with structured content
        """
        pages: list[dict[str, Any]] = pdf_mid_data.get("pdf_info") or []
        text_blocks: list[TextBlock] = []
        images: list[ImageBlock] = []
        tables: list[TableBlock] = []
        num_pages = len(pages)

        for position, page_info in enumerate(pages):
            page_num = int(page_info.get("page_idx", position))
            num_pages = max(num_pages, page_num + 1)

            if page_info.get("need_drop"):
                logger.debug(
                    f"  Page {page_num} flagged by MinerU ({page_info.get('drop_reason')}), "
                    "keeping its content"
                )

            for block in page_info.get("para_blocks") or []:
                block_type = block.get("type")

                if block_type in _TEXT_LAYOUT_TYPES:
                    text_block = self._make_text_block(
                        block, page_num, _TEXT_LAYOUT_TYPES[block_type]
                    )
                    if text_block:
                        text_blocks.append(text_block)
                elif block_type == _BlockType.IMAGE:
                    self._convert_image_block(block, page_num, image_dir, text_blocks, images)
                elif block_type == _BlockType.TABLE:
                    self._convert_table_block(
                        block, page_num, image_dir, text_blocks, images, tables
                    )
                else:
                    logger.debug(
                        f"  Skipping unsupported MinerU block type '{block_type}' "
                        f"on page {page_num}"
                    )

        return ParsedDocument(
            text_blocks=text_blocks,
            images=images,
            charts=[],
            tables=tables,
            num_pages=num_pages,
            metadata={
                "parser": "mineru_unipipe",
                "version": pdf_mid_data.get("_version_name", "0.6.1"),
                "parse_type": pdf_mid_data.get("_parse_type"),
                "source": str(file_path),
            },
        )

    def _convert_image_block(
        self,
        block: dict[str, Any],
        page_num: int,
        image_dir: Path,
        text_blocks: list[TextBlock],
        images: list[ImageBlock],
    ) -> None:
        """Convert an image para block into an ImageBlock and caption TextBlocks.

        The caption is emitted as text (so it is chunked and searchable) and is also
        attached to the image's ``visual_features``.
        """
        captions = self._make_sub_text_blocks(block, _BlockType.IMAGE_CAPTION, page_num, "caption")
        caption = " ".join(c.content for c in captions) or None

        image = self._load_body_image(
            block,
            _BlockType.IMAGE_BODY,
            _SpanType.IMAGE,
            page_num,
            image_dir,
            visual_features={"block_type": _BlockType.IMAGE, "caption": caption},
        )
        if image:
            images.append(image)
        text_blocks.extend(captions)

    def _convert_table_block(
        self,
        block: dict[str, Any],
        page_num: int,
        image_dir: Path,
        text_blocks: list[TextBlock],
        images: list[ImageBlock],
        tables: list[TableBlock],
    ) -> None:
        """Convert a table para block into a TableBlock, its image and caption/footnote text.

        magic-pdf 0.6.1 does not recognise table cells: the table body is only a cropped
        image. Rows are filled only when the body span carries HTML; otherwise ``rows``
        stays empty and the crop is kept as an ImageBlock so the table is not lost.
        """
        captions = self._make_sub_text_blocks(block, _BlockType.TABLE_CAPTION, page_num, "caption")
        footnotes = self._make_sub_text_blocks(
            block, _BlockType.TABLE_FOOTNOTE, page_num, "footnote"
        )
        caption = " ".join(c.content for c in captions) or None

        rows: list[list[str]] = [[]]
        headers = None
        body_span = self._find_body_span(block, _BlockType.TABLE_BODY, _SpanType.TABLE)
        html_content = body_span.get("html") if body_span else None
        if html_content:
            rows = self._parse_html_table(html_content)
            # Simple heuristic: treat the first row as headers
            headers = rows[0] if rows and rows[0] else None

        tables.append(
            TableBlock(
                rows=rows,
                page=page_num,
                bbox=self._to_bbox(block.get("bbox")),
                headers=headers,
            )
        )

        image = self._load_body_image(
            block,
            _BlockType.TABLE_BODY,
            _SpanType.TABLE,
            page_num,
            image_dir,
            visual_features={"block_type": _BlockType.TABLE, "caption": caption},
        )
        if image:
            images.append(image)
        text_blocks.extend(captions)
        text_blocks.extend(footnotes)

    def _load_body_image(
        self,
        block: dict[str, Any],
        body_type: str,
        span_type: str,
        page_num: int,
        image_dir: Path,
        visual_features: dict[str, Any],
    ) -> ImageBlock | None:
        """Read the cropped image referenced by an image/table body span.

        Args:
            block: Image or table para block
            body_type: Sub-block type holding the crop (image_body / table_body)
            span_type: Span type carrying ``image_path`` (image / table)
            page_num: 0-based page number
            image_dir: Root directory of the image writer
            visual_features: Metadata attached to the ImageBlock

        Returns:
            ImageBlock, or None if the block has no readable image
        """
        span = self._find_body_span(block, body_type, span_type)
        image_path = span.get("image_path") if span else None
        if not span or not image_path:
            logger.debug(f"  No {body_type} image on page {page_num}, skipping image")
            return None

        # image_path is relative to the DiskReaderWriter root (e.g. "<sha256>.jpg")
        image_data, image_format = self._read_image_as_bytes(image_dir / image_path)
        if not image_data:
            return None

        return ImageBlock(
            image_data=image_data,
            page=page_num,
            bbox=self._to_bbox(span.get("bbox") or block.get("bbox")),
            format=image_format,
            visual_features=visual_features,
        )

    def _make_text_block(
        self, block: dict[str, Any], page_num: int, layout_type: str
    ) -> TextBlock | None:
        """Build a TextBlock from a block's spans, or None if it holds no text."""
        content = self._merge_block_text(block)
        if not content:
            return None

        return TextBlock(
            content=content,
            page=page_num,
            bbox=self._to_bbox(block.get("bbox")),
            font_size=None,
            layout_type=layout_type,
        )

    def _make_sub_text_blocks(
        self, block: dict[str, Any], sub_type: str, page_num: int, layout_type: str
    ) -> list[TextBlock]:
        """Build TextBlocks for the nested sub-blocks of ``sub_type`` (captions, footnotes)."""
        text_blocks = []
        for sub_block in block.get("blocks") or []:
            if sub_block.get("type") == sub_type:
                text_block = self._make_text_block(sub_block, page_num, layout_type)
                if text_block:
                    text_blocks.append(text_block)
        return text_blocks

    @staticmethod
    def _find_body_span(
        block: dict[str, Any], body_type: str, span_type: str
    ) -> dict[str, Any] | None:
        """Return the first ``span_type`` span inside the block's ``body_type`` sub-block."""
        for sub_block in block.get("blocks") or []:
            if sub_block.get("type") != body_type:
                continue
            for line in sub_block.get("lines") or []:
                for span in line.get("spans") or []:
                    if span.get("type") == span_type:
                        return span
        return None

    @staticmethod
    def _merge_block_text(block: dict[str, Any]) -> str:
        """Join the text of a block's spans in reading order.

        Like magic-pdf's ``merge_para_with_text`` (spans separated by spaces, inline
        equations as ``$...$``, interline equations as ``$$...$$``) but without
        markdown escaping. Image and table spans carry no text and are skipped.
        """
        parts: list[str] = []
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                content = (span.get("content") or "").strip()
                if not content:
                    continue

                span_type = span.get("type")
                if span_type == _SpanType.TEXT:
                    parts.append(content)
                elif span_type == _SpanType.INLINE_EQUATION:
                    parts.append(f"${content}$")
                elif span_type == _SpanType.INTERLINE_EQUATION:
                    parts.append(f"$${content}$$")
        return " ".join(parts)

    @staticmethod
    def _to_bbox(bbox_data: Any) -> tuple[float, float, float, float]:
        """Convert a MinerU bbox list to a 4-float tuple ((0, 0, 0, 0) if missing)."""
        if not bbox_data or len(bbox_data) < 4:
            return (0.0, 0.0, 0.0, 0.0)
        x0, y0, x1, y1 = (float(x) for x in bbox_data[:4])
        return (x0, y0, x1, y1)
