"""RAG-Anything PDF parser implementation using MinerU."""

import logging
import mimetypes
import os
import tempfile
from pathlib import Path

from app.models.parsing import (
    ImageBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)

logger = logging.getLogger(__name__)


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
            error_msg = f"MinerU is required but not available: {e}\nInstall with: uv pip install 'mineru[core]'"
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

            # Get parsed content
            logger.info("  Extracting content...")
            content_list = pipe.pipe_mk_uni_format(str(image_output_dir.name), drop_mode="none")

            # Convert to ParsedDocument
            parsed_doc = self._convert_to_parsed_document(content_list, file_path, image_output_dir)

            logger.info(
                f"✓ MinerU parsing complete: "
                f"{len(parsed_doc.text_blocks)} text blocks, "
                f"{len(parsed_doc.images)} images, "
                f"{len(parsed_doc.tables)} tables"
            )

            return parsed_doc

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
        self, content_list: list[dict], file_path: Path, image_dir: Path
    ) -> ParsedDocument:
        """Convert MinerU content list to ParsedDocument.

        Args:
            content_list: List of content dictionaries from UNIPipe
            file_path: Original PDF file path
            image_dir: Directory containing extracted images

        Returns:
            ParsedDocument with structured content
        """
        text_blocks = []
        images = []
        tables = []
        charts = []
        num_pages = 0

        for page_content in content_list:
            page_num = page_content.get("page_idx", 0)
            num_pages = max(num_pages, page_num + 1)

            # Extract text blocks
            for block in page_content.get("para_blocks", []):
                if block.get("type") == "text":
                    # Extract text from spans
                    text_parts = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("type") == "text":
                                text_parts.append(span.get("content", ""))

                    if text_parts:
                        text_content = " ".join(text_parts)
                        bbox_data = block.get("bbox", [0, 0, 0, 0])
                        bbox = tuple(float(x) for x in bbox_data[:4])

                        text_blocks.append(
                            TextBlock(
                                content=text_content,
                                page=page_num,
                                bbox=bbox,
                                font_size=None,
                                layout_type="paragraph",
                            )
                        )

            # Extract images
            for img in page_content.get("images", []):
                img_path = img.get("img_path")
                if img_path:
                    # Convert relative path to absolute
                    full_img_path = image_dir / img_path

                    # Read image data and detect format
                    image_data, image_format = self._read_image_as_bytes(full_img_path)

                    if image_data:  # Only add if image was read successfully
                        bbox_data = img.get("bbox", [0, 0, 0, 0])
                        bbox = tuple(float(x) for x in bbox_data[:4])

                        images.append(
                            ImageBlock(
                                image_data=image_data,
                                page=page_num,
                                bbox=bbox,
                                format=image_format,
                            )
                        )

            # Extract tables
            for table in page_content.get("tables", []):
                html_content = table.get("html", None)

                if html_content:
                    # Parse HTML table to rows/columns structure
                    rows = self._parse_html_table(html_content)

                    bbox_data = table.get("bbox", [0, 0, 0, 0])
                    bbox = tuple(float(x) for x in bbox_data[:4])

                    # Extract headers (first row if it contains th elements)
                    headers = None
                    if rows and len(rows) > 0:
                        # Check if first row could be headers
                        # (this is a simple heuristic)
                        headers = rows[0] if rows else None

                    tables.append(
                        TableBlock(
                            rows=rows,
                            page=page_num,
                            bbox=bbox,
                            headers=headers,
                        )
                    )

        return ParsedDocument(
            text_blocks=text_blocks,
            images=images,
            charts=charts,
            tables=tables,
            num_pages=num_pages,
            metadata={
                "parser": "mineru_unipipe",
                "version": "0.6.1",
                "source": str(file_path),
            },
        )
