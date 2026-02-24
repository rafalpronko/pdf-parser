"""OpenAI GPT-4o Vision-based PDF parser."""

import base64
import io
import logging
from pathlib import Path

from app.models.parsing import ImageBlock, ParsedDocument, TableBlock, TextBlock

logger = logging.getLogger(__name__)


class OpenAIPDFParser:
    """PDF parser using OpenAI GPT-4o Vision for multimodal extraction.

    Converts PDF pages to images and uses GPT-4o Vision to extract:
    - Text content with layout preservation
    - Tables as structured rows/columns
    - Image descriptions

    This approach avoids complex dependencies like Paddle/OpenCV.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        """Initialize OpenAI PDF parser.

        Args:
            api_key: OpenAI API key. If None, loads from settings.
            model: OpenAI model to use (gpt-4o, gpt-4o-mini)
        """
        self.logger = logger
        self.model = model

        # Get API key from settings if not provided
        if not api_key:
            from app.config import get_settings
            settings = get_settings()
            api_key = settings.openai_api_key

        # Create synchronous OpenAI client (not async)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

        logger.info(f"OpenAI PDF Parser initialized with model: {model}")

    def parse_pdf(self, file_path: Path) -> ParsedDocument:
        """Extract text, images, tables from PDF using GPT-4o Vision.

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

        logger.info(f"→ Parsing PDF with GPT-4o native support: {file_path.name}")

        # Read PDF as base64
        pdf_base64 = self._pdf_to_base64(file_path)
        logger.info(f"  PDF loaded ({file_path.stat().st_size / 1024:.1f} KB)")

        # Extract content with GPT-4o Vision (processes entire PDF)
        logger.info(f"  Sending to GPT-4o for extraction...")
        result = self._extract_pdf_content(pdf_base64)

        # Parse the result
        text_blocks = []
        all_tables = []

        # Extract pages from result
        for page_data in result.get("pages", []):
            page_num = page_data.get("page_number", 0)

            # Add text blocks
            if page_data.get("text"):
                text_blocks.append(
                    TextBlock(
                        content=page_data["text"],
                        page=page_num,
                        bbox=(0, 0, 0, 0),  # GPT-4o doesn't provide precise bbox
                        font_size=None,
                        layout_type="page",
                    )
                )

            # Add tables
            for table_data in page_data.get("tables", []):
                all_tables.append(
                    TableBlock(
                        rows=table_data["rows"],
                        page=page_num,
                        bbox=(0, 0, 0, 0),
                        headers=table_data.get("headers"),
                    )
                )

        num_pages = result.get("num_pages", len(result.get("pages", [])))

        logger.info(
            f"✓ GPT-4o parsing complete: {len(text_blocks)} text blocks, {len(all_tables)} tables"
        )

        return ParsedDocument(
            text_blocks=text_blocks,
            images=[],  # GPT-4o describes images in text, not separate
            charts=[],
            tables=all_tables,
            num_pages=num_pages,
            metadata={
                "parser": "openai_gpt4o_vision",
                "model": self.model,
                "source": str(file_path),
            },
        )

    def _pdf_to_base64(self, pdf_path: Path) -> str:
        """Convert PDF file to base64 string.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Base64-encoded PDF string
        """
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return base64.b64encode(pdf_bytes).decode("utf-8")

    def _extract_pdf_content(self, pdf_base64: str) -> dict:
        """Extract content from entire PDF using GPT-4o Vision.

        Args:
            pdf_base64: Base64-encoded PDF file

        Returns:
            Dictionary with extracted text and tables per page
        """
        # Construct prompt for GPT-4o Vision
        prompt = """Analyze this PDF document and extract ALL content in JSON format.

For EACH PAGE in the document, extract:
1. ALL TEXT from the page (preserve layout and structure)
2. ALL TABLES (if any) as structured rows and columns

Return ONLY valid JSON in this exact format:
{
  "num_pages": 10,
  "pages": [
    {
      "page_number": 0,
      "text": "Full text content from page 1...",
      "tables": [
        {
          "headers": ["Column1", "Column2"],
          "rows": [
            ["value1", "value2"],
            ["value3", "value4"]
          ]
        }
      ]
    },
    {
      "page_number": 1,
      "text": "Full text content from page 2...",
      "tables": []
    }
  ]
}

IMPORTANT:
- Process ALL pages in the document
- Page numbers are 0-indexed (first page = 0)
- If a page has no tables, use empty array []
- Include ALL text, even headers, footers, page numbers
- Preserve paragraph breaks with newlines"""

        try:
            import json

            # Use OpenAI client to call GPT-4o Vision with PDF
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{pdf_base64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=16000,  # Larger for multi-page PDFs
                temperature=0.0,  # Deterministic for data extraction
            )

            # Parse JSON response
            content = response.choices[0].message.content

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            logger.error(f"Failed to extract content from PDF: {e}")
            return {"num_pages": 0, "pages": []}
