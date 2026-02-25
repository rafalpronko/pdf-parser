"""Adobe PDF Extract API parser implementation."""

import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

from adobe.pdfservices.operation.auth.service_principal_credentials import (
    ServicePrincipalCredentials,
)
from adobe.pdfservices.operation.exception.exceptions import (
    SdkException,
    ServiceApiException,
    ServiceUsageException,
)
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
    ExtractElementType,
)
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
    ExtractPDFParams,
)
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
    ExtractPDFResult,
)

from app.models.parsing import ParsedDocument, TableBlock, TextBlock

logger = logging.getLogger(__name__)


class AdobePDFParser:
    """PDF parser using Adobe PDF Extract API.

    Uses Adobe Sensei AI to extract structured content from PDFs including:
    - Text blocks with formatting and structure
    - Complex tables with cell-level data
    - Images and figures
    - Document structure (headings, paragraphs, lists, etc.)
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ):
        """Initialize Adobe PDF parser.

        Args:
            client_id: Adobe API client ID. If None, loads from env var PDF_SERVICES_CLIENT_ID
            client_secret: Adobe API client secret. If None, loads from env var PDF_SERVICES_CLIENT_SECRET
        """
        self.logger = logger

        # Get credentials from parameters or environment variables
        self.client_id = client_id or os.getenv("PDF_SERVICES_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PDF_SERVICES_CLIENT_SECRET")

        if not all([self.client_id, self.client_secret]):
            raise ValueError(
                "Adobe PDF Services credentials not provided. "
                "Set PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET "
                "environment variables or pass them to constructor."
            )

        logger.info("Adobe PDF Parser initialized")

    def parse_pdf(self, file_path: Path) -> ParsedDocument:
        """Extract text, images, tables from PDF using Adobe PDF Extract API.

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

        logger.info(f"→ Parsing PDF with Adobe Extract API: {file_path.name}")

        try:
            # Read PDF file
            with open(file_path, "rb") as f:
                input_stream = f.read()

            # Create credentials
            credentials = ServicePrincipalCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret,
            )

            # Create PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Upload PDF
            logger.info("  Uploading PDF to Adobe...")
            input_asset = pdf_services.upload(
                input_stream=input_stream, mime_type=PDFServicesMediaType.PDF
            )

            # Create extraction parameters - extract text and tables
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES]
            )

            # Create extraction job
            extract_pdf_job = ExtractPDFJob(
                input_asset=input_asset, extract_pdf_params=extract_pdf_params
            )

            # Submit job and get result
            logger.info("  Submitting extraction job...")
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Get result asset
            logger.info("  Downloading extraction results...")
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Save to temporary file
            temp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            temp_zip.write(stream_asset.get_input_stream())
            temp_zip.close()

            # Extract and parse JSON from ZIP
            logger.info("  Parsing extraction results...")
            parsed_doc = self._parse_extraction_results(Path(temp_zip.name), file_path)

            # Clean up temp file
            os.unlink(temp_zip.name)

            logger.info(
                f"✓ Adobe Extract complete: "
                f"{len(parsed_doc.text_blocks)} text blocks, "
                f"{len(parsed_doc.tables)} tables"
            )

            return parsed_doc

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logger.error(f"Adobe PDF Extract API error: {e}")
            raise

    def _parse_extraction_results(self, zip_path: Path, original_pdf: Path) -> ParsedDocument:
        """Parse Adobe Extract API results from ZIP file.

        Args:
            zip_path: Path to ZIP file with extraction results
            original_pdf: Path to original PDF file

        Returns:
            ParsedDocument with structured content
        """
        text_blocks = []
        tables = []
        images = []
        num_pages = 0

        with zipfile.ZipFile(zip_path, "r") as archive:
            # Read structured data JSON
            with archive.open("structuredData.json") as json_file:
                data = json.load(json_file)

            # Get page count
            num_pages = len(data.get("pages", []))

            # Process elements
            for element in data.get("elements", []):
                path = element.get("Path", "")
                page = element.get("Page", 0)
                text = element.get("Text", "")
                bounds = element.get("Bounds", [0, 0, 0, 0])

                # Convert bounds to tuple
                bbox = tuple(float(x) for x in bounds[:4]) if bounds else (0, 0, 0, 0)

                # Extract text blocks (paragraphs, headings, etc.)
                if text and not path.endswith("/Table"):
                    # Determine layout type from path
                    layout_type = "paragraph"
                    if "/H1" in path or "/H2" in path or "/H3" in path:
                        layout_type = "heading"
                    elif "/L" in path:  # List
                        layout_type = "list"
                    elif "/Title" in path:
                        layout_type = "title"

                    # Get font info if available
                    font_info = element.get("Font", {})
                    font_size = element.get("TextSize")

                    text_blocks.append(
                        TextBlock(
                            content=text,
                            page=page,
                            bbox=bbox,
                            font_size=font_size,
                            layout_type=layout_type,
                        )
                    )

                # Extract tables
                elif path.endswith("/Table"):
                    # Tables in Adobe Extract have filePaths pointing to CSV/PNG
                    file_paths = element.get("filePaths", [])

                    # Try to read CSV data if available
                    rows = []
                    headers = None

                    for file_path in file_paths:
                        if file_path.endswith(".csv"):
                            try:
                                with archive.open(file_path) as csv_file:
                                    import csv

                                    reader = csv.reader(
                                        csv_file.read().decode("utf-8").splitlines()
                                    )
                                    csv_rows = list(reader)
                                    if csv_rows:
                                        headers = csv_rows[0]
                                        rows = csv_rows[1:]
                            except Exception as e:
                                logger.warning(f"Failed to parse table CSV: {e}")

                    if rows or text:  # Create table if we have data
                        tables.append(
                            TableBlock(
                                rows=rows if rows else [[]],
                                page=page,
                                bbox=bbox,
                                headers=headers,
                            )
                        )

        return ParsedDocument(
            text_blocks=text_blocks,
            images=images,
            charts=[],
            tables=tables,
            num_pages=num_pages,
            metadata={
                "parser": "adobe_pdf_extract",
                "source": str(original_pdf),
            },
        )
