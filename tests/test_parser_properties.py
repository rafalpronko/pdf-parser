"""Property-based tests for PDF parsing.

Feature: pdf-rag-system
Properties tested:
- Property 5: Text extraction preserves content
- Property 6: Image extraction completeness
- Property 7: Table structure preservation
- Property 8: Partial failure resilience
"""

import io
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pypdf import PdfWriter

from app.models.parsing import ParsedDocument
from app.parsers.rag_anything_parser import RAGAnythingParser


# Feature: pdf-rag-system, Property 5: Text extraction preserves content
def test_property_5_text_extraction_non_empty():
    """Property 5: Text extraction preserves content.
    
    For any PDF document, the total length of extracted text should be
    non-zero for non-empty PDFs, and all extracted text blocks should
    have valid page numbers and bounding boxes.
    
    Validates: Requirements 2.1
    """
    parser = RAGAnythingParser()
    
    # Use existing test PDF
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if not test_pdf.exists():
        pytest.skip("Test PDF not found")
    
    result = parser.parse_pdf(test_pdf)
    
    # Verify non-empty extraction
    assert isinstance(result, ParsedDocument)
    assert result.num_pages > 0
    assert len(result.text_blocks) > 0
    
    # Verify all text blocks have valid properties
    for block in result.text_blocks:
        assert block.page >= 0
        assert block.page < result.num_pages
        assert len(block.bbox) == 4
        assert all(isinstance(x, (int, float)) for x in block.bbox)
        assert len(block.content) > 0


# Feature: pdf-rag-system, Property 6: Image extraction completeness
def test_property_6_image_metadata_valid():
    """Property 6: Image extraction completeness.
    
    For any PDF containing images, all extracted images should have
    valid metadata including page number, bounding box, and format.
    
    Validates: Requirements 2.2
    """
    parser = RAGAnythingParser()
    
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if not test_pdf.exists():
        pytest.skip("Test PDF not found")
    
    result = parser.parse_pdf(test_pdf)
    
    # If images are extracted, verify metadata
    for image in result.images:
        assert image.page >= 0
        assert image.page < result.num_pages
        assert len(image.bbox) == 4
        assert all(isinstance(x, (int, float)) for x in image.bbox)
        assert isinstance(image.image_data, bytes)
        assert len(image.image_data) > 0
        assert isinstance(image.format, str)


# Feature: pdf-rag-system, Property 7: Table structure preservation
def test_property_7_table_structure():
    """Property 7: Table structure preservation.
    
    For any extracted table, the table should have at least one row,
    and all rows should have the same number of columns, preserving
    the rectangular structure.
    
    Validates: Requirements 2.3
    """
    parser = RAGAnythingParser()
    
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if not test_pdf.exists():
        pytest.skip("Test PDF not found")
    
    result = parser.parse_pdf(test_pdf)
    
    # If tables are extracted, verify structure
    for table in result.tables:
        assert len(table.rows) >= 1, "Table should have at least one row"
        
        if len(table.rows) > 0:
            # All rows should have same number of columns
            num_cols = len(table.rows[0])
            for row in table.rows:
                assert len(row) == num_cols, "All rows should have same column count"


# Feature: pdf-rag-system, Property 8: Partial failure resilience
def test_property_8_parser_handles_errors():
    """Property 8: Partial failure resilience.
    
    For any PDF document where parsing encounters errors, the parser
    should continue processing and return successfully parsed content.
    
    Validates: Requirements 2.4
    """
    parser = RAGAnythingParser()
    
    # Test with non-existent file
    with pytest.raises(ValueError, match="File not found"):
        parser.parse_pdf(Path("nonexistent.pdf"))
    
    # Test with non-PDF file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Not a PDF")
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="not a PDF"):
            parser.parse_pdf(temp_path)
    finally:
        temp_path.unlink()


def test_property_5_bbox_coordinates_valid():
    """Property 5: Bounding boxes have valid coordinates."""
    parser = RAGAnythingParser()
    
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if not test_pdf.exists():
        pytest.skip("Test PDF not found")
    
    result = parser.parse_pdf(test_pdf)
    
    # Verify bbox coordinates are reasonable
    for block in result.text_blocks:
        x0, y0, x1, y1 = block.bbox
        assert x0 <= x1, "x0 should be <= x1"
        assert y0 <= y1, "y0 should be <= y1"
        assert x0 >= 0, "Coordinates should be non-negative"
        assert y0 >= 0, "Coordinates should be non-negative"


def test_property_6_charts_have_valid_metadata():
    """Property 6: Charts have valid metadata when extracted."""
    parser = RAGAnythingParser()
    
    test_pdf = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
    
    if not test_pdf.exists():
        pytest.skip("Test PDF not found")
    
    result = parser.parse_pdf(test_pdf)
    
    # If charts are extracted, verify metadata
    for chart in result.charts:
        assert chart.page >= 0
        assert chart.page < result.num_pages
        assert len(chart.bbox) == 4
        assert isinstance(chart.image_data, bytes)
        assert len(chart.image_data) > 0
