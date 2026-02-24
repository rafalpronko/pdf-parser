"""Test Adobe PDF Extract API parser."""

import os
from pathlib import Path

from app.parsers.adobe_pdf_parser import AdobePDFParser

# Set credentials (you'll need to set these as environment variables)
# export PDF_SERVICES_CLIENT_ID="046fdceafbfc40fcba6a4dfdf1195d75"
# export PDF_SERVICES_CLIENT_SECRET="your-secret-here"

# Test PDF path
test_pdf = Path(
    "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
)

print(f"Testing Adobe PDF Extract API Parser")
print(f"File: {test_pdf}")
print(f"File exists: {test_pdf.exists()}")

# Initialize parser
print("\n1. Initializing Adobe PDF parser...")
parser = AdobePDFParser(
    client_id=os.getenv("PDF_SERVICES_CLIENT_ID"),
    client_secret=os.getenv("PDF_SERVICES_CLIENT_SECRET"),
)
print("✓ Parser initialized")

# Parse PDF
print("\n2. Parsing PDF...")
result = parser.parse_pdf(test_pdf)
print(f"✓ Parsing complete!")

# Show results
print(f"\n3. Results:")
print(f"   - Pages: {result.num_pages}")
print(f"   - Text blocks: {len(result.text_blocks)}")
print(f"   - Images: {len(result.images)}")
print(f"   - Tables: {len(result.tables)}")
print(f"   - Charts: {len(result.charts)}")

# Show first text block
if result.text_blocks:
    print(f"\n4. First text block (Page {result.text_blocks[0].page}):")
    print(f"   Layout: {result.text_blocks[0].layout_type}")
    print(f"   {result.text_blocks[0].content[:500]}...")

# Show first table
if result.tables:
    print(f"\n5. First table (Page {result.tables[0].page}):")
    print(f"   Headers: {result.tables[0].headers}")
    print(f"   Rows: {len(result.tables[0].rows)}")
    if result.tables[0].rows:
        print(f"   First row: {result.tables[0].rows[0]}")
