"""Test OpenAI GPT-4o Vision PDF parser."""
from pathlib import Path

from app.parsers.openai_pdf_parser import OpenAIPDFParser

# Test PDF path
test_pdf = Path(
    "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
)

print(f"Testing OpenAI GPT-4o Vision PDF Parser")
print(f"File: {test_pdf}")
print(f"File exists: {test_pdf.exists()}")

# Initialize parser
print("\n1. Initializing OpenAI parser...")
parser = OpenAIPDFParser(model="gpt-4o-mini")  # Use mini for testing (cheaper/faster)
print("✓ Parser initialized")

# Parse PDF (only first 2 pages for testing)
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
    print(f"   {result.text_blocks[0].content[:500]}...")

# Show first table
if result.tables:
    print(f"\n5. First table (Page {result.tables[0].page}):")
    print(f"   Headers: {result.tables[0].headers}")
    print(f"   Rows: {len(result.tables[0].rows)}")
    if result.tables[0].rows:
        print(f"   First row: {result.tables[0].rows[0]}")
