"""Debug what Adobe PDF Extract extracts from the PDF."""

import os
from pathlib import Path

from app.parsers.adobe_pdf_parser import AdobePDFParser

# Set credentials
os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"

# Test PDF
test_pdf = Path(
    "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
)

print("Parsing PDF with Adobe...")
parser = AdobePDFParser()
result = parser.parse_pdf(test_pdf)

print(f"\nTotal pages: {result.num_pages}")
print(f"Total text blocks: {len(result.text_blocks)}")
print(f"Total tables: {len(result.tables)}")

print("\n" + "=" * 80)
print("FIRST 30 TEXT BLOCKS:")
print("=" * 80)

for i, block in enumerate(result.text_blocks[:30], 1):
    print(f"\n--- Block {i} (Page {block.page}, Type: {block.layout_type}) ---")
    print(block.content)

# Search for blocks containing "składk"
print("\n" + "=" * 80)
print("BLOCKS CONTAINING 'składk':")
print("=" * 80)

skladka_blocks = [b for b in result.text_blocks if "składk" in b.content.lower()]
print(f"\nFound {len(skladka_blocks)} blocks containing 'składk'\n")

for i, block in enumerate(skladka_blocks[:10], 1):
    print(f"\n--- Block {i} (Page {block.page}, Type: {block.layout_type}) ---")
    print(block.content[:500])
