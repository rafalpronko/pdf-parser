"""Direct parser test to see exact error."""

from pathlib import Path

from app.parsers.rag_anything_parser import RAGAnythingParser

# Test PDF path
test_pdf = Path(
    "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
)

print(f"Testing parser with: {test_pdf}")
print(f"File exists: {test_pdf.exists()}")

# Initialize parser
print("\n1. Initializing parser...")
parser = RAGAnythingParser()
print("✓ Parser initialized")

# Parse PDF
print("\n2. Parsing PDF...")
result = parser.parse_pdf(test_pdf)
print("✓ Parsing complete!")

# Show results
print("\n3. Results:")
print(f"   - Pages: {result.num_pages}")
print(f"   - Text blocks: {len(result.text_blocks)}")
print(f"   - Images: {len(result.images)}")
print(f"   - Tables: {len(result.tables)}")
print(f"   - Charts: {len(result.charts)}")

# Show first text block
if result.text_blocks:
    print("\n4. First text block:")
    print(f"   Page: {result.text_blocks[0].page}")
    print(f"   Content: {result.text_blocks[0].content[:200]}...")
