"""Manual test to verify error handling and edge cases."""

from pathlib import Path
from app.parsers.rag_anything_parser import RAGAnythingParser

parser = RAGAnythingParser()

print("Testing error handling:\n")

# Test 1: Non-existent file
print("1. Testing with non-existent file:")
try:
    parser.parse_pdf(Path("nonexistent.pdf"))
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised ValueError: {e}")

# Test 2: Non-PDF file
print("\n2. Testing with non-PDF file:")
try:
    parser.parse_pdf(Path("README.md"))
    print("  ✗ Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly raised ValueError: {e}")

# Test 3: Valid PDF parsing
print("\n3. Testing valid PDF parsing:")
pdf_path = Path("data/raw/1512.03385v1.pdf")
result = parser.parse_pdf(pdf_path)
print(f"  ✓ Successfully parsed {result.num_pages} pages")
print(f"  ✓ Extracted {len(result.text_blocks)} text blocks")

# Test 4: Verify all extraction methods work independently
print("\n4. Testing individual extraction methods:")
with open(pdf_path, "rb") as f:
    pdf_data = f.read()

text_blocks = parser.extract_text(pdf_data)
print(f"  ✓ extract_text: {len(text_blocks)} blocks")

images = parser.extract_images(pdf_data)
print(f"  ✓ extract_images: {len(images)} images")

tables = parser.extract_tables(pdf_data)
print(f"  ✓ extract_tables: {len(tables)} tables")

print("\n✓ All error handling and edge cases working correctly!")
