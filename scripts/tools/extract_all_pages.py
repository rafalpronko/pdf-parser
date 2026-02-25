#!/usr/bin/env python3
"""Extract text from all pages of Warta PDF."""

import sys
from pathlib import Path

# Add app to path
sys.path.append(".")

from app.parsers.rag_anything_parser import RAGAnythingParser


def extract_all_pages():
    """Extract text from all pages."""

    parser = RAGAnythingParser()
    pdf_path = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    print("=== Extracting all pages ===")
    parsed_doc = parser.parse_pdf(pdf_path)

    print(f"Total pages: {parsed_doc.num_pages}")
    print(f"Total text blocks: {len(parsed_doc.text_blocks)}")

    # Show text from each page
    for i, block in enumerate(parsed_doc.text_blocks):
        print(f"\n{'=' * 60}")
        print(f"PAGE {block.page} - Block {i + 1}")
        print("=" * 60)
        print(block.content)

        # Search for payment terms in this block
        payment_terms = ["gotówka", "karta", "punkt sprzedaży", "dostępność", "forma płatności"]
        for term in payment_terms:
            if term.lower() in block.content.lower():
                print(f"\n*** FOUND '{term}' in this block! ***")


if __name__ == "__main__":
    extract_all_pages()
