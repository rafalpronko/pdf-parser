#!/usr/bin/env python3
"""Extract full text from Warta PDF to see what parser actually got."""

from pathlib import Path
from app.parsers.rag_anything_parser import RAGAnythingParser

def extract_full_text():
    """Extract and display full text from Warta PDF."""
    
    try:
        parser = RAGAnythingParser()
        pdf_path = Path("data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf")
        
        if not pdf_path.exists():
            print(f"ERROR: File not found: {pdf_path}")
            return
    
    print("=== Parsing PDF ===")
    parsed_doc = parser.parse_pdf(pdf_path)
    
    print(f"Pages: {parsed_doc.num_pages}")
    print(f"Text blocks: {len(parsed_doc.text_blocks)}")
    
    # Extract all text and search for payment terms
    all_text = ""
    for block in parsed_doc.text_blocks:
        all_text += f"\n--- PAGE {block.page} ---\n"
        all_text += block.content
        all_text += "\n"
    
    print(f"\nTotal text length: {len(all_text)} characters")
    
    # Search for key terms
    search_terms = [
        "gotówka", "gotówką",
        "karta płatnicza", "kartą płatniczą", 
        "punkt sprzedaży",
        "dostępność",
        "forma płatności",
        "przy zawarciu umowy",
        "pierwsza rata"
    ]
    
    print("\n=== Searching for key terms in extracted text ===")
    for term in search_terms:
        if term.lower() in all_text.lower():
            print(f"✓ Found: '{term}'")
            # Find context around the term
            text_lower = all_text.lower()
            pos = text_lower.find(term.lower())
            if pos != -1:
                start = max(0, pos - 100)
                end = min(len(all_text), pos + len(term) + 100)
                context = all_text[start:end].replace('\n', ' ')
                print(f"  Context: ...{context}...")
        else:
            print(f"✗ NOT found: '{term}'")
    
    # Save full text to file for inspection
    with open("warta_full_text.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    
    print(f"\nFull text saved to: warta_full_text.txt")
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_full_text()