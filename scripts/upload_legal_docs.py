#!/usr/bin/env python3
"""Script to upload legal/insurance documents with optimized chunking."""

import asyncio
from pathlib import Path

import httpx

API_BASE_URL = "http://localhost:8000"


async def upload_legal_document(pdf_path: Path):
    """Upload a legal document with special handling.

    Args:
        pdf_path: Path to PDF file
    """
    print(f"Uploading legal document: {pdf_path.name}...")

    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            metadata = {
                "tags": ["legal", "insurance", "polish"],
                "description": f"Legal document: {pdf_path.name}",
            }

            response = await client.post(
                f"{API_BASE_URL}/api/v1/documents/upload",
                files=files,
                data={"metadata": str(metadata)},
            )

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Uploaded successfully - ID: {data['doc_id']}")
            return data
        else:
            print(f"✗ Failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return None


async def main():
    """Upload WARTA document."""
    pdf_path = Path(
        "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
    )

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return

    # Check API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            if response.status_code != 200:
                print("Error: API is not responding")
                return
        except httpx.ConnectError:
            print(f"Error: Cannot connect to API at {API_BASE_URL}")
            print("Start the server: uvicorn app.main:app --reload")
            return

    await upload_legal_document(pdf_path)

    print("\n✅ Document uploaded! Now you can query it:")
    print("   python scripts/query_documents.py")
    print("\nExample questions:")
    print("   - Co jest przedmiotem ubezpieczenia?")
    print("   - Jakie są wyłączenia odpowiedzialności?")
    print("   - Gdzie obowiązuje ubezpieczenie?")


if __name__ == "__main__":
    asyncio.run(main())
