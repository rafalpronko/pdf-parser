#!/usr/bin/env python3
"""Script to upload all PDFs from data/raw/ directory to the API."""

import asyncio
import sys
from pathlib import Path

import httpx


API_BASE_URL = "http://localhost:8000"
PDF_DIR = Path("data/raw")


async def upload_pdf(client: httpx.AsyncClient, pdf_path: Path) -> dict:
    """Upload a single PDF file to the API.
    
    Args:
        client: HTTP client
        pdf_path: Path to PDF file
        
    Returns:
        Response data from API
    """
    print(f"Uploading {pdf_path.name}...")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        metadata = {
            "tags": ["sample", "test"],
            "description": f"Sample PDF: {pdf_path.name}"
        }
        
        response = await client.post(
            f"{API_BASE_URL}/api/v1/documents/upload",
            files=files,
            data={"metadata": str(metadata)}
        )
        
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Uploaded {pdf_path.name} - ID: {data['doc_id']}")
        return data
    else:
        print(f"✗ Failed to upload {pdf_path.name}: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


async def main():
    """Upload all PDFs from data/raw/ directory."""
    if not PDF_DIR.exists():
        print(f"Error: Directory {PDF_DIR} does not exist")
        sys.exit(1)
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF files to upload\n")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Check if API is running
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            if response.status_code != 200:
                print("Error: API is not responding correctly")
                sys.exit(1)
        except httpx.ConnectError:
            print(f"Error: Cannot connect to API at {API_BASE_URL}")
            print("Make sure the API server is running:")
            print("  uvicorn app.main:app --reload")
            sys.exit(1)
        
        # Upload all PDFs
        results = []
        for pdf_path in pdf_files:
            result = await upload_pdf(client, pdf_path)
            if result:
                results.append(result)
            await asyncio.sleep(0.5)  # Small delay between uploads
        
        print(f"\n{'='*60}")
        print(f"Upload complete: {len(results)}/{len(pdf_files)} successful")
        print(f"{'='*60}")
        
        if results:
            print("\nUploaded documents:")
            for result in results:
                print(f"  - {result['filename']}: {result['doc_id']}")


if __name__ == "__main__":
    asyncio.run(main())
