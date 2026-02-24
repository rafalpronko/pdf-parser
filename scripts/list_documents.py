#!/usr/bin/env python3
"""Script to list and inspect processed documents."""

import asyncio
import sys
from datetime import datetime

import httpx


API_BASE_URL = "http://localhost:8000"


async def list_documents(client: httpx.AsyncClient) -> list[dict]:
    """List all documents in the system.
    
    Args:
        client: HTTP client
        
    Returns:
        List of document info dictionaries
    """
    response = await client.get(f"{API_BASE_URL}/api/v1/documents")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []


async def get_document_details(client: httpx.AsyncClient, doc_id: str) -> dict:
    """Get detailed information about a document.
    
    Args:
        client: HTTP client
        doc_id: Document ID
        
    Returns:
        Document details dictionary
    """
    response = await client.get(f"{API_BASE_URL}/api/v1/documents/{doc_id}")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_datetime(dt_str: str) -> str:
    """Format datetime string in readable format.
    
    Args:
        dt_str: ISO format datetime string
        
    Returns:
        Formatted datetime string
    """
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def print_document_summary(docs: list[dict]):
    """Print summary of all documents.
    
    Args:
        docs: List of document info dictionaries
    """
    print(f"\n{'='*80}")
    print(f"Documents in System: {len(docs)}")
    print(f"{'='*80}\n")
    
    if not docs:
        print("No documents found.")
        return
    
    # Calculate totals
    total_size = sum(doc['file_size'] for doc in docs)
    total_pages = sum(doc['num_pages'] for doc in docs)
    total_chunks = sum(doc['num_chunks'] for doc in docs)
    
    print(f"{'ID':<38} {'Filename':<30} {'Size':<10} {'Pages':<7} {'Chunks':<7}")
    print("-" * 80)
    
    for doc in docs:
        doc_id = doc['doc_id'][:36]
        filename = doc['filename'][:28] + '..' if len(doc['filename']) > 30 else doc['filename']
        size = format_size(doc['file_size'])
        pages = str(doc['num_pages'])
        chunks = str(doc['num_chunks'])
        
        print(f"{doc_id:<38} {filename:<30} {size:<10} {pages:<7} {chunks:<7}")
    
    print("-" * 80)
    print(f"{'Total:':<38} {len(docs)} documents {format_size(total_size):<10} {total_pages:<7} {total_chunks:<7}")
    print()


def print_document_details(doc: dict):
    """Print detailed information about a document.
    
    Args:
        doc: Document details dictionary
    """
    print(f"\n{'='*80}")
    print(f"Document Details")
    print(f"{'='*80}\n")
    
    print(f"ID:           {doc['doc_id']}")
    print(f"Filename:     {doc['filename']}")
    print(f"File Size:    {format_size(doc['file_size'])}")
    print(f"Pages:        {doc['num_pages']}")
    print(f"Chunks:       {doc['num_chunks']}")
    print(f"Created:      {format_datetime(doc['created_at'])}")
    
    if doc.get('tags'):
        print(f"Tags:         {', '.join(doc['tags'])}")
    
    print()


async def main():
    """Main function to list and inspect documents."""
    async with httpx.AsyncClient(timeout=30.0) as client:
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
        
        # List all documents
        docs = await list_documents(client)
        print_document_summary(docs)
        
        # If a document ID is provided, show details
        if len(sys.argv) > 1:
            doc_id = sys.argv[1]
            print(f"Fetching details for document: {doc_id}")
            details = await get_document_details(client, doc_id)
            if details:
                print_document_details(details)
        elif docs:
            print("Tip: To see details of a specific document, run:")
            print(f"  python scripts/list_documents.py <doc_id>")
            print(f"\nExample:")
            print(f"  python scripts/list_documents.py {docs[0]['doc_id']}")


if __name__ == "__main__":
    asyncio.run(main())
