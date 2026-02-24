#!/usr/bin/env python3
"""Script to query documents with sample questions."""

import asyncio
import sys

import httpx


API_BASE_URL = "http://localhost:8000"

# Sample questions relevant to common PDF types
SAMPLE_QUESTIONS = [
    "What is the main topic of this document?",
    "Can you summarize the key findings?",
    "What are the main conclusions?",
    "What methodology was used?",
    "Who are the authors?",
    "What is the date of this document?",
    "What are the recommendations?",
    "What data or statistics are presented?",
]


async def query_api(client: httpx.AsyncClient, question: str, top_k: int = 5) -> dict:
    """Send a query to the API.
    
    Args:
        client: HTTP client
        question: Question to ask
        top_k: Number of results to retrieve
        
    Returns:
        Response data from API
    """
    response = await client.post(
        f"{API_BASE_URL}/api/v1/query",
        json={
            "question": question,
            "top_k": top_k,
            "temperature": 0.7,
            "include_sources": True
        }
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def print_response(question: str, response: dict):
    """Print query response in a formatted way.
    
    Args:
        question: The question asked
        response: API response data
    """
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}")
    print(f"\nAnswer:\n{response['answer']}\n")
    
    if response.get('sources'):
        print(f"Sources ({len(response['sources'])}):")
        for i, source in enumerate(response['sources'], 1):
            print(f"\n  {i}. {source['filename']} (page {source['page']})")
            print(f"     Relevance: {source['relevance_score']:.2%}")
            print(f"     Content: {source['chunk_content'][:100]}...")
    
    print(f"\nProcessing time: {response['processing_time']:.2f}s")


async def interactive_mode(client: httpx.AsyncClient):
    """Run in interactive mode, allowing user to ask questions.
    
    Args:
        client: HTTP client
    """
    print("\n" + "="*80)
    print("Interactive Query Mode")
    print("="*80)
    print("Enter your questions (or 'quit' to exit)")
    print()
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            response = await query_api(client, question)
            if response:
                print_response(question, response)
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


async def sample_queries_mode(client: httpx.AsyncClient):
    """Run sample queries from predefined list.
    
    Args:
        client: HTTP client
    """
    print("\n" + "="*80)
    print("Running Sample Queries")
    print("="*80)
    
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n[{i}/{len(SAMPLE_QUESTIONS)}] Querying...")
        response = await query_api(client, question)
        
        if response:
            print_response(question, response)
        
        if i < len(SAMPLE_QUESTIONS):
            await asyncio.sleep(1)  # Small delay between queries


async def main():
    """Main function to run query script."""
    mode = "interactive"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sample":
            mode = "sample"
        elif sys.argv[1] == "--help":
            print("Usage: python query_documents.py [--sample|--interactive]")
            print()
            print("Modes:")
            print("  --interactive  Ask questions interactively (default)")
            print("  --sample       Run predefined sample questions")
            sys.exit(0)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
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
        
        # Check if there are any documents
        response = await client.get(f"{API_BASE_URL}/api/v1/documents")
        if response.status_code == 200:
            docs = response.json()
            if not docs:
                print("Warning: No documents found in the system")
                print("Upload some documents first:")
                print("  python scripts/upload_pdfs.py")
                sys.exit(1)
            print(f"Found {len(docs)} documents in the system")
        
        # Run in selected mode
        if mode == "sample":
            await sample_queries_mode(client)
        else:
            await interactive_mode(client)


if __name__ == "__main__":
    asyncio.run(main())
