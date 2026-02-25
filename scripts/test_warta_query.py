#!/usr/bin/env python3
"""Test script for WARTA document query."""

import asyncio
import sys
from pathlib import Path

import httpx

API_BASE_URL = "http://localhost:8000"
WARTA_PDF = (
    "data/raw/WARTA_OWU_AutoCasco_Standard_ACS_C6201_IPID_dla_umow_zawieranych_do_31.03.2022.pdf"
)

EXPECTED_ANSWER = """Pojazd (silnikowy lub przyczepa/naczepa), wraz z jego wyposażeniem, Warta ubezpiecza od:
ü uszkodzenia
ü całkowitego zniszczenia
ü kradzieży w całości lub jego części (z wyłączeniem motorowerów)
będących następstwem wszelkich zdarzeń (tzw. all risks) nieujętych w wyłączeniach odpowiedzialności w OWU
Autocasco Standard to elastyczna oferta kierowana do posiadaczy wszystkich pojazdów bez względu na ich rodzaj, wiek i wartość.
Suma ubezpieczenia odpowiada wartości ubezpieczonego pojazdu w danym momencie trwania umowy.
Szczegółowy opis przedmiotu i zakresu ubezpieczenia zawarty jest w §2 i §3 OWU."""


async def check_api():
    """Check if API is running."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            return response.status_code == 200
        except httpx.ConnectError:
            return False


async def upload_document(pdf_path: Path):
    """Upload WARTA document."""
    print(f"📤 Uploading: {pdf_path.name}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            metadata = {
                "tags": ["warta", "insurance", "test"],
                "description": "WARTA AutoCasco document for testing",
            }

            response = await client.post(
                f"{API_BASE_URL}/api/v1/documents/upload",
                files=files,
                data={"metadata": str(metadata)},
            )

        if response.status_code in [200, 201]:
            data = response.json()
            print("✅ Uploaded successfully!")
            print(f"   Document ID: {data['doc_id']}")
            print(f"   Status: {data['status']}")
            return data["doc_id"]
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None


async def query_document(question: str):
    """Query the document."""
    print(f"\n❓ Question: {question}")
    print("-" * 80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/api/v1/query",
            json={
                "question": question,
                "top_k": 10,  # Increased to get more context
                "temperature": 0.3,  # Lower temperature for more factual answers
                "include_sources": True,
            },
        )

        if response.status_code == 200:
            data = response.json()

            print("\n💬 Answer:")
            print(data["answer"])

            print(f"\n📚 Sources ({len(data['sources'])}):")
            for i, source in enumerate(data["sources"], 1):
                print(f"\n  {i}. {source['filename']} (page {source['page']})")
                print(f"     Relevance: {source['relevance_score']:.2%}")
                print("     Content preview:")
                content = source["chunk_content"][:300].replace("\n", " ")
                print(f"     {content}...")

            print(f"\n⏱️  Processing time: {data['processing_time']:.2f}s")

            return data
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None


def compare_answers(actual: str, expected: str):
    """Compare actual answer with expected answer."""
    print(f"\n{'=' * 80}")
    print("📊 ANSWER COMPARISON")
    print(f"{'=' * 80}")

    # Key phrases to check
    key_phrases = [
        "pojazd",
        "uszkodzenia",
        "całkowitego zniszczenia",
        "kradzieży",
        "all risks",
        "§2 i §3 OWU",
    ]

    actual_lower = actual.lower()

    print("\n✓ Key phrases found:")
    found_count = 0
    for phrase in key_phrases:
        if phrase.lower() in actual_lower:
            print(f"  ✅ '{phrase}'")
            found_count += 1
        else:
            print(f"  ❌ '{phrase}' - MISSING")

    coverage = (found_count / len(key_phrases)) * 100
    print(f"\n📈 Coverage: {found_count}/{len(key_phrases)} ({coverage:.1f}%)")

    if coverage >= 80:
        print("✅ PASS - Answer contains most key information")
    elif coverage >= 50:
        print("⚠️  PARTIAL - Answer is incomplete")
    else:
        print("❌ FAIL - Answer is missing critical information")

    return coverage


async def main():
    """Main test function."""
    print("=" * 80)
    print("🧪 WARTA Document Query Test")
    print("=" * 80)

    # Check API
    print("\n1️⃣  Checking API...")
    if not await check_api():
        print("❌ API is not running!")
        print("   Start it with: uvicorn app.main:app --reload")
        sys.exit(1)
    print("✅ API is running")

    # Check if document exists
    pdf_path = Path(WARTA_PDF)
    if not pdf_path.exists():
        print(f"❌ Document not found: {pdf_path}")
        sys.exit(1)

    # Upload document
    print("\n2️⃣  Uploading document...")
    doc_id = await upload_document(pdf_path)
    if not doc_id:
        sys.exit(1)

    # Wait a moment for processing
    print("\n⏳ Waiting for processing to complete...")
    await asyncio.sleep(2)

    # Query document
    print("\n3️⃣  Querying document...")
    question = "Co jest przedmiotem ubezpieczenia?"
    result = await query_document(question)

    if not result:
        sys.exit(1)

    # Compare with expected answer
    print("\n4️⃣  Comparing with expected answer...")
    coverage = compare_answers(result["answer"], EXPECTED_ANSWER)

    # Final summary
    print(f"\n{'=' * 80}")
    print("📋 TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Document: {pdf_path.name}")
    print(f"Question: {question}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"Sources: {len(result['sources'])}")

    if coverage >= 80:
        print("\n✅ TEST PASSED - System correctly answers the question!")
    else:
        print("\n⚠️  TEST NEEDS IMPROVEMENT - Answer could be more complete")
        print("\nExpected key information:")
        print(EXPECTED_ANSWER)


if __name__ == "__main__":
    asyncio.run(main())
