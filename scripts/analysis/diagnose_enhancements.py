"""Diagnose which enhancement is causing the problem."""

import asyncio
import os
from pathlib import Path

from app.models.query import QueryRequest
from app.parsers.adobe_pdf_parser import AdobePDFParser
from app.services.query_service import QueryService
from app.storage.vector_store import VectorStore

# Set Adobe credentials
os.environ["PDF_SERVICES_CLIENT_ID"] = "046fdceafbfc40fcba6a4dfdf1195d75"
os.environ["PDF_SERVICES_CLIENT_SECRET"] = "p8e-AS99RVT34WM6K-Rpqyt3ix0ecUG2LUYf"


async def test_configuration(config_name: str, hybrid: bool, expansion: bool, reranking: bool):
    """Test a specific configuration of enhancements."""
    # Set environment variables
    os.environ["ENABLE_HYBRID_SEARCH"] = "true" if hybrid else "false"
    os.environ["ENABLE_QUERY_EXPANSION"] = "true" if expansion else "false"
    os.environ["ENABLE_RERANKING"] = "true" if reranking else "false"

    # Reload settings
    from app.config import reload_settings
    reload_settings()

    # Initialize Query Service
    query_service = QueryService()

    # Execute query
    question = "Jak i kiedy należy opłacać składki?"
    query_request = QueryRequest(
        question=question,
        top_k=5,
        include_sources=True,
        include_visual=False,
    )

    result = await query_service.query(query_request)

    # Check result
    key_phrases = [
        "składka",
        "gotówką",
        "kartą płatniczą",
        "przelewem bankowym",
        "zawarciu umowy",
    ]

    found_count = sum(1 for phrase in key_phrases if phrase.lower() in result.answer.lower())
    success = found_count >= 4

    # Check if answer contains problematic TOC text
    toc_text = "Co należy do obowiązków Ubezpieczonego?"
    has_toc = toc_text.lower() in result.answer.lower()

    return {
        "config_name": config_name,
        "hybrid": hybrid,
        "expansion": expansion,
        "reranking": reranking,
        "success": success,
        "found_phrases": found_count,
        "total_phrases": len(key_phrases),
        "has_toc_text": has_toc,
        "answer_preview": result.answer[:150],
    }


async def main():
    print("=" * 80)
    print("ENHANCEMENT DIAGNOSTICS")
    print("=" * 80)
    print("\nThis test will run the same query with different enhancement combinations")
    print("to identify which enhancement is causing incorrect results.\n")

    # Document is already indexed from previous test
    # We're just testing different query configurations

    configs = [
        ("Baseline (None)", False, False, False),
        ("Only Hybrid Search", True, False, False),
        ("Only Query Expansion", False, True, False),
        ("Only Reranking", False, False, True),
        ("Hybrid + Expansion", True, True, False),
        ("Hybrid + Reranking", True, False, True),
        ("Expansion + Reranking", False, True, True),
        ("All Enabled", True, True, True),
    ]

    results = []
    for config_name, hybrid, expansion, reranking in configs:
        print(f"\nTesting: {config_name}")
        print(f"  Hybrid={hybrid}, Expansion={expansion}, Reranking={reranking}")

        result = await test_configuration(config_name, hybrid, expansion, reranking)
        results.append(result)

        status = "✓ SUCCESS" if result["success"] else "✗ FAILURE"
        print(f"  {status}: Found {result['found_phrases']}/{result['total_phrases']} phrases")
        if result["has_toc_text"]:
            print(f"  ⚠ WARNING: Answer contains TOC text")
        print(f"  Preview: {result['answer_preview']}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n✓ WORKING Configurations:")
    for r in results:
        if r["success"] and not r["has_toc_text"]:
            print(f"  - {r['config_name']}")

    print("\n✗ BROKEN Configurations:")
    for r in results:
        if not r["success"] or r["has_toc_text"]:
            flags = []
            if r["hybrid"]:
                flags.append("Hybrid")
            if r["expansion"]:
                flags.append("Expansion")
            if r["reranking"]:
                flags.append("Reranking")
            flags_str = " + ".join(flags) if flags else "None"
            print(f"  - {r['config_name']} [{flags_str}]")

    # Identify culprit
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    baseline_works = results[0]["success"]
    hybrid_only = results[1]
    expansion_only = results[2]
    reranking_only = results[3]

    if baseline_works:
        print("\n✓ Baseline (no enhancements) works correctly")

        culprits = []
        if not hybrid_only["success"] or hybrid_only["has_toc_text"]:
            culprits.append("Hybrid Search (BM25)")
        if not expansion_only["success"] or expansion_only["has_toc_text"]:
            culprits.append("Query Expansion")
        if not reranking_only["success"] or reranking_only["has_toc_text"]:
            culprits.append("Reranking")

        if culprits:
            print(f"\n⚠ PROBLEM IDENTIFIED:")
            for culprit in culprits:
                print(f"  - {culprit} is causing incorrect results")
        else:
            print("\n✓ All individual enhancements work correctly")
            print("⚠ Problem only occurs when combining enhancements")
    else:
        print("\n✗ Even baseline doesn't work - there may be a deeper issue")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
