# Example Scripts

This directory contains example scripts for testing the PDF RAG System with sample PDFs.

## Prerequisites

1. Start the API server:
```bash
uvicorn app.main:app --reload
```

2. Ensure you have sample PDFs in `data/raw/` directory

3. Install httpx if not already installed:
```bash
uv pip install httpx
```

## Scripts

### 1. Upload PDFs (`upload_pdfs.py`)

Uploads all PDF files from the `data/raw/` directory to the API.

```bash
python scripts/upload_pdfs.py
```

**What it does:**
- Scans `data/raw/` for PDF files
- Uploads each PDF to the API
- Displays upload progress and results
- Returns document IDs for uploaded files

**Example output:**
```
Found 3 PDF files to upload

Uploading document1.pdf...
✓ Uploaded document1.pdf - ID: 550e8400-e29b-41d4-a716-446655440000
Uploading document2.pdf...
✓ Uploaded document2.pdf - ID: 660e8400-e29b-41d4-a716-446655440001
...

Upload complete: 3/3 successful
```

### 2. Query Documents (`query_documents.py`)

Query the knowledge base with natural language questions.

**Interactive mode (default):**
```bash
python scripts/query_documents.py
```

Allows you to ask questions interactively. Type your questions and get answers with source citations.

**Sample queries mode:**
```bash
python scripts/query_documents.py --sample
```

Runs a set of predefined sample questions against the knowledge base.

**Example questions:**
- "What is the main topic of this document?"
- "Can you summarize the key findings?"
- "What methodology was used?"
- "What are the main conclusions?"

**Example output:**
```
================================================================================
Question: What is the main topic of this document?
================================================================================

Answer:
The document discusses advanced techniques in machine learning...

Sources (3):
  1. research_paper.pdf (page 3)
     Relevance: 92.5%
     Content: Our research demonstrates that...

Processing time: 1.23s
```

### 3. List Documents (`list_documents.py`)

List all processed documents and inspect their details.

**List all documents:**
```bash
python scripts/list_documents.py
```

**Get details of a specific document:**
```bash
python scripts/list_documents.py <doc_id>
```

**Example output:**
```
================================================================================
Documents in System: 3
================================================================================

ID                                     Filename                       Size       Pages   Chunks
--------------------------------------------------------------------------------
550e8400-e29b-41d4-a716-446655440000   research_paper.pdf            2.3 MB     15      45
660e8400-e29b-41d4-a716-446655440001   technical_doc.pdf             1.8 MB     10      32
...
--------------------------------------------------------------------------------
Total:                                 3 documents 5.1 MB             35      120
```

## Sample Workflow

Here's a typical workflow for testing the system:

```bash
# 1. Start the API server (in a separate terminal)
uvicorn app.main:app --reload

# 2. Upload sample PDFs
python scripts/upload_pdfs.py

# 3. List uploaded documents
python scripts/list_documents.py

# 4. Query the knowledge base (interactive)
python scripts/query_documents.py

# Or run sample queries
python scripts/query_documents.py --sample
```

## Sample PDFs

The `data/raw/` directory contains sample PDFs for testing:

- `1512.03385v1.pdf` - Research paper (Xception architecture)
- `Bodea Brochure.pdf` - Marketing brochure
- `Voucher841127-PARKLOT.pdf` - Invoice/voucher
- `autotagPDFInput.pdf` - Technical document
- `ocrInput.pdf` - OCR test document
- `pdfPropertiesInput.pdf` - PDF properties test

## Troubleshooting

### Connection Error

```
Error: Cannot connect to API at http://localhost:8000
```

**Solution:** Make sure the API server is running:
```bash
uvicorn app.main:app --reload
```

### No Documents Found

```
Warning: No documents found in the system
```

**Solution:** Upload documents first:
```bash
python scripts/upload_pdfs.py
```

### Upload Timeout

If uploads are timing out for large PDFs, the scripts use a 300-second timeout. For very large files, you may need to increase this in the script.

## Customization

You can modify these scripts to:

- Change the API base URL (default: `http://localhost:8000`)
- Add custom metadata to uploads
- Modify sample questions
- Adjust query parameters (top_k, temperature)
- Add filtering or sorting to document lists

## Notes

- All scripts use async/await for efficient I/O operations
- Error handling is included for common failure scenarios
- Scripts provide helpful error messages and usage tips
- Progress indicators show operation status
