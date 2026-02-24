# PDF RAG System

Universal PDF parsing and knowledge base system with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- 📄 Advanced PDF parsing with MinerU (RAG-Anything framework)
- 🔍 **Hybrid Search** - Vector + BM25 keyword search with RRF fusion
- 🎯 **Cross-Encoder Reranking** - Improved result relevance
- 📝 **Query Expansion** - HyDE and multi-query for better retrieval
- 💬 Conversational interface powered by OpenAI GPT-4o-mini
- 🚀 FastAPI REST API with automatic OpenAPI documentation
- ✅ Comprehensive data validation with Pydantic
- 🧪 Property-based testing with Hypothesis

## Requirements

- Python 3.11+
- OpenAI API key
- MinerU (magic-pdf) for PDF parsing

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Set up the project

```bash
# Clone the repository
git clone <repository-url>
cd pdf-rag-system

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Copy environment template and configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Configuration

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_FILE_SIZE=52428800
VECTOR_DB_PATH=./data/vectordb
UPLOAD_DIR=./data/uploads

# RAG Enhancements (recommended)
ENABLE_HYBRID_SEARCH=true
ENABLE_QUERY_EXPANSION=true
ENABLE_RERANKING=true
EXPANSION_METHOD=multi-query
```

## Usage

### Start the API server

```bash
./start_server.sh
# Or manually:
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Start the Chat UI (Optional)

For a visual interface to test the RAG system:

```bash
./start_frontend.sh
```

The chat UI will be available at `http://localhost:3000`

Features:
- 💬 Interactive chat interface
- 📚 View uploaded documents
- 📎 See source citations for answers
- ⏱️ Monitor query processing time

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example API Usage

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Upload a PDF Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -F "file=@document.pdf" \
  -F "metadata={\"tags\":[\"research\",\"ml\"],\"description\":\"Research paper on ML\"}"
```

Response:
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "status": "completed",
  "message": "Document processed successfully",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### List All Documents

```bash
curl "http://localhost:8000/api/v1/documents?skip=0&limit=10"
```

#### Get Document Details

```bash
curl "http://localhost:8000/api/v1/documents/550e8400-e29b-41d4-a716-446655440000"
```

#### Query the Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings of the research?",
    "top_k": 5,
    "temperature": 0.7,
    "include_sources": true
  }'
```

Response:
```json
{
  "answer": "The main findings include...",
  "sources": [
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "document.pdf",
      "page": 3,
      "chunk_content": "Our research demonstrates...",
      "relevance_score": 0.92
    }
  ],
  "processing_time": 1.23
}
```

#### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/550e8400-e29b-41d4-a716-446655440000"
```

## Development

### Development Guidelines

This project follows Python best practices:

- **Type Hints**: All functions and classes include comprehensive type annotations
- **Code Formatting**: Use `ruff` for consistent code style
- **Testing**: Property-based testing with Hypothesis for robust validation
- **Documentation**: Docstrings for all public APIs
- **Error Handling**: Explicit error handling with custom exceptions

### Code Quality

This project uses `ruff` for linting and formatting:

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/test_models.py

# Run property-based tests only
uv run pytest tests/ -k "property"

# Run with verbose output
uv run pytest -v
```

### Type Checking

All code includes comprehensive type hints. Use mypy for type checking:

```bash
uv run mypy app/
```

### Adding New Features

When extending the system:

1. **New Parsers**: Implement the `BaseParser` interface in `app/parsers/`
2. **New Embedding Models**: Implement the `BaseEmbedder` interface
3. **New LLM Providers**: Implement the `BaseLLMClient` interface
4. **New Vector Stores**: Implement the `BaseVectorStore` interface

See the design document in `.kiro/specs/pdf-rag-system/design.md` for detailed extension points.

## Project Structure

```
pdf-rag-system/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic
│   ├── parsers/             # PDF parsing
│   ├── processing/          # Document processing
│   ├── storage/             # Vector database
│   └── clients/             # External API clients
├── docs/                    # Project documentation
├── scripts/                 # Utility and debug scripts
│   ├── analysis/            # Analysis tools
│   ├── debug/               # Debugging scripts
│   ├── tools/               # Helper tools
│   └── utils/               # Common utilities
├── tests/                   # Test suite
│   └── adhoc/               # Ad-hoc integration tests
├── data/
│   ├── raw/                 # Sample PDFs
│   ├── uploads/             # Uploaded documents
│   └── vectordb/            # Vector database storage
├── logs/                    # Logs and debug output
├── frontend/                # React chat UI
│   ├── src/
│   │   ├── App.js           # Main chat component
│   │   └── App.css          # Styling
│   └── package.json
├── pyproject.toml           # Project configuration
├── start_server.sh          # Backend startup script
├── start_frontend.sh        # Frontend startup script
└── README.md
```

## Architecture

The system follows a layered architecture:

1. **API Layer**: FastAPI endpoints with request/response validation
2. **Service Layer**: Business logic and workflow orchestration
3. **Processing Layer**: PDF parsing, chunking, embedding generation
4. **Data Layer**: Vector database and file storage

### Document Processing Pipeline

```
PDF Upload → Validation → Parsing → Chunking → Embedding → Vector Storage
```

1. **Upload**: File validation (format, size)
2. **Parsing**: Extract text, images, tables using RAG-Anything
3. **Chunking**: Split into semantic chunks with overlap
4. **Embedding**: Generate vector embeddings with OpenAI
5. **Storage**: Store in ChromaDB with metadata

### Query Pipeline

```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
```

1. **Embedding**: Convert query to vector
2. **Search**: Find top-k similar chunks
3. **Context**: Retrieve relevant document sections
4. **Generation**: Generate answer with GPT-4o-mini
5. **Response**: Return answer with source citations

## Troubleshooting

### Common Issues

#### OpenAI API Key Not Found

```
Error: OPENAI_API_KEY environment variable not set
```

**Solution**: Ensure your `.env` file contains a valid OpenAI API key:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

#### PDF Parsing Fails

```
Error: Failed to parse PDF document
```

**Solutions**:
- Ensure the PDF is not corrupted
- Check if the PDF is password-protected (not supported)
- Verify the file size is under the limit (default 50MB)

#### Vector Database Connection Error

```
Error: Failed to connect to vector database
```

**Solution**: Ensure the vector database directory exists and has write permissions:
```bash
mkdir -p data/vectordb
chmod 755 data/vectordb
```

#### Out of Memory During Processing

**Solution**: Reduce chunk size or process smaller documents:
```env
CHUNK_SIZE=256
CHUNK_OVERLAP=25
```

#### Rate Limit Errors from OpenAI

```
Error: Rate limit exceeded
```

**Solution**: The system automatically retries with exponential backoff. If the issue persists, check your OpenAI API usage limits.

### Debug Mode

Enable debug logging for troubleshooting:

```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

### Getting Help

- Check the [API documentation](http://localhost:8000/docs) for endpoint details
- Review test files in `tests/` for usage examples
- See design document in `.kiro/specs/pdf-rag-system/design.md` for architecture details

## Performance Considerations

- **Batch Processing**: Embeddings are generated in batches for efficiency
- **Async Operations**: All I/O operations use async/await
- **Connection Pooling**: HTTP connections to OpenAI are reused
- **Caching**: Consider implementing embedding caching for identical chunks

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `uv run pytest`
2. Code is formatted: `uv run ruff format .`
3. No linting errors: `uv run ruff check .`
4. Type hints are included for new code
5. Property-based tests are added for new features where applicable

Submit PRs with a clear description of changes and any relevant issue numbers.
