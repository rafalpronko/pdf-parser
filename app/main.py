"""FastAPI application entry point."""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import os

from app.config import get_settings
from app.logging_config import get_logger, set_request_id, clear_request_id, setup_logging
from app.models.document import DocumentInfo, DocumentMetadata, DocumentUploadResponse
from app.models.error import ErrorResponse
from app.models.query import QueryRequest, QueryResponse
from app.services.document_service import DocumentProcessingError, DocumentService
from app.services.query_service import QueryService
from app.storage.file_storage import FileValidationError

# Setup logging configuration
setup_logging()
logger = get_logger(__name__)

# Load and validate configuration at startup
settings = get_settings()

# Global service instances
document_service: DocumentService | None = None
query_service: QueryService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    global document_service, query_service

    logger.info("Starting PDF RAG System...")
    logger.info(
        f"Configuration: chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}"
    )

    # Initialize services
    document_service = DocumentService()
    query_service = QueryService(document_service=document_service)

    logger.info("PDF RAG System started successfully")

    yield

    # Shutdown
    logger.info("Shutting down PDF RAG System...")

    if document_service:
        await document_service.close()
    if query_service:
        await query_service.close()

    logger.info("PDF RAG System shut down successfully")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Universal PDF parsing and knowledge base system with RAG capabilities",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID and error handling middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID tracking and error handling."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Set request ID in logging context
    set_request_id(request_id)

    try:
        logger.info(f"Request started: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}"
        )
        return response
    except Exception as e:
        logger.error(
            f"Unhandled exception: {str(e)}",
            exc_info=True,
            extra={"path": request.url.path},
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=str(e),
                timestamp=datetime.now(UTC),
                request_id=request_id,
            ).model_dump(mode="json"),
        )
    finally:
        # Clear request ID from context
        clear_request_id()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Extract detailed field-level errors
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']}")

    detail = "; ".join(errors)

    logger.warning(
        f"Validation error for request {request_id}: {detail}",
        extra={"request_id": request_id, "path": request.url.path},
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=detail,
            timestamp=datetime.now(UTC),
            request_id=request_id,
        ).model_dump(mode="json"),
    )


@app.exception_handler(FileValidationError)
async def file_validation_exception_handler(request: Request, exc: FileValidationError):
    """Handle file validation errors."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.warning(
        f"File validation error for request {request_id}: {str(exc)}",
        extra={"request_id": request_id},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="File Validation Error",
            detail=str(exc),
            timestamp=datetime.now(UTC),
            request_id=request_id,
        ).model_dump(mode="json"),
    )


@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    """Handle document processing errors."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.error(
        f"Document processing error for request {request_id}: {str(exc)}",
        extra={"request_id": request_id},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Document Processing Error",
            detail=str(exc),
            timestamp=datetime.now(UTC),
            request_id=request_id,
        ).model_dump(mode="json"),
    )


# API Endpoints


@app.get("/health")
async def health_check():
    """Health check endpoint with multimodal capabilities info.

    Returns:
        dict: Health status and multimodal capabilities information
    """
    return {
        "status": "healthy",
        "service": "RAG-Anything Multimodal System",
        "version": settings.api_version,
        "capabilities": {
            "multimodal": settings.enable_multimodal_chunking,
            "vlm_enabled": settings.enable_vlm,
            "vlm_provider": settings.vlm_provider,
            "vision_encoder": settings.vision_encoder,
            "rag_anything": settings.use_rag_anything,
        },
    }


@app.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a PDF document",
    description="Upload a PDF file to be parsed, chunked, embedded, and stored in the knowledge base.",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload"),
    tags: str = "",
    description: str = "",
) -> DocumentUploadResponse:
    """Upload and process a PDF document.

    This endpoint accepts a PDF file and processes it through the complete pipeline:
    1. Validates file format and size
    2. Parses PDF to extract text, images, and tables
    3. Chunks text into semantic segments
    4. Generates embeddings for chunks
    5. Stores embeddings in vector database

    Args:
        request: FastAPI request object
        file: Uploaded PDF file
        tags: Comma-separated tags for the document
        description: Optional description of the document

    Returns:
        DocumentUploadResponse with document ID and processing status

    Raises:
        HTTPException: If file validation or processing fails
    """
    if not document_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not initialized",
        )

    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Read file content
    file_content = await file.read()

    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    # Create metadata
    metadata = DocumentMetadata(
        filename=file.filename,
        content_type="application/pdf",
        tags=tag_list,
        description=description if description else None,
    )

    # Process document
    logger.info(f"Processing upload request for '{file.filename}'")

    try:
        result = await document_service.process_document(
            file_content=file_content,
            metadata=metadata,
        )

        logger.info(f"Document '{result.doc_id}' uploaded successfully")
        return result

    except FileValidationError:
        # Re-raise to be handled by exception handler
        raise

    except DocumentProcessingError:
        # Re-raise to be handled by exception handler
        raise

    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@app.get(
    "/api/v1/documents",
    response_model=list[DocumentInfo],
    summary="List all documents",
    description="Retrieve a paginated list of all processed documents in the knowledge base.",
)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
) -> list[DocumentInfo]:
    """List all processed documents.

    Args:
        skip: Number of documents to skip (for pagination)
        limit: Maximum number of documents to return (max 100)

    Returns:
        List of DocumentInfo objects

    Raises:
        HTTPException: If service is unavailable
    """
    if not document_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not initialized",
        )

    # Validate pagination parameters
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="skip must be non-negative",
        )

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 100",
        )

    try:
        documents = await document_service.list_documents(skip=skip, limit=limit)
        logger.info(f"Listed {len(documents)} documents (skip={skip}, limit={limit})")
        return documents

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@app.get(
    "/api/v1/documents/{doc_id}",
    response_model=DocumentInfo,
    summary="Get document details",
    description="Retrieve detailed information about a specific document.",
)
async def get_document(doc_id: str) -> DocumentInfo:
    """Get document details.

    Args:
        doc_id: Document identifier

    Returns:
        DocumentInfo with document details

    Raises:
        HTTPException: If document not found or service unavailable
    """
    if not document_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not initialized",
        )

    try:
        document = await document_service.get_document(doc_id)
        logger.info(f"Retrieved document '{doc_id}'")
        return document

    except ValueError as e:
        logger.warning(f"Document not found: {doc_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except Exception as e:
        logger.error(f"Error retrieving document '{doc_id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}",
        )


@app.delete(
    "/api/v1/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Delete a document and all associated data (embeddings, chunks, files).",
)
async def delete_document(doc_id: str):
    """Delete a document.

    Performs cascade deletion:
    - Removes embeddings from vector store
    - Deletes file from storage
    - Removes metadata

    Args:
        doc_id: Document identifier

    Raises:
        HTTPException: If document not found or deletion fails
    """
    if not document_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service not initialized",
        )

    try:
        await document_service.delete_document(doc_id)
        logger.info(f"Deleted document '{doc_id}'")
        return None  # 204 No Content

    except ValueError as e:
        logger.warning(f"Document not found for deletion: {doc_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except Exception as e:
        logger.error(f"Error deleting document '{doc_id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description="Submit a natural language question and receive an answer based on document content.",
)
async def query_knowledge_base(
    request: QueryRequest,
) -> QueryResponse:
    """Query the knowledge base.

    This endpoint implements the RAG (Retrieval-Augmented Generation) pipeline:
    1. Embeds the user's question
    2. Retrieves relevant document chunks from vector store
    3. Constructs prompt with question and context
    4. Generates answer using LLM
    5. Returns answer with source citations

    Args:
        request: Query request with question and parameters

    Returns:
        QueryResponse with answer and source citations

    Raises:
        HTTPException: If query processing fails
    """
    if not query_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query service not initialized",
        )

    logger.info(f"Processing query: '{request.question[:100]}...'")

    try:
        response = await query_service.query(request)
        logger.info(
            f"Query completed in {response.processing_time:.2f}s "
            f"with {len(response.sources)} sources"
        )
        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)

        # Check if it's a timeout error
        if "timeout" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API request timed out. Please try again.",
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}",
        )


# Serve React Frontend (SPA)
# This must be placed after API routes to avoid shadowing them
static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(static_dir):
    # Mount static assets directory explicitly if it exists (React usually puts css/js here)
    static_assets_dir = os.path.join(static_dir, "static")
    if os.path.exists(static_assets_dir):
        app.mount("/static", StaticFiles(directory=static_assets_dir), name="static_assets")

    # Catch-all route for SPA
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow API routes to bubble up (though they should match first anyway)
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")

        # Check if specific file exists in static root (e.g. favicon.ico, manifest.json)
        file_path = os.path.join(static_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

        # Fallback to index.html for React Router paths
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)

        # If no index.html (e.g. locally without build), return 404
        raise HTTPException(status_code=404, detail="Frontend not found (build required)")
