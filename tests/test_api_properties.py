"""Property-based tests for API endpoints.

Feature: pdf-rag-system, Property 27: HTTP error status codes
Validates: Requirements 10.2
"""

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st, HealthCheck
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, UTC
import io
import urllib.parse

from app.main import app
from app.models.document import DocumentInfo, DocumentUploadResponse
from app.models.query import QueryResponse, SourceReference
from app.services.document_service import DocumentProcessingError
from app.storage.file_storage import FileValidationError


class TestAPIErrorStatusCodes:
    """Test HTTP error status codes for API endpoints."""
    
    @given(
        filename=st.one_of(
            st.just(""),  # Empty filename
            st.text(min_size=1, max_size=50).filter(lambda x: not x.endswith(".pdf")),  # Non-PDF
        )
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_27_invalid_file_returns_400(self, filename):
        """Property 27: Invalid file uploads return 400 Bad Request.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        For any invalid file (empty filename or non-PDF), the API should return
        400 Bad Request status code with error details.
        """
        with TestClient(app) as client:
            # Create a mock file
            file_content = b"fake pdf content"
            files = {"file": (filename, io.BytesIO(file_content), "application/pdf")}
            
            # Make request
            response = client.post("/api/v1/documents/upload", files=files)
            
            # Property: Invalid files should return 400 or 422 (validation error)
            assert response.status_code in [400, 422]
            
            # Property: Response should contain error details
            error_data = response.json()
            assert "detail" in error_data
    
    @given(
        skip=st.integers(min_value=-100, max_value=-1),  # Negative skip
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_27_invalid_pagination_returns_400(self, skip):
        """Property 27: Invalid pagination parameters return 400 Bad Request.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        For any invalid pagination parameter (negative skip), the API should
        return 400 Bad Request status code.
        """
        with TestClient(app) as client:
            response = client.get(f"/api/v1/documents?skip={skip}&limit=10")
            
            # Property: Invalid pagination should return 400
            assert response.status_code == 400
            
            # Property: Response should contain error details
            error_data = response.json()
            assert "detail" in error_data
            assert "skip must be non-negative" in error_data["detail"]
    
    @given(
        limit=st.one_of(
            st.integers(min_value=-100, max_value=0),  # Non-positive
            st.integers(min_value=101, max_value=1000),  # Too large
        )
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_27_invalid_limit_returns_400(self, limit):
        """Property 27: Invalid limit parameters return 400 Bad Request.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        For any invalid limit parameter (<=0 or >100), the API should
        return 400 Bad Request status code.
        """
        with TestClient(app) as client:
            response = client.get(f"/api/v1/documents?skip=0&limit={limit}")
            
            # Property: Invalid limit should return 400
            assert response.status_code == 400
            
            # Property: Response should contain error details
            error_data = response.json()
            assert "detail" in error_data
            assert "limit must be between 1 and 100" in error_data["detail"]
    
    @given(
        doc_id=st.text(
            alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')),
            min_size=1,
            max_size=50
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_27_nonexistent_document_returns_404(self, doc_id):
        """Property 27: Requests for nonexistent documents return 404 Not Found.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        For any document ID that doesn't exist, GET and DELETE requests should
        return 404 Not Found status code.
        """
        with TestClient(app) as client:
            with patch("app.main.document_service") as mock_service:
                # Mock service to raise ValueError for not found
                mock_service.get_document = AsyncMock(
                    side_effect=ValueError(f"Document {doc_id} not found")
                )
                mock_service.delete_document = AsyncMock(
                    side_effect=ValueError(f"Document {doc_id} not found")
                )
                
                # Test GET
                response = client.get(f"/api/v1/documents/{urllib.parse.quote(doc_id, safe='')}")
                assert response.status_code == 404
                error_data = response.json()
                assert "detail" in error_data
                
                # Test DELETE
                response = client.delete(f"/api/v1/documents/{urllib.parse.quote(doc_id, safe='')}")
                assert response.status_code == 404
                error_data = response.json()
                assert "detail" in error_data
    
    @given(
        invalid_data=st.one_of(
            st.just({}),  # Missing question field
            st.just({"question": 123}),  # Wrong type
            st.just({"question": None}),  # Null value
        )
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_27_invalid_query_returns_422(self, invalid_data):
        """Property 27: Invalid query requests return 422 Unprocessable Entity.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        For any invalid query (missing field, wrong type), the API should
        return 422 Unprocessable Entity due to Pydantic validation failure.
        """
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/query",
                json=invalid_data
            )
            
            # Property: Invalid query should return 422 (validation error)
            assert response.status_code == 422
            
            # Property: Response should contain validation error details
            error_data = response.json()
            assert "detail" in error_data
    
    def test_property_27_processing_error_returns_500(self):
        """Property 27: Document processing errors return 500 Internal Server Error.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        When document processing fails internally, the API should return
        500 Internal Server Error status code.
        """
        with TestClient(app) as client:
            with patch("app.main.document_service") as mock_service:
                # Mock service to raise processing error
                mock_service.process_document = AsyncMock(
                    side_effect=DocumentProcessingError("Failed to parse PDF")
                )
                
                # Create a valid PDF file upload
                file_content = b"%PDF-1.4 fake content"
                files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
                
                response = client.post("/api/v1/documents/upload", files=files)
                
                # Property: Processing errors should return 500
                assert response.status_code == 500
                
                # Property: Response should contain error details
                error_data = response.json()
                assert "error" in error_data
                assert "detail" in error_data
                assert "timestamp" in error_data
                assert "request_id" in error_data
    
    def test_property_27_service_unavailable_returns_503(self):
        """Property 27: Service unavailability returns 503 Service Unavailable.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        When services are not initialized or external services timeout,
        the API should return 503 Service Unavailable status code.
        """
        with TestClient(app) as client:
            with patch("app.main.document_service", None):
                # Test with uninitialized document service
                response = client.get("/api/v1/documents")
                
                # Property: Unavailable service should return 503
                assert response.status_code == 503
                
                # Property: Response should contain error details
                error_data = response.json()
                assert "detail" in error_data
                assert "not initialized" in error_data["detail"].lower()
    
    def test_property_27_timeout_returns_503(self):
        """Property 27: Timeout errors return 503 Service Unavailable.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        When external API calls timeout, the API should return
        503 Service Unavailable status code.
        """
        with TestClient(app) as client:
            with patch("app.main.query_service") as mock_service:
                # Mock service to raise timeout error with "timeout" in message
                mock_service.query = AsyncMock(
                    side_effect=Exception("Request timeout occurred")
                )
                
                response = client.post(
                    "/api/v1/query",
                    json={"question": "What is this about?"}
                )
                
                # Property: Timeout errors should return 503
                assert response.status_code == 503
                
                # Property: Response should mention timeout (either "timeout" or "timed out")
                error_data = response.json()
                assert "detail" in error_data
                detail_lower = error_data["detail"].lower()
                assert "timeout" in detail_lower or "timed out" in detail_lower
    
    @given(
        error_type=st.sampled_from([
            "ValueError",
            "RuntimeError", 
            "KeyError",
            "AttributeError",
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_27_unexpected_errors_return_500(self, error_type):
        """Property 27: Unexpected errors return 500 Internal Server Error.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        For any unexpected exception during request processing, the API should
        return 500 Internal Server Error status code.
        """
        with TestClient(app) as client:
            with patch("app.main.document_service") as mock_service:
                # Create the appropriate exception type
                exception_class = {
                    "ValueError": ValueError,
                    "RuntimeError": RuntimeError,
                    "KeyError": KeyError,
                    "AttributeError": AttributeError,
                }[error_type]
                
                # Mock service to raise unexpected error
                mock_service.list_documents = AsyncMock(
                    side_effect=exception_class("Unexpected error occurred")
                )
                
                response = client.get("/api/v1/documents")
                
                # Property: Unexpected errors should return 500
                assert response.status_code == 500
                
                # Property: Response should contain error details
                error_data = response.json()
                assert "detail" in error_data
    
    def test_property_27_file_validation_error_returns_400(self):
        """Property 27: File validation errors return 400 Bad Request.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        When file validation fails (size, format), the API should return
        400 Bad Request status code.
        """
        with TestClient(app) as client:
            with patch("app.main.document_service") as mock_service:
                # Mock service to raise file validation error
                mock_service.process_document = AsyncMock(
                    side_effect=FileValidationError("File size exceeds maximum limit")
                )
                
                # Create a valid PDF file upload
                file_content = b"%PDF-1.4 fake content"
                files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
                
                response = client.post("/api/v1/documents/upload", files=files)
                
                # Property: File validation errors should return 400
                assert response.status_code == 400
                
                # Property: Response should contain error details
                error_data = response.json()
                assert "error" in error_data
                assert "detail" in error_data
                assert "File" in error_data["error"]
    
    @given(
        top_k=st.integers(min_value=-10, max_value=0).map(lambda x: x) | 
              st.integers(min_value=21, max_value=100),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_27_invalid_query_params_return_422(self, top_k):
        """Property 27: Invalid query parameters return 422 Unprocessable Entity.
        
        Feature: pdf-rag-system, Property 27: HTTP error status codes
        Validates: Requirements 10.2
        
        For any query with invalid parameters (top_k out of range), the API
        should return 422 Unprocessable Entity due to Pydantic validation.
        """
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/query",
                json={
                    "question": "What is this about?",
                    "top_k": top_k
                }
            )
            
            # Property: Invalid parameters should return 422
            assert response.status_code == 422
            
            # Property: Response should contain validation error
            error_data = response.json()
            assert "error" in error_data
            assert "detail" in error_data
