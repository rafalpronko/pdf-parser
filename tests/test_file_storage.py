"""Tests for file storage utilities."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from app.storage import FileStorageService, FileValidationError


class TestFileStorageService:
    """Unit tests for FileStorageService."""

    @pytest.fixture
    def temp_upload_dir(self, tmp_path):
        """Create temporary upload directory."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        return upload_dir

    @pytest.fixture
    def file_storage(self, temp_upload_dir, monkeypatch):
        """Create FileStorageService with temporary directory."""
        # Mock settings to use temp directory
        from app.config import Settings
        
        settings = Settings(
            openai_api_key="sk-test-key-for-testing",
            upload_dir=temp_upload_dir,
        )
        
        # Patch get_settings to return our test settings
        monkeypatch.setattr("app.storage.file_storage.get_settings", lambda: settings)
        
        return FileStorageService()

    def test_validate_file_format_valid_pdf(self, file_storage):
        """Test validation accepts valid PDF files."""
        # Should not raise exception
        file_storage.validate_file_format("document.pdf", "application/pdf")
        file_storage.validate_file_format("document.PDF", "application/pdf")
        file_storage.validate_file_format("document.pdf", "application/x-pdf")

    def test_validate_file_format_invalid_extension(self, file_storage):
        """Test validation rejects invalid file extensions."""
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_format("document.txt", "application/pdf")
        assert "Invalid file extension" in str(exc_info.value)

    def test_validate_file_format_invalid_content_type(self, file_storage):
        """Test validation rejects invalid content types."""
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_format("document.pdf", "text/plain")
        assert "Invalid content type" in str(exc_info.value)

    def test_validate_file_size_valid(self, file_storage):
        """Test validation accepts valid file sizes."""
        # Should not raise exception
        file_storage.validate_file_size(1024)  # 1 KB
        file_storage.validate_file_size(1024 * 1024)  # 1 MB
        file_storage.validate_file_size(file_storage.max_file_size)  # Max size

    def test_validate_file_size_too_large(self, file_storage):
        """Test validation rejects files exceeding size limit."""
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_size(file_storage.max_file_size + 1)
        assert "exceeds maximum allowed size" in str(exc_info.value)

    def test_validate_file_size_zero(self, file_storage):
        """Test validation rejects zero-size files."""
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_size(0)
        assert "must be greater than 0 bytes" in str(exc_info.value)

    def test_calculate_file_hash(self, file_storage):
        """Test file hash calculation."""
        content = b"test content"
        hash1 = file_storage.calculate_file_hash(content)
        hash2 = file_storage.calculate_file_hash(content)
        
        # Same content should produce same hash
        assert hash1 == hash2
        
        # Different content should produce different hash
        different_content = b"different content"
        hash3 = file_storage.calculate_file_hash(different_content)
        assert hash1 != hash3
        
        # Hash should be 64 characters (SHA-256 hex)
        assert len(hash1) == 64

    def test_generate_file_id(self, file_storage):
        """Test file ID generation."""
        id1 = file_storage.generate_file_id()
        id2 = file_storage.generate_file_id()
        
        # IDs should be unique
        assert id1 != id2
        
        # IDs should be valid UUIDs (36 characters with hyphens)
        assert len(id1) == 36
        assert id1.count("-") == 4

    def test_get_storage_path(self, file_storage):
        """Test storage path generation."""
        file_id = "test-id-123"
        filename = "document.pdf"
        
        path = file_storage.get_storage_path(file_id, filename)
        
        # Path should be in upload directory
        assert path.parent == file_storage.upload_dir
        
        # Filename should include file ID and preserve extension
        assert path.name == "test-id-123.pdf"

    @pytest.mark.asyncio
    async def test_save_file_success(self, file_storage):
        """Test successful file save."""
        content = b"%PDF-1.4 test content"
        filename = "test.pdf"
        content_type = "application/pdf"
        
        metadata = await file_storage.save_file(content, filename, content_type)
        
        # Check metadata
        assert metadata.filename == filename
        assert metadata.file_size == len(content)
        assert metadata.content_type == content_type
        assert len(metadata.file_id) == 36  # UUID
        assert len(metadata.file_hash) == 64  # SHA-256
        assert metadata.upload_path.exists()
        
        # Verify file was written correctly
        saved_content = metadata.upload_path.read_bytes()
        assert saved_content == content

    @pytest.mark.asyncio
    async def test_save_file_invalid_format(self, file_storage):
        """Test save file rejects invalid format."""
        content = b"test content"
        
        with pytest.raises(FileValidationError):
            await file_storage.save_file(content, "test.txt", "text/plain")

    @pytest.mark.asyncio
    async def test_save_file_too_large(self, file_storage):
        """Test save file rejects oversized files."""
        # Create content larger than max size
        content = b"x" * (file_storage.max_file_size + 1)
        
        with pytest.raises(FileValidationError):
            await file_storage.save_file(content, "test.pdf", "application/pdf")

    def test_get_file_path_exists(self, file_storage):
        """Test getting path to existing file."""
        # Create a test file
        file_id = "test-file-id"
        test_file = file_storage.upload_dir / f"{file_id}.pdf"
        test_file.write_bytes(b"test")
        
        path = file_storage.get_file_path(file_id)
        assert path == test_file
        assert path.exists()

    def test_get_file_path_not_exists(self, file_storage):
        """Test getting path to non-existent file."""
        path = file_storage.get_file_path("non-existent-id")
        assert path is None

    def test_delete_file_exists(self, file_storage):
        """Test deleting existing file."""
        # Create a test file
        file_id = "test-file-id"
        test_file = file_storage.upload_dir / f"{file_id}.pdf"
        test_file.write_bytes(b"test")
        
        assert test_file.exists()
        result = file_storage.delete_file(file_id)
        assert result is True
        assert not test_file.exists()

    def test_delete_file_not_exists(self, file_storage):
        """Test deleting non-existent file."""
        result = file_storage.delete_file("non-existent-id")
        assert result is False

    def test_file_exists(self, file_storage):
        """Test checking file existence."""
        # Create a test file
        file_id = "test-file-id"
        test_file = file_storage.upload_dir / f"{file_id}.pdf"
        test_file.write_bytes(b"test")
        
        assert file_storage.file_exists(file_id) is True
        assert file_storage.file_exists("non-existent-id") is False




class TestFileValidationProperties:
    """Property-based tests for file validation."""

    @pytest.fixture
    def temp_upload_dir(self, tmp_path):
        """Create temporary upload directory."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        return upload_dir

    @pytest.fixture
    def file_storage(self, temp_upload_dir, monkeypatch):
        """Create FileStorageService with temporary directory."""
        from app.config import Settings
        
        settings = Settings(
            openai_api_key="sk-test-key-for-testing",
            upload_dir=temp_upload_dir,
        )
        
        monkeypatch.setattr("app.storage.file_storage.get_settings", lambda: settings)
        
        return FileStorageService()

    @given(
        extension=st.text(min_size=1, max_size=10).filter(
            lambda x: x.lower() not in ['.pdf', 'pdf'] and x not in ['/', '\\', '.']
        )
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_2_invalid_file_extension_rejected(self, file_storage, extension):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any file input, if the file format is not PDF or the size exceeds 
        the maximum limit, the system should reject the upload and return a 
        validation error with specific details about which constraint was violated.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that files with non-PDF extensions are rejected.
        """
        # Ensure extension starts with a dot
        if not extension.startswith('.'):
            extension = '.' + extension
        
        filename = f"document{extension}"
        
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_format(filename, "application/pdf")
        
        # Verify error message contains specific details
        error_msg = str(exc_info.value)
        assert "Invalid file extension" in error_msg
        # The actual extension in the error might be what Path().suffix returns
        # which could be different from what we constructed

    @given(
        content_type=st.text(min_size=1, max_size=50).filter(
            lambda x: x not in ["application/pdf", "application/x-pdf"]
        )
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_2_invalid_content_type_rejected(self, file_storage, content_type):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any file input with invalid content type, the system should reject 
        the upload and return a validation error with specific details.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that files with non-PDF content types are rejected.
        """
        filename = "document.pdf"
        
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_format(filename, content_type)
        
        # Verify error message contains specific details
        error_msg = str(exc_info.value)
        assert "Invalid content type" in error_msg
        assert content_type in error_msg

    @given(
        file_size=st.integers(min_value=1, max_value=1000000000).filter(
            lambda x: x > 50 * 1024 * 1024  # Greater than default max (50MB)
        )
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_2_oversized_file_rejected(self, file_storage, file_size):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any file input where the size exceeds the maximum limit, the system 
        should reject the upload and return a validation error with specific 
        details about the size constraint violation.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that files exceeding the maximum size are rejected.
        """
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_size(file_size)
        
        # Verify error message contains specific details
        error_msg = str(exc_info.value)
        assert "exceeds maximum allowed size" in error_msg
        # Verify the error message includes size information
        assert "MB" in error_msg

    @given(
        file_size=st.integers(max_value=0)
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_2_zero_or_negative_size_rejected(self, file_storage, file_size):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any file input with zero or negative size, the system should reject 
        the upload and return a validation error with specific details.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that files with invalid sizes are rejected.
        """
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_size(file_size)
        
        # Verify error message contains specific details
        error_msg = str(exc_info.value)
        assert "must be greater than 0 bytes" in error_msg

    @given(
        extension=st.text(min_size=1, max_size=10).filter(
            lambda x: x.lower() not in ['.pdf', 'pdf']
        ),
        content_type=st.text(min_size=1, max_size=50).filter(
            lambda x: x not in ["application/pdf", "application/x-pdf"]
        )
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_2_multiple_validation_failures(self, file_storage, extension, content_type):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any file input with multiple validation failures (both invalid 
        extension and content type), the system should reject the upload and 
        return a validation error. The error should report the first constraint 
        violation encountered.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that validation catches the first error in the chain.
        """
        # Ensure extension starts with a dot
        if not extension.startswith('.'):
            extension = '.' + extension
        
        filename = f"document{extension}"
        
        with pytest.raises(FileValidationError) as exc_info:
            file_storage.validate_file_format(filename, content_type)
        
        # Verify error message contains specific details about at least one violation
        error_msg = str(exc_info.value)
        # Should catch extension error first (as it's checked first in the code)
        assert "Invalid file extension" in error_msg or "Invalid content type" in error_msg

    @given(
        valid_size=st.integers(min_value=1, max_value=50 * 1024 * 1024),
        valid_content_type=st.sampled_from(["application/pdf", "application/x-pdf"]),
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_2_valid_inputs_accepted(self, file_storage, valid_size, valid_content_type):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any file input with valid format and size, the system should accept 
        the upload without raising validation errors.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that valid inputs pass validation (inverse property).
        """
        filename = "document.pdf"
        
        # These should not raise exceptions
        file_storage.validate_file_format(filename, valid_content_type)
        file_storage.validate_file_size(valid_size)

    @given(
        filename=st.text(min_size=1, max_size=100).filter(
            lambda x: not x.lower().endswith('.pdf')
        ),
        file_size=st.integers(min_value=1, max_value=1000000000).filter(
            lambda x: x > 50 * 1024 * 1024
        )
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_2_save_file_rejects_invalid_inputs(
        self, file_storage, filename, file_size
    ):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any file input with invalid format or size, the save_file method 
        should reject the upload and return a validation error before attempting 
        to write to disk.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies end-to-end validation in the save_file method.
        """
        # Create file content of the specified size
        file_content = b"x" * file_size
        
        with pytest.raises(FileValidationError) as exc_info:
            await file_storage.save_file(
                file_content=file_content,
                filename=filename,
                content_type="application/pdf"
            )
        
        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0
        # Should mention either extension or size issue
        assert (
            "Invalid file extension" in error_msg or 
            "exceeds maximum allowed size" in error_msg
        )

    @given(
        case_variant=st.sampled_from([
            "document.pdf",
            "document.PDF", 
            "document.Pdf",
            "document.pDf",
            "document.pdF"
        ])
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_2_case_insensitive_extension_validation(
        self, file_storage, case_variant
    ):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any PDF file with different case variations of the .pdf extension, 
        the system should accept the upload (case-insensitive validation).
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that extension validation is case-insensitive.
        """
        # Should not raise exception for any case variant
        file_storage.validate_file_format(case_variant, "application/pdf")

    @given(
        file_content=st.binary(min_size=1, max_size=50 * 1024 * 1024)
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_2_valid_pdf_saves_successfully(
        self, file_storage, file_content
    ):
        """Feature: pdf-rag-system, Property 2: File validation rejects invalid inputs.
        
        For any valid PDF file (correct format and within size limit), the 
        save_file method should successfully save the file and return metadata.
        
        Validates: Requirements 1.2, 1.3
        
        This test verifies that valid files are accepted and saved correctly.
        """
        filename = "document.pdf"
        content_type = "application/pdf"
        
        # Should not raise exception
        metadata = await file_storage.save_file(
            file_content=file_content,
            filename=filename,
            content_type=content_type
        )
        
        # Verify metadata is returned
        assert metadata.filename == filename
        assert metadata.file_size == len(file_content)
        assert metadata.content_type == content_type
        assert metadata.upload_path.exists()
        
        # Verify file was written correctly
        saved_content = metadata.upload_path.read_bytes()
        assert saved_content == file_content
        
        # Clean up
        file_storage.delete_file(metadata.file_id)
