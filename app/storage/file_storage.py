"""File storage utilities for handling PDF uploads."""

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, Field

from app.config import get_settings


class FileMetadata(BaseModel):
    """Metadata extracted from uploaded file."""
    
    file_id: str = Field(description="Unique file identifier")
    filename: str = Field(description="Original filename")
    file_size: int = Field(ge=0, description="File size in bytes")
    content_type: str = Field(description="MIME type of the file")
    file_hash: str = Field(description="SHA-256 hash of file content")
    upload_path: Path = Field(description="Path where file is stored")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


class FileStorageService:
    """Service for handling file uploads, validation, and storage."""
    
    ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/x-pdf",
    }
    
    ALLOWED_EXTENSIONS = {".pdf"}
    
    def __init__(self):
        """Initialize file storage service with settings."""
        self.settings = get_settings()
        self.upload_dir = self.settings.upload_dir
        self.max_file_size = self.settings.max_file_size
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_file_format(self, filename: str, content_type: str) -> None:
        """Validate file format based on extension and content type.
        
        Args:
            filename: Original filename
            content_type: MIME type of the file
            
        Raises:
            FileValidationError: If file format is invalid
        """
        # Check file extension
        file_path = Path(filename)
        if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"Invalid file extension '{file_path.suffix}'. "
                f"Allowed extensions: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
        
        # Check content type
        if content_type not in self.ALLOWED_CONTENT_TYPES:
            raise FileValidationError(
                f"Invalid content type '{content_type}'. "
                f"Allowed types: {', '.join(self.ALLOWED_CONTENT_TYPES)}"
            )
    
    def validate_file_size(self, file_size: int) -> None:
        """Validate file size against maximum limit.
        
        Args:
            file_size: Size of the file in bytes
            
        Raises:
            FileValidationError: If file size exceeds limit
        """
        if file_size <= 0:
            raise FileValidationError("File size must be greater than 0 bytes")
        
        if file_size > self.max_file_size:
            max_mb = self.max_file_size / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            raise FileValidationError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed "
                f"size ({max_mb:.2f} MB)"
            )
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content.
        
        Args:
            file_content: Binary content of the file
            
        Returns:
            Hexadecimal string representation of the hash
        """
        return hashlib.sha256(file_content).hexdigest()
    
    def generate_file_id(self) -> str:
        """Generate unique file identifier.
        
        Returns:
            UUID string for the file
        """
        return str(uuid.uuid4())
    
    def get_storage_path(self, file_id: str, original_filename: str) -> Path:
        """Get storage path for a file.
        
        Args:
            file_id: Unique file identifier
            original_filename: Original filename to preserve extension
            
        Returns:
            Path where the file should be stored
        """
        # Preserve the original extension
        extension = Path(original_filename).suffix
        filename = f"{file_id}{extension}"
        return self.upload_dir / filename
    
    async def save_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> FileMetadata:
        """Save uploaded file with validation and metadata extraction.
        
        This method performs the following operations:
        1. Validates file format (extension and content type)
        2. Validates file size
        3. Generates unique file ID
        4. Calculates file hash
        5. Persists file to upload directory
        6. Returns file metadata
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            content_type: MIME type of the file
            
        Returns:
            FileMetadata with all extracted information
            
        Raises:
            FileValidationError: If validation fails
            IOError: If file cannot be written to disk
        """
        # Validate file format
        self.validate_file_format(filename, content_type)
        
        # Validate file size
        file_size = len(file_content)
        self.validate_file_size(file_size)
        
        # Generate unique file ID
        file_id = self.generate_file_id()
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(file_content)
        
        # Get storage path
        storage_path = self.get_storage_path(file_id, filename)
        
        # Write file to disk
        try:
            storage_path.write_bytes(file_content)
        except Exception as e:
            raise IOError(f"Failed to write file to disk: {e}") from e
        
        # Create and return metadata
        metadata = FileMetadata(
            file_id=file_id,
            filename=filename,
            file_size=file_size,
            content_type=content_type,
            file_hash=file_hash,
            upload_path=storage_path,
            created_at=datetime.now(UTC),
        )
        
        return metadata
    
    def get_file_path(self, file_id: str) -> Path | None:
        """Get path to stored file by file ID.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Path to the file if it exists, None otherwise
        """
        # Look for file with this ID (any extension)
        for file_path in self.upload_dir.glob(f"{file_id}.*"):
            if file_path.is_file():
                return file_path
        return None
    
    def delete_file(self, file_id: str) -> bool:
        """Delete stored file by file ID.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            True if file was deleted, False if file not found
        """
        file_path = self.get_file_path(file_id)
        if file_path and file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def file_exists(self, file_id: str) -> bool:
        """Check if file exists in storage.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            True if file exists, False otherwise
        """
        return self.get_file_path(file_id) is not None

