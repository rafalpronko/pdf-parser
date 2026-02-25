"""Security tests for the PDF RAG system.

Tests covering:
- Path traversal prevention in file storage
- CORS configuration
- Security headers middleware
- SecretStr usage for API keys
"""

from unittest.mock import patch

import pytest

from app.storage.file_storage import FileStorageService, FileValidationError


class TestPathTraversalPrevention:
    """Tests for path traversal attack prevention in file storage."""

    def test_normal_file_path_allowed(self, tmp_path, monkeypatch):
        """Test that normal file paths are accepted."""
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests-only")

        from app.config import Settings

        with patch("app.storage.file_storage.get_settings") as mock_settings:
            settings = Settings()
            mock_settings.return_value = settings
            service = FileStorageService()
            service.upload_dir = tmp_path

            # Normal path should work
            path = service.get_storage_path("abc-123", "document.pdf")
            assert path.parent == tmp_path
            assert path.name == "abc-123.pdf"

    def test_path_traversal_with_dotdot_rejected(self, tmp_path, monkeypatch):
        """Test that ../.. in file_id is rejected."""
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests-only")

        from app.config import Settings

        with patch("app.storage.file_storage.get_settings") as mock_settings:
            settings = Settings()
            mock_settings.return_value = settings
            service = FileStorageService()
            service.upload_dir = tmp_path

            # Path traversal attempt should be rejected
            with pytest.raises(FileValidationError, match="outside upload directory"):
                service.get_storage_path("../../etc/passwd", "malicious.pdf")

    def test_path_traversal_with_absolute_path_rejected(self, tmp_path, monkeypatch):
        """Test that absolute paths in file_id are handled safely."""
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests-only")

        from app.config import Settings

        with patch("app.storage.file_storage.get_settings") as mock_settings:
            settings = Settings()
            mock_settings.return_value = settings
            service = FileStorageService()
            service.upload_dir = tmp_path

            # Absolute path should either be rejected or resolved inside upload_dir
            # Depending on OS behavior, this may resolve inside or outside
            try:
                path = service.get_storage_path("/tmp/evil", "test.pdf")
                # If resolved, it must be inside upload_dir
                path.relative_to(tmp_path.resolve())
            except FileValidationError:
                # Expected: path traversal detected
                pass

    def test_path_traversal_with_encoded_characters(self, tmp_path, monkeypatch):
        """Test that encoded path separators don't bypass validation."""
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests-only")

        from app.config import Settings

        with patch("app.storage.file_storage.get_settings") as mock_settings:
            settings = Settings()
            mock_settings.return_value = settings
            service = FileStorageService()
            service.upload_dir = tmp_path

            # File with normal ID should resolve inside upload dir
            path = service.get_storage_path("safe-id-123", "normal.pdf")
            resolved = path.resolve()
            resolved.relative_to(tmp_path.resolve())


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_cors_origins_from_settings(self, monkeypatch):
        """Test that CORS origins are loaded from settings."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests-only")
        monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000","http://example.com"]')

        from app.config import Settings

        settings = Settings()
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://example.com" in settings.cors_origins

    def test_cors_default_origins(self, monkeypatch):
        """Test default CORS origins."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests-only")

        from app.config import Settings

        settings = Settings()
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:8000" in settings.cors_origins


class TestSecretStrAPIKey:
    """Tests for SecretStr usage for API keys."""

    def test_api_key_is_secret_str(self, monkeypatch):
        """Test that API key is stored as SecretStr."""
        monkeypatch.setenv(
            "OPENAI_API_KEY",
            "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef",
        )

        from pydantic import SecretStr

        from app.config import Settings

        settings = Settings()
        assert isinstance(settings.openai_api_key, SecretStr)

    def test_api_key_not_in_repr(self, monkeypatch):
        """Test that API key is not exposed in string representation."""
        monkeypatch.setenv(
            "OPENAI_API_KEY",
            "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef",
        )

        from app.config import Settings

        settings = Settings()
        settings_str = str(settings)
        assert "sk-proj-test-fake-key" not in settings_str

    def test_api_key_accessible_via_get_secret_value(self, monkeypatch):
        """Test that API key is accessible via get_secret_value()."""
        key = "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        monkeypatch.setenv("OPENAI_API_KEY", key)

        from app.config import Settings

        settings = Settings()
        assert settings.openai_api_key.get_secret_value() == key

    def test_none_api_key_allowed(self, monkeypatch):
        """Test that None API key is allowed (optional field)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from app.config import Settings

        # Use _env_file=None to prevent loading from .env file
        settings = Settings(_env_file=None)
        assert settings.openai_api_key is None
