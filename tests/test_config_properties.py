"""Property-based tests for configuration validation.

Feature: pdf-rag-system
Properties tested:
- Property 23: Configuration validation at startup
- Property 24: Environment variable loading
"""

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.config import Settings


# Feature: pdf-rag-system, Property 23: Configuration validation at startup
@given(
    chunk_size=st.integers(min_value=-100, max_value=3000),
    chunk_overlap=st.integers(min_value=-100, max_value=1000),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_23_invalid_config_fails_at_startup(
    chunk_size: int, chunk_overlap: int
):
    """Property 23: Configuration validation at startup.
    
    For any application startup, if required configuration values are missing
    or invalid (e.g., invalid API key format, negative chunk size), the system
    should fail to start and log specific configuration errors.
    
    Validates: Requirements 6.4
    """
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    
    try:
        # Invalid configurations should raise ValidationError
        if chunk_size < 100 or chunk_size > 2000:
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "chunk_size" in str(exc_info.value).lower()
        elif chunk_overlap < 0 or chunk_overlap > 500:
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "chunk_overlap" in str(exc_info.value).lower()
        elif chunk_overlap >= chunk_size:
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "overlap" in str(exc_info.value).lower()
        else:
            # Valid configuration should succeed
            config = Settings()
            assert config.chunk_size == chunk_size
            assert config.chunk_overlap == chunk_overlap
    finally:
        # Cleanup
        os.environ.pop("CHUNK_SIZE", None)
        os.environ.pop("CHUNK_OVERLAP", None)


# Feature: pdf-rag-system, Property 24: Environment variable loading
@given(
    api_title=st.text(
        min_size=1, 
        max_size=100, 
        alphabet=st.characters(blacklist_categories=("Cs", "Cc"))  # Exclude control characters including null
    ),
    api_version=st.text(
        min_size=1, 
        max_size=20, 
        alphabet=st.characters(blacklist_categories=("Cs", "Cc"))
    ),
    chunk_size=st.integers(min_value=100, max_value=2000),
    enable_vlm=st.booleans(),
    enable_multimodal=st.booleans(),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_24_environment_variable_loading(
    api_title: str,
    api_version: str,
    chunk_size: int,
    enable_vlm: bool,
    enable_multimodal: bool,
):
    """Property 24: Environment variable loading.
    
    For any configuration setting defined in environment variables, the system
    should correctly load and validate the value, with proper type conversion
    (strings to integers, booleans, etc.).
    
    Validates: Requirements 7.5
    """
    # Set required environment variables
    os.environ["OPENAI_API_KEY"] = "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
    os.environ["API_TITLE"] = api_title
    os.environ["API_VERSION"] = api_version
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["ENABLE_VLM"] = str(enable_vlm)
    os.environ["ENABLE_MULTIMODAL_CHUNKING"] = str(enable_multimodal)
    os.environ["UPLOAD_DIR"] = "/tmp/test_uploads"
    
    try:
        # Load configuration
        config = Settings()
        
        # Verify type conversions
        assert config.api_title == api_title
        assert config.api_version == api_version
        assert config.chunk_size == chunk_size
        assert isinstance(config.chunk_size, int)
        assert config.enable_vlm == enable_vlm
        assert isinstance(config.enable_vlm, bool)
        assert config.enable_multimodal_chunking == enable_multimodal
        assert isinstance(config.enable_multimodal_chunking, bool)
        assert isinstance(config.upload_dir, Path)
    finally:
        # Cleanup
        for key in ["API_TITLE", "API_VERSION", "CHUNK_SIZE", "ENABLE_VLM", "ENABLE_MULTIMODAL_CHUNKING", "UPLOAD_DIR"]:
            os.environ.pop(key, None)


def test_property_23_api_key_optional():
    """Property 23: API key is now optional (can be None)."""
    # API key can be None or a valid key
    # This test just verifies the config loads successfully
    config = Settings()
    # API key should be either None or a valid string
    assert config.openai_api_key is None or isinstance(config.openai_api_key, str)


def test_property_23_invalid_api_key_format_fails(monkeypatch):
    """Property 23: Invalid API key format should fail at startup."""
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-key-format")
    
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    assert "sk-" in str(exc_info.value)


def test_property_24_log_level_conversion(monkeypatch):
    """Property 24: Log level should be converted to uppercase."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    
    config = Settings()
    assert config.log_level == "DEBUG"
    assert isinstance(config.log_level, str)


def test_property_23_invalid_log_level_fails(monkeypatch):
    """Property 23: Invalid log level should fail at startup."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef")
    monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")
    
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    assert "log_level" in str(exc_info.value).lower()


def test_property_24_path_conversion(monkeypatch, tmp_path):
    """Property 24: String paths should be converted to Path objects."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef")
    upload_path = str(tmp_path / "test_uploads")
    monkeypatch.setenv("UPLOAD_DIR", upload_path)
    
    config = Settings()
    assert isinstance(config.upload_dir, Path)
    assert str(config.upload_dir) == upload_path
    # Directory should be created
    assert config.upload_dir.exists()


def test_property_23_chunk_overlap_greater_than_size_fails(monkeypatch):
    """Property 23: Chunk overlap >= chunk size should fail."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    error_str = str(exc_info.value).lower()
    assert "overlap" in error_str or "chunk" in error_str
