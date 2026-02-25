"""Tests for configuration module."""

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.config import Settings, get_settings, reload_settings


class TestSettingsValidation:
    """Test configuration validation."""

    def test_settings_with_valid_config(self, monkeypatch):
        """Test that settings load correctly with valid configuration."""
        # Set required environment variables
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", "512")
        monkeypatch.setenv("CHUNK_OVERLAP", "50")

        settings = Settings()

        assert (
            settings.openai_api_key.get_secret_value()
            == "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.api_title == "PDF RAG System"

    def test_settings_missing_api_key(self, monkeypatch):
        """Test that missing API key raises validation error."""
        # Clear the API key if it exists
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Disable .env file loading for this test
        monkeypatch.setenv("OPENAI_API_KEY", "")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "openai_api_key" in str(exc_info.value)

    def test_settings_invalid_api_key_format(self, monkeypatch):
        """Test that invalid API key format raises validation error."""
        monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "should start with 'sk-'" in str(exc_info.value)

    def test_settings_placeholder_api_key(self, monkeypatch):
        """Test that placeholder API key raises validation error."""
        monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key_here")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        # Check that validation error mentions the API key format requirement
        assert "should start with 'sk-'" in str(exc_info.value).lower()

    def test_settings_invalid_chunk_size(self, monkeypatch):
        """Test that invalid chunk size raises validation error."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", "50")  # Too small

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "chunk_size" in str(exc_info.value)

    def test_settings_chunk_overlap_greater_than_size(self, monkeypatch):
        """Test that chunk overlap >= chunk size raises validation error."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", "200")  # Small chunk size
        monkeypatch.setenv("CHUNK_OVERLAP", "200")  # Equal to chunk size

        with pytest.raises(ValueError) as exc_info:
            Settings()

        assert "chunk_overlap" in str(exc_info.value)
        assert "must be less than" in str(exc_info.value)

    def test_settings_invalid_log_level(self, monkeypatch):
        """Test that invalid log level raises validation error."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("LOG_LEVEL", "INVALID")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "log_level" in str(exc_info.value)

    def test_settings_log_level_case_insensitive(self, monkeypatch):
        """Test that log level is case insensitive."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("LOG_LEVEL", "debug")

        settings = Settings()

        assert settings.log_level == "DEBUG"

    def test_settings_creates_directories(self, monkeypatch, tmp_path):
        """Test that settings creates required directories."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        upload_dir = tmp_path / "uploads"
        vector_db_dir = tmp_path / "vectordb"
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
        monkeypatch.setenv("VECTOR_DB_PATH", str(vector_db_dir))

        Settings()  # noqa: F841 -- side effect creates directories

        assert upload_dir.exists()
        assert vector_db_dir.exists()

    def test_settings_default_values(self, monkeypatch):
        """Test that default values are set correctly."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )

        settings = Settings()

        assert settings.api_title == "PDF RAG System"
        assert settings.api_version == "1.0.0"
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.openai_embedding_model == "text-embedding-3-small"
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50
        assert settings.max_file_size == 50 * 1024 * 1024
        assert settings.text_collection == "text_chunks"  # Default value for text collection
        assert settings.log_level == "INFO"

    def test_get_settings_singleton(self, monkeypatch):
        """Test that get_settings returns the same instance."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )

        # Clear the global settings first
        import app.config

        app.config._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings(self, monkeypatch):
        """Test that reload_settings creates a new instance."""
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", "512")

        settings1 = reload_settings()

        # Change environment variable
        monkeypatch.setenv("CHUNK_SIZE", "1024")

        settings2 = reload_settings()

        assert settings1 is not settings2
        assert settings1.chunk_size == 512
        assert settings2.chunk_size == 1024


class TestConfigurationProperties:
    """Property-based tests for configuration validation."""

    @given(
        chunk_size=st.integers(min_value=-1000, max_value=50),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_23_invalid_chunk_size_rejected(self, chunk_size, monkeypatch):
        """Feature: pdf-rag-system, Property 23: Configuration validation at startup.

        For any application startup, if required configuration values are missing
        or invalid (e.g., invalid API key format, negative chunk size), the system
        should fail to start and log specific configuration errors.

        Validates: Requirements 6.4

        This test verifies that invalid chunk sizes (below minimum of 100) are rejected.
        """
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", str(chunk_size))

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        # Verify that the error message mentions chunk_size
        error_str = str(exc_info.value)
        assert "chunk_size" in error_str.lower()

    @given(
        chunk_size=st.integers(min_value=2100, max_value=10000),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_23_chunk_size_too_large_rejected(self, chunk_size, monkeypatch):
        """Feature: pdf-rag-system, Property 23: Configuration validation at startup.

        For any application startup, if chunk size exceeds maximum (2000),
        the system should fail to start and log specific configuration errors.

        Validates: Requirements 6.4
        """
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", str(chunk_size))

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_str = str(exc_info.value)
        assert "chunk_size" in error_str.lower()

    @given(
        api_key=st.text(
            alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
            min_size=1,
            max_size=100,
        ).filter(lambda x: not x.startswith("sk-") and "\x00" not in x)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_23_invalid_api_key_format_rejected(self, api_key, monkeypatch):
        """Feature: pdf-rag-system, Property 23: Configuration validation at startup.

        For any application startup, if the API key format is invalid
        (doesn't start with 'sk-'), the system should fail to start
        and log specific configuration errors.

        Validates: Requirements 6.4
        """
        monkeypatch.setenv("OPENAI_API_KEY", api_key)

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_str = str(exc_info.value)
        assert "openai_api_key" in error_str.lower() or "sk-" in error_str

    @given(
        chunk_overlap=st.integers(min_value=-500, max_value=-1),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_23_negative_chunk_overlap_rejected(self, chunk_overlap, monkeypatch):
        """Feature: pdf-rag-system, Property 23: Configuration validation at startup.

        For any application startup, if chunk overlap is negative,
        the system should fail to start and log specific configuration errors.

        Validates: Requirements 6.4
        """
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_OVERLAP", str(chunk_overlap))

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_str = str(exc_info.value)
        assert "chunk_overlap" in error_str.lower()

    @given(
        chunk_size=st.integers(min_value=100, max_value=500),
        chunk_overlap=st.integers(min_value=100, max_value=500),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_23_chunk_overlap_gte_chunk_size_rejected(
        self, chunk_size, chunk_overlap, monkeypatch
    ):
        """Feature: pdf-rag-system, Property 23: Configuration validation at startup.

        For any application startup, if chunk overlap is greater than or equal to
        chunk size, the system should fail to start and log specific configuration errors.

        Validates: Requirements 6.4
        """
        # Only test cases where overlap >= size
        if chunk_overlap < chunk_size:
            pytest.skip("Only testing cases where overlap >= size")

        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", str(chunk_size))
        monkeypatch.setenv("CHUNK_OVERLAP", str(chunk_overlap))

        with pytest.raises(ValueError) as exc_info:
            Settings()

        error_str = str(exc_info.value)
        assert "chunk_overlap" in error_str.lower()
        assert "must be less than" in error_str.lower()

    @given(
        log_level=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll"), min_codepoint=65, max_codepoint=122
            ),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_23_invalid_log_level_rejected(self, log_level, monkeypatch):
        """Feature: pdf-rag-system, Property 23: Configuration validation at startup.

        For any application startup, if log level is not one of the valid values,
        the system should fail to start and log specific configuration errors.

        Validates: Requirements 6.4
        """
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("LOG_LEVEL", log_level)

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_str = str(exc_info.value)
        assert "log_level" in error_str.lower()

    @given(
        chunk_size=st.integers(min_value=100, max_value=2000),
        chunk_overlap=st.integers(min_value=0, max_value=99),
        max_file_size=st.integers(min_value=1024, max_value=100 * 1024 * 1024),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_24_environment_variable_loading(
        self, chunk_size, chunk_overlap, max_file_size, monkeypatch, tmp_path
    ):
        """Feature: pdf-rag-system, Property 24: Environment variable loading.

        For any configuration setting defined in environment variables, the system
        should correctly load and validate the value, with proper type conversion
        (strings to integers, booleans, etc.).

        Validates: Requirements 7.5

        This test verifies that integer configuration values are correctly loaded
        from environment variables and converted to the appropriate types.
        """
        # Ensure chunk_overlap < chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size - 1

        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("CHUNK_SIZE", str(chunk_size))
        monkeypatch.setenv("CHUNK_OVERLAP", str(chunk_overlap))
        monkeypatch.setenv("MAX_FILE_SIZE", str(max_file_size))

        # Use tmp_path to avoid creating directories in the project
        upload_dir = tmp_path / "uploads"
        vector_db_dir = tmp_path / "vectordb"
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
        monkeypatch.setenv("VECTOR_DB_PATH", str(vector_db_dir))

        settings = Settings()

        # Verify type conversion from string to int
        assert isinstance(settings.chunk_size, int)
        assert isinstance(settings.chunk_overlap, int)
        assert isinstance(settings.max_file_size, int)

        # Verify values match what was set
        assert settings.chunk_size == chunk_size
        assert settings.chunk_overlap == chunk_overlap
        assert settings.max_file_size == max_file_size

    @given(
        api_title=st.text(
            alphabet=st.characters(
                blacklist_characters="\x00",
                blacklist_categories=("Cs",),  # Exclude surrogates
            ),
            min_size=1,
            max_size=100,
        ),
        api_version=st.text(
            alphabet=st.characters(
                whitelist_categories=("Nd", "Pd"),
                blacklist_characters="\x00",
                blacklist_categories=("Cs",),  # Exclude surrogates
            ),
            min_size=1,
            max_size=20,
        ).filter(lambda x: len(x) > 0),
        text_collection=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                min_codepoint=48,
                max_codepoint=122,
                blacklist_characters="\x00",
                blacklist_categories=("Cs",),  # Exclude surrogates
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_24_string_environment_variable_loading(
        self, api_title, api_version, text_collection, monkeypatch, tmp_path
    ):
        """Feature: pdf-rag-system, Property 24: Environment variable loading.

        For any configuration setting defined in environment variables, the system
        should correctly load and validate the value, with proper type conversion.

        Validates: Requirements 7.5

        This test verifies that string configuration values are correctly loaded
        from environment variables without type conversion issues.
        """
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("API_TITLE", api_title)
        monkeypatch.setenv("API_VERSION", api_version)
        monkeypatch.setenv("TEXT_COLLECTION", text_collection)

        # Use tmp_path to avoid creating directories in the project
        upload_dir = tmp_path / "uploads"
        vector_db_dir = tmp_path / "vectordb"
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
        monkeypatch.setenv("VECTOR_DB_PATH", str(vector_db_dir))

        settings = Settings()

        # Verify type is string
        assert isinstance(settings.api_title, str)
        assert isinstance(settings.api_version, str)
        assert isinstance(settings.text_collection, str)

        # Verify values match what was set
        assert settings.api_title == api_title
        assert settings.api_version == api_version
        assert settings.text_collection == text_collection

    @given(
        log_level=st.sampled_from(
            [
                "debug",
                "info",
                "warning",
                "error",
                "critical",
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
                "DeBuG",
                "InFo",
            ]
        )
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_24_case_insensitive_loading(self, log_level, monkeypatch, tmp_path):
        """Feature: pdf-rag-system, Property 24: Environment variable loading.

        For any configuration setting defined in environment variables, the system
        should correctly load and validate the value. For log level specifically,
        it should handle case-insensitive input and normalize to uppercase.

        Validates: Requirements 7.5
        """
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )
        monkeypatch.setenv("LOG_LEVEL", log_level)

        # Use tmp_path to avoid creating directories in the project
        upload_dir = tmp_path / "uploads"
        vector_db_dir = tmp_path / "vectordb"
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
        monkeypatch.setenv("VECTOR_DB_PATH", str(vector_db_dir))

        settings = Settings()

        # Verify log level is normalized to uppercase
        assert settings.log_level == log_level.upper()
        assert settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    @given(
        upload_dir_str=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd", "Pd", "Pc"),
                min_codepoint=45,
                max_codepoint=122,
            ),
            min_size=1,
            max_size=50,
        ).filter(lambda x: "/" not in x and "\\" not in x and x not in {".", ".."}),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_24_path_type_conversion(self, upload_dir_str, monkeypatch, tmp_path):
        """Feature: pdf-rag-system, Property 24: Environment variable loading.

        For any configuration setting defined in environment variables, the system
        should correctly load and validate the value, with proper type conversion.
        Specifically, string paths should be converted to Path objects.

        Validates: Requirements 7.5
        """
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-proj-test-fake-key-for-unit-tests-only-1234567890abcdef"
        )

        # Use tmp_path to create a valid directory path
        upload_dir = tmp_path / upload_dir_str
        vector_db_dir = tmp_path / "vectordb"
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
        monkeypatch.setenv("VECTOR_DB_PATH", str(vector_db_dir))

        settings = Settings()

        # Verify type conversion from string to Path
        assert isinstance(settings.upload_dir, Path)
        assert str(settings.upload_dir) == str(upload_dir)

        # Verify directory was created
        assert settings.upload_dir.exists()
        assert settings.upload_dir.is_dir()
