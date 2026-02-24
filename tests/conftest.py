"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_pdf_path():
    """Path to sample PDF for testing."""
    return "data/raw/1512.03385v1.pdf"
