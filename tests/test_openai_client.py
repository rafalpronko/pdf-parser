"""Unit tests for OpenAI client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai import APIError, APITimeoutError, RateLimitError

from app.clients.openai_client import OpenAIClient


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("app.clients.openai_client.AsyncOpenAI") as mock:
        yield mock


@pytest.mark.asyncio
async def test_openai_client_initialization():
    """Test OpenAI client initialization."""
    client = OpenAIClient(
        api_key="test-fake-api-key-for-unit-tests",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        timeout=30.0,
        max_retries=3,
    )
    
    assert client.model == "gpt-4o-mini"
    assert client.embedding_model == "text-embedding-3-small"
    assert client.timeout == 30.0
    assert client.max_retries == 3


@pytest.mark.asyncio
async def test_embed_text_success(mock_openai_client):
    """Test successful text embedding."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    
    mock_instance = AsyncMock()
    mock_instance.embeddings.create = AsyncMock(return_value=mock_response)
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
    result = await client.embed_text("test text")
    
    assert result == [0.1, 0.2, 0.3]
    mock_instance.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_embed_batch_success(mock_openai_client):
    """Test successful batch embedding."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2], index=0),
        MagicMock(embedding=[0.3, 0.4], index=1),
    ]
    
    mock_instance = AsyncMock()
    mock_instance.embeddings.create = AsyncMock(return_value=mock_response)
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
    result = await client.embed_batch(["text1", "text2"])
    
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_instance.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_embed_batch_empty_list(mock_openai_client):
    """Test batch embedding with empty list."""
    mock_instance = AsyncMock()
    mock_openai_client.return_value = mock_instance
    
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
    result = await client.embed_batch([])
    
    assert result == []
    mock_instance.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_generate_success(mock_openai_client):
    """Test successful text generation."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Generated response"))]
    
    mock_instance = AsyncMock()
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
    result = await client.generate("test prompt")
    
    assert result == "Generated response"
    mock_instance.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_with_system_message(mock_openai_client):
    """Test generation with system message."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
    
    mock_instance = AsyncMock()
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
    result = await client.generate(
        "test prompt",
        system_message="You are a helpful assistant"
    )
    
    assert result == "Response"
    
    # Verify system message was included
    call_args = mock_instance.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_generate_with_context(mock_openai_client):
    """Test RAG-style generation with context."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Answer based on context"))]
    
    mock_instance = AsyncMock()
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
    result = await client.generate_with_context(
        question="What is X?",
        context="X is a thing."
    )
    
    assert result == "Answer based on context"
    
    # Verify context was included in prompt
    call_args = mock_instance.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert "Context:" in messages[1]["content"]
    assert "X is a thing." in messages[1]["content"]


@pytest.mark.asyncio
async def test_retry_with_backoff_success_after_retry(mock_openai_client):
    """Test retry logic succeeds after initial failure."""
    # Setup mock to fail once then succeed
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    
    # Create a proper APIError with required arguments
    mock_request = MagicMock()
    api_error = APIError("Temporary error", request=mock_request, body=None)
    
    mock_instance = AsyncMock()
    mock_instance.embeddings.create = AsyncMock(
        side_effect=[
            api_error,
            mock_response,
        ]
    )
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=3)
    result = await client.embed_text("test text")
    
    assert result == [0.1, 0.2, 0.3]
    assert mock_instance.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_retry_with_backoff_all_retries_fail(mock_openai_client):
    """Test retry logic fails after all retries exhausted."""
    # Setup mock to always fail
    mock_request = MagicMock()
    api_error = APIError("Persistent error", request=mock_request, body=None)
    
    mock_instance = AsyncMock()
    mock_instance.embeddings.create = AsyncMock(
        side_effect=api_error
    )
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=3)
    
    with pytest.raises(APIError):
        await client.embed_text("test text")
    
    assert mock_instance.embeddings.create.call_count == 3


@pytest.mark.asyncio
async def test_retry_with_timeout_error(mock_openai_client):
    """Test retry logic with timeout errors."""
    # Setup mock to fail with timeout
    mock_instance = AsyncMock()
    mock_instance.embeddings.create = AsyncMock(
        side_effect=APITimeoutError("Request timeout")
    )
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=2)
    
    with pytest.raises(APITimeoutError):
        await client.embed_text("test text")
    
    assert mock_instance.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_retry_with_rate_limit_error(mock_openai_client):
    """Test retry logic with rate limit errors."""
    # Setup mock to fail with rate limit then succeed
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    
    # Create a proper RateLimitError with required arguments
    mock_http_response = MagicMock()
    mock_http_response.status_code = 429
    rate_limit_error = RateLimitError(
        "Rate limit exceeded",
        response=mock_http_response,
        body=None
    )
    
    mock_instance = AsyncMock()
    mock_instance.embeddings.create = AsyncMock(
        side_effect=[
            rate_limit_error,
            mock_response,
        ]
    )
    mock_openai_client.return_value = mock_instance
    
    # Test
    client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=3)
    result = await client.embed_text("test text")
    
    assert result == [0.1, 0.2, 0.3]
    assert mock_instance.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_context_manager(mock_openai_client):
    """Test async context manager usage."""
    mock_instance = AsyncMock()
    mock_instance.close = AsyncMock()
    mock_openai_client.return_value = mock_instance
    
    async with OpenAIClient(api_key="test-fake-api-key-for-unit-tests") as client:
        assert client is not None
    
    mock_instance.close.assert_called_once()


# ============================================================================
# Property-Based Tests
# ============================================================================

from hypothesis import given, settings, strategies as st, assume
from hypothesis import HealthCheck
import time


class TestEmbeddingDimensionalityConsistency:
    """Property-based tests for Property 13: Embedding dimensionality consistency."""

    @given(
        num_texts=st.integers(min_value=1, max_value=10),
        text_length=st.integers(min_value=10, max_value=500)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_13_embedding_dimensionality_consistency(
        self, num_texts, text_length, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 13: Embedding dimensionality consistency.
        
        For any set of document chunks processed by the embedding model, all 
        generated embeddings should have the same dimensionality, and that 
        dimensionality should match the model's specification.
        
        Validates: Requirements 4.1
        """
        # Generate random texts
        texts = [
            "".join(["word" + str(i) + " " for i in range(text_length // 10)])
            for _ in range(num_texts)
        ]
        
        # Mock response with consistent dimensionality
        # text-embedding-3-small has 1536 dimensions
        expected_dim = 1536
        
        mock_embeddings = []
        for i in range(num_texts):
            mock_embedding = [0.1 * (i + 1)] * expected_dim
            mock_embeddings.append(MagicMock(embedding=mock_embedding, index=i))
        
        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
        result = await client.embed_batch(texts)
        
        # Property 1: All embeddings should have the same dimensionality
        assert len(result) == num_texts
        
        if len(result) > 0:
            first_dim = len(result[0])
            for embedding in result:
                assert len(embedding) == first_dim, \
                    f"Inconsistent embedding dimensions: expected {first_dim}, got {len(embedding)}"
            
            # Property 2: Dimensionality should match model specification
            assert first_dim == expected_dim, \
                f"Embedding dimension {first_dim} doesn't match expected {expected_dim}"

    @given(
        text=st.text(min_size=1, max_size=1000)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_13_single_embedding_dimensionality(
        self, text, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 13: Embedding dimensionality consistency.
        
        For any single text, the embedding should have the correct dimensionality.
        
        Validates: Requirements 4.1
        """
        # Skip empty or whitespace-only text
        assume(len(text.strip()) > 0)
        
        # Mock response with correct dimensionality
        expected_dim = 1536
        mock_embedding = [0.1] * expected_dim
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests")
        result = await client.embed_text(text)
        
        # Property: Embedding should have correct dimensionality
        assert len(result) == expected_dim, \
            f"Embedding dimension {len(result)} doesn't match expected {expected_dim}"


class TestRetryWithExponentialBackoff:
    """Property-based tests for Property 16: Retry with exponential backoff."""

    @given(
        num_failures=st.integers(min_value=1, max_value=2),
        max_retries=st.integers(min_value=2, max_value=3)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_16_retry_with_exponential_backoff(
        self, num_failures, max_retries, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 16: Retry with exponential backoff.
        
        For any embedding generation that fails, the system should retry up to 
        three times with exponentially increasing delays (e.g., 1s, 2s, 4s), 
        and only fail permanently after all retries are exhausted.
        
        Validates: Requirements 4.5
        """
        # Ensure we have at least one retry
        assume(num_failures < max_retries)
        
        # Setup mock to fail num_failures times then succeed
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
        mock_request = MagicMock()
        api_error = APIError("Temporary error", request=mock_request, body=None)
        
        side_effects = [api_error] * num_failures + [mock_response]
        
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(side_effect=side_effects)
        mock_openai_client.return_value = mock_instance
        
        # Test with timing
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=max_retries)
        
        start_time = time.time()
        result = await client.embed_text("test text")
        elapsed_time = time.time() - start_time
        
        # Property 1: Should succeed after retries
        assert result == [0.1, 0.2, 0.3]
        
        # Property 2: Should have called the API num_failures + 1 times
        assert mock_instance.embeddings.create.call_count == num_failures + 1
        
        # Property 3: Should have exponential backoff delays
        # Expected delays: 1s, 2s, 4s (2^0, 2^1, 2^2)
        # Total expected delay for num_failures: sum(2^i for i in range(num_failures))
        expected_min_delay = sum(2**i for i in range(num_failures))
        
        # Allow some tolerance for execution time
        assert elapsed_time >= expected_min_delay * 0.8, \
            f"Expected at least {expected_min_delay}s delay, got {elapsed_time:.2f}s"

    @given(
        max_retries=st.integers(min_value=1, max_value=3)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_16_retry_exhaustion(
        self, max_retries, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 16: Retry with exponential backoff.
        
        For any embedding generation that fails all retries, the system should
        fail permanently after exhausting all retry attempts.
        
        Validates: Requirements 4.5
        """
        # Setup mock to always fail
        mock_request = MagicMock()
        api_error = APIError("Persistent error", request=mock_request, body=None)
        
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(side_effect=api_error)
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=max_retries)
        
        with pytest.raises(APIError):
            await client.embed_text("test text")
        
        # Property: Should have attempted exactly max_retries times
        assert mock_instance.embeddings.create.call_count == max_retries

    @given(
        failure_type=st.sampled_from([APIError, RateLimitError, APITimeoutError]),
        num_failures=st.integers(min_value=1, max_value=2)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_16_retry_different_error_types(
        self, failure_type, num_failures, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 16: Retry with exponential backoff.
        
        For any type of retryable error (APIError, RateLimitError, APITimeoutError),
        the system should retry with exponential backoff.
        
        Validates: Requirements 4.5
        """
        # Setup mock to fail with specific error type then succeed
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
        # Create appropriate error based on type
        if failure_type == APIError:
            mock_request = MagicMock()
            error = APIError("Error", request=mock_request, body=None)
        elif failure_type == RateLimitError:
            mock_http_response = MagicMock()
            mock_http_response.status_code = 429
            error = RateLimitError("Rate limit", response=mock_http_response, body=None)
        else:  # APITimeoutError
            error = APITimeoutError("Timeout")
        
        side_effects = [error] * num_failures + [mock_response]
        
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(side_effect=side_effects)
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=3)
        result = await client.embed_text("test text")
        
        # Property: Should succeed after retries regardless of error type
        assert result == [0.1, 0.2, 0.3]
        assert mock_instance.embeddings.create.call_count == num_failures + 1


class TestTimeoutHandling:
    """Property-based tests for Property 28: Timeout handling."""

    @given(
        timeout=st.floats(min_value=1.0, max_value=60.0)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_28_timeout_handling(
        self, timeout, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 28: Timeout handling.
        
        For any external service call (OpenAI API) that times out, the system 
        should catch the timeout exception and return a 503 status code with an 
        informative message about service unavailability.
        
        Validates: Requirements 10.3
        
        Note: This test verifies that timeout errors are properly raised and can
        be caught by the API layer. The API layer is responsible for converting
        the exception to a 503 status code.
        """
        # Setup mock to raise timeout error
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(
            side_effect=APITimeoutError("Request timeout")
        )
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", timeout=timeout, max_retries=2)
        
        # Property: Should raise APITimeoutError that can be caught
        with pytest.raises(APITimeoutError) as exc_info:
            await client.embed_text("test text")
        
        # Property: Error message should be informative
        error_msg = str(exc_info.value).lower()
        assert "timeout" in error_msg or "timed out" in error_msg
        
        # Property: Should have retried before failing
        assert mock_instance.embeddings.create.call_count == 2

    @given(
        operation=st.sampled_from(["embed_text", "embed_batch", "generate"])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_28_timeout_handling_all_operations(
        self, operation, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 28: Timeout handling.
        
        For any OpenAI client operation that times out, the system should
        handle the timeout gracefully and raise an appropriate exception.
        
        Validates: Requirements 10.3
        """
        # Setup mock to raise timeout error
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(
            side_effect=APITimeoutError("Request timeout")
        )
        mock_instance.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError("Request timeout")
        )
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=1)
        
        # Property: All operations should handle timeouts
        with pytest.raises(APITimeoutError):
            if operation == "embed_text":
                await client.embed_text("test text")
            elif operation == "embed_batch":
                await client.embed_batch(["text1", "text2"])
            else:  # generate
                await client.generate("test prompt")
        
        # Property: Should have attempted the operation
        if operation in ["embed_text", "embed_batch"]:
            assert mock_instance.embeddings.create.call_count >= 1
        else:
            assert mock_instance.chat.completions.create.call_count >= 1

    @given(
        max_retries=st.integers(min_value=1, max_value=3)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_28_timeout_retry_behavior(
        self, max_retries, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 28: Timeout handling.
        
        For any timeout error, the system should retry according to the
        configured max_retries before failing permanently.
        
        Validates: Requirements 10.3
        """
        # Setup mock to always timeout
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(
            side_effect=APITimeoutError("Request timeout")
        )
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=max_retries)
        
        with pytest.raises(APITimeoutError):
            await client.embed_text("test text")
        
        # Property: Should have retried exactly max_retries times
        assert mock_instance.embeddings.create.call_count == max_retries

    @given(
        num_timeouts=st.integers(min_value=1, max_value=2)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_property_28_timeout_recovery(
        self, num_timeouts, mock_openai_client
    ):
        """Feature: pdf-rag-system, Property 28: Timeout handling.
        
        For any operation that times out initially but succeeds on retry,
        the system should recover gracefully and return the result.
        
        Validates: Requirements 10.3
        """
        # Setup mock to timeout then succeed
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
        side_effects = [APITimeoutError("Timeout")] * num_timeouts + [mock_response]
        
        mock_instance = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(side_effect=side_effects)
        mock_openai_client.return_value = mock_instance
        
        # Test
        client = OpenAIClient(api_key="test-fake-api-key-for-unit-tests", max_retries=3)
        result = await client.embed_text("test text")
        
        # Property: Should recover and return result
        assert result == [0.1, 0.2, 0.3]
        
        # Property: Should have retried the correct number of times
        assert mock_instance.embeddings.create.call_count == num_timeouts + 1
