"""OpenAI client for embeddings and chat completions."""

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI API with retry logic and error handling.
    
    Provides async methods for generating embeddings and chat completions
    with built-in retry logic, timeout handling, and comprehensive error
    responses.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name for chat completions (default: gpt-4o-mini)
            embedding_model: Model name for embeddings (default: text-embedding-3-small)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # We handle retries manually
        )
        self.model = model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_retries = max_retries

    async def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with exponential backoff retry logic.
        
        Retries up to max_retries times with exponentially increasing delays:
        1s, 2s, 4s, etc.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from successful function execution
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except (APIError, RateLimitError, APITimeoutError) as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff: 1s, 2s, 4s
                    delay = 2 ** attempt
                    logger.warning(
                        f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}: {str(e)}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"OpenAI API call failed after {self.max_retries} attempts: "
                        f"{type(e).__name__}: {str(e)}"
                    )
        
        # If we get here, all retries failed
        raise last_exception

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            APITimeoutError: If request times out after retries
            APIError: If API call fails after retries
        """
        async def _embed():
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        
        return await self._retry_with_backoff(_embed)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            APITimeoutError: If request times out after retries
            APIError: If API call fails after retries
        """
        if not texts:
            return []
        
        async def _embed():
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        
        return await self._retry_with_backoff(_embed)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_message: str | None = None,
    ) -> str:
        """Generate completion from OpenAI.
        
        Args:
            prompt: User prompt/question
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            system_message: Optional system message to set context
            
        Returns:
            Generated text response
            
        Raises:
            APITimeoutError: If request times out after retries
            APIError: If API call fails after retries
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        async def _generate():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        
        return await self._retry_with_backoff(_generate)

    async def generate_with_context(
        self,
        question: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate answer with retrieved context (RAG pattern).
        
        Args:
            question: User's question
            context: Retrieved context from documents
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer based on context
            
        Raises:
            APITimeoutError: If request times out after retries
            APIError: If API call fails after retries
        """
        system_message = (
            "You are a helpful assistant that answers questions based ONLY on the provided context. "
            "Extract and provide the exact, complete information from the context. "
            "Include all relevant details, lists, and specific points mentioned in the context. "
            "If the answer cannot be found in the context, say so clearly. "
            "Maintain the original language of the context in your answer."
        )
        
        prompt = f"""Based on the following context, answer the question completely and accurately.
Include all relevant details from the context.

Context:
{context}

Question: {question}

Provide a complete answer with all details from the context:"""
        
        return await self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
        )

    async def close(self):
        """Close the client connection."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
