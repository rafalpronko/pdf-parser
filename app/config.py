"""Application configuration using Pydantic Settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    All settings are validated at startup. Missing required settings
    or invalid values will cause the application to fail fast with
    clear error messages.
    """

    # API Settings
    api_title: str = Field(default="RAG-Anything Multimodal System", description="API title")
    api_version: str = Field(default="2.0.0", description="API version")

    # LLM Settings
    llm_provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, local)",
    )
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for text generation",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for text embeddings",
    )

    # VLM (Vision-Language Model) Settings
    vlm_provider: str = Field(
        default="openai",
        description="VLM provider (openai for gpt-4v, local for llava/qwen-vl)",
    )
    vlm_model: str = Field(
        default="gpt-4-vision-preview",
        description="VLM model for visual understanding",
    )
    enable_vlm: bool = Field(
        default=True,
        description="Enable VLM for visual content understanding",
    )

    # Vision Encoder Settings
    vision_encoder: str = Field(
        default="openai/clip-vit-large-patch14",
        description="Vision encoder for visual embeddings (clip-vit-large, siglip, etc.)",
    )

    # RAG-Anything Settings
    use_rag_anything: bool = Field(
        default=True,
        description="Use RAG-Anything framework for multimodal processing",
    )
    mineru_config_path: str | None = Field(
        default=None,
        description="Path to MinerU configuration file",
    )

    # Processing Settings
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Maximum chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks",
    )
    max_file_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        ge=1024,  # At least 1KB
        description="Maximum upload file size in bytes",
    )
    enable_multimodal_chunking: bool = Field(
        default=True,
        description="Enable multimodal chunking (text + visual)",
    )

    # Vector Database Settings
    vector_db_path: str = Field(
        default="./data/vectordb",
        description="Path to vector database storage",
    )
    text_collection: str = Field(
        default="text_chunks",
        min_length=1,
        description="Collection name for text chunks",
    )
    visual_collection: str = Field(
        default="visual_chunks",
        min_length=1,
        description="Collection name for visual chunks",
    )
    multimodal_collection: str = Field(
        default="multimodal_chunks",
        min_length=1,
        description="Collection name for multimodal chunks",
    )

    # Storage Settings
    upload_dir: Path = Field(
        default=Path("./data/uploads"),
        description="Directory for uploaded files",
    )

    # Logging Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Retrieval Enhancement Settings
    # Reranking
    enable_reranking: bool = Field(
        default=True,
        description="Enable cross-encoder reranking of results",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking (bge-reranker-v2-m3 recommended for multilingual)",
    )
    reranking_top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Number of initial candidates to retrieve for reranking",
    )
    final_top_k: int = Field(
        default=10,
        ge=1,
        le=40,
        description="Number of results to return after reranking",
    )

    # Hybrid Search
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid search (vector + keyword)",
    )
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid search",
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid search",
    )
    bm25_k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 k1 parameter (term frequency saturation)",
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter (length normalization)",
    )

    # Query Expansion
    enable_query_expansion: bool = Field(
        default=True,
        description="Enable query expansion for better retrieval",
    )
    expansion_method: str = Field(
        default="multi-query",
        description="Query expansion method (hyde, multi-query, hybrid, none)",
    )
    num_query_variations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of query variations to generate",
    )
    expansion_cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Query expansion cache TTL in seconds",
    )

    # Chunking Strategy
    chunking_strategy: str = Field(
        default="semantic",
        description="Chunking strategy (fixed, semantic, sentence-window)",
    )
    use_structure_aware_chunking: bool = Field(
        default=True,
        description="Use document structure for chunking",
    )
    sentence_window_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Sentence window size for sentence-window chunking",
    )

    # Performance
    enable_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration for reranking",
    )
    reranking_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for reranking operations",
    )
    cache_reranking_scores: bool = Field(
        default=True,
        description="Cache reranking scores for repeated queries",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        # Note: chunk_size might not be set yet during validation
        # This will be checked in model_post_init if needed
        if v < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return v

    @field_validator("expansion_method")
    @classmethod
    def validate_expansion_method(cls, v: str) -> str:
        """Ensure expansion method is valid."""
        valid_methods = {"hyde", "multi-query", "hybrid", "none"}
        v_lower = v.lower()
        if v_lower not in valid_methods:
            raise ValueError(
                f"expansion_method must be one of {valid_methods}, got '{v}'"
            )
        return v_lower

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, v: str) -> str:
        """Ensure chunking strategy is valid."""
        valid_strategies = {"fixed", "semantic", "sentence-window"}
        v_lower = v.lower()
        if v_lower not in valid_strategies:
            raise ValueError(
                f"chunking_strategy must be one of {valid_strategies}, got '{v}'"
            )
        return v_lower

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got '{v}'"
            )
        return v_upper

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Ensure API key is not empty and has reasonable format if provided."""
        if v is None:
            return v
        if v.strip() == "":
            raise ValueError("openai_api_key cannot be empty string")
        if v == "your-openai-api-key-here":
            raise ValueError(
                "openai_api_key must be set to a valid API key, "
                "not the placeholder value"
            )
        # OpenAI keys typically start with 'sk-'
        if not v.startswith("sk-"):
            raise ValueError(
                "openai_api_key should start with 'sk-' "
                "(OpenAI API key format)"
            )
        return v

    @field_validator("upload_dir", mode="before")
    @classmethod
    def validate_upload_dir(cls, v) -> Path:
        """Convert string to Path and ensure it's valid."""
        if isinstance(v, str):
            return Path(v)
        return v

    def model_post_init(self, __context) -> None:
        """Additional validation after model initialization."""
        # Ensure chunk overlap is less than chunk size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        # Normalize hybrid search weights to sum to 1.0
        weight_sum = self.vector_weight + self.keyword_weight
        if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point errors
            self.vector_weight = self.vector_weight / weight_sum
            self.keyword_weight = self.keyword_weight / weight_sum

        # Validate reranking_top_k >= final_top_k
        if self.reranking_top_k < self.final_top_k:
            raise ValueError(
                f"reranking_top_k ({self.reranking_top_k}) must be >= "
                f"final_top_k ({self.final_top_k})"
            )

        # Create upload directory if it doesn't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Create vector db directory if it doesn't exist
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
# This will be initialized when the module is imported
# and will fail fast if configuration is invalid
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance.
    
    Returns:
        Settings: The application settings
        
    Raises:
        ValueError: If settings validation fails
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Convenience function to reload settings (useful for testing)
def reload_settings() -> Settings:
    """Reload settings from environment.
    
    Returns:
        Settings: The reloaded application settings
    """
    global _settings
    _settings = Settings()
    return _settings
