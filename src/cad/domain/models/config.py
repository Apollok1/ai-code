"""
CAD Estimator Pro - Configuration Models (Pydantic v2)

Validated configuration classes for application settings.
"""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Literal


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""

    model_config = ConfigDict(frozen=True)

    host: str = Field(default="cad-postgres", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="cad_estimator", description="Database name")
    user: str = Field(default="cad_user", description="Database user")
    password: str = Field(default="cad_password_2024", description="Database password")
    pool_min_size: int = Field(default=2, ge=1, description="Min connection pool size")
    pool_max_size: int = Field(default=10, ge=1, le=100, description="Max connection pool size")


class OllamaConfig(BaseModel):
    """Ollama AI service configuration."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(default="http://ollama:11434", description="Ollama server URL")
    text_model: str = Field(default="qwen2.5:7b", description="Model for text/JSON analysis")
    vision_model: str | None = Field(default="qwen2.5vl:7b", description="Model for image analysis")
    embed_model: str = Field(default="nomic-embed-text", description="Embedding model")
    embed_dim: int = Field(default=768, ge=128, le=4096, description="Embedding dimension")
    cache_ttl_seconds: int = Field(default=300, ge=0, description="Model cache TTL")
    timeout_seconds: int = Field(default=500, ge=10, le=600, description="Request timeout")


class ParserConfig(BaseModel):
    """File parser configuration."""

    model_config = ConfigDict(frozen=True)

    excel_data_start_row: int = Field(default=11, ge=0, description="Excel data start row (0-indexed)")
    excel_multipliers_row: int = Field(default=9, ge=0, description="Excel multipliers row (0-indexed)")
    pdf_max_pages: int = Field(default=200, ge=1, description="Max PDF pages to process")
    image_max_resolution: int = Field(default=1280, ge=256, le=4096, description="Max image size (px)")
    image_quality: int = Field(default=85, ge=1, le=100, description="JPEG quality for compression")


class LearningConfig(BaseModel):
    """Machine learning configuration."""

    model_config = ConfigDict(frozen=True)

    enable_pattern_learning: bool = Field(default=True, description="Enable pattern learning from feedback")
    enable_bundle_learning: bool = Field(default=True, description="Enable bundle (parent→sub) learning")
    welford_outlier_threshold: float = Field(default=3.0, ge=1.0, description="Z-score threshold for outlier detection (e.g., 3.0 = 3 std deviations)")
    welford_min_n: int = Field(default=5, ge=2, description="Minimal number of observations before applying Z-score outlier detection")
    min_pattern_occurrences: int = Field(default=2, ge=1, description="Min occurrences to keep pattern")
    min_bundle_occurrences: int = Field(default=2, ge=1, description="Min occurrences to keep bundle")
    fuzzy_match_threshold: int = Field(default=88, ge=0, le=100, description="Fuzzy match threshold (0-100)")
    semantic_similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Semantic similarity threshold")


class MultiModelConfig(BaseModel):
    """Multi-model pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable multi-model pipeline")
    stage1_model: str = Field(default="qwen2.5:14b", description="Technical analysis model (reasoning)")
    stage2_model: str = Field(default="qwen2.5:14b", description="Structural decomposition model (CRITICAL - errors here propagate)")
    stage3_model: str = Field(default="qwen2.5:7b", description="Hours estimation model (fast + pattern matching)")
    stage4_model: str = Field(default="qwen2.5:14b", description="Risk & optimization model (critical)")
    fallback_model: str = Field(default="qwen2.5:7b", description="Fallback if stage model unavailable")

    # Model family consistency (prefer staying within one family)
    preferred_family: str = Field(default="qwen2.5", description="Preferred model family for consistency")


class UIConfig(BaseModel):
    """UI/Presentation configuration."""

    model_config = ConfigDict(frozen=True)

    hourly_rate_default: int = Field(default=150, ge=1, description="Default hourly rate PLN")
    max_workers: int = Field(default=4, ge=1, le=32, description="Max parallel workers for batch import")
    enable_web_lookup: bool = Field(default=False, description="Enable web lookup for norms/benchmarks")
    page_title: str = Field(default="CAD Estimator Pro", description="Streamlit page title")
    page_icon: str = Field(default="🚀", description="Streamlit page icon")


class AppConfig(BaseModel):
    """
    Complete application configuration.

    Can be loaded from environment variables or .env file.
    """

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        frozen=True
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    multi_model: MultiModelConfig = Field(default_factory=MultiModelConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Load configuration from environment variables.

        Environment variables:
        - DATABASE__HOST -> database.host
        - OLLAMA__URL -> ollama.url
        - etc.

        Returns:
            AppConfig instance
        """
        return cls()

    @classmethod
    def for_testing(cls) -> "AppConfig":
        """
        Create minimal configuration for testing.

        Returns:
            AppConfig with test-friendly defaults
        """
        return cls(
            database=DatabaseConfig(
                host="localhost",
                database="cad_estimator_test",
                user="test_user",
                password="test_pass"
            ),
            ollama=OllamaConfig(
                url="http://localhost:11434",
                text_model="llama3:latest",
                vision_model=None,
                cache_ttl_seconds=0  # Disable cache for tests
            ),
            ui=UIConfig(max_workers=2)  # Limit workers in tests
        )
