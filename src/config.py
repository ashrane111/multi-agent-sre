"""
Configuration management for Multi-Agent SRE Platform.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""
from dotenv import load_dotenv
load_dotenv()  
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # API Keys
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # Default models for each agent
    orchestrator_model: str = Field(
        default="groq/llama-3.3-70b-versatile", alias="ORCHESTRATOR_MODEL"
    )
    monitor_model: str = Field(default="ollama/llama3.1:8b", alias="MONITOR_MODEL")
    diagnose_model: str = Field(default="ollama/mistral:7b", alias="DIAGNOSE_MODEL")
    policy_model: str = Field(default="ollama/llama3.1:8b", alias="POLICY_MODEL")
    report_model: str = Field(default="ollama/llama3.1:8b", alias="REPORT_MODEL")

    # Embedding model (local)
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")

    # Rate limits
    groq_rate_limit: int = Field(default=30, alias="GROQ_RATE_LIMIT")
    gemini_rate_limit: int = Field(default=15, alias="GEMINI_RATE_LIMIT")


class ObservabilitySettings(BaseSettings):
    """Observability and tracing configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # LangSmith
    langchain_api_key: SecretStr | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="multi-agent-sre", alias="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT"
    )

    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_service_name: str = Field(default="multi-agent-sre", alias="OTEL_SERVICE_NAME")


class ReliabilitySettings(BaseSettings):
    """Reliability and resilience configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Circuit breaker
    circuit_breaker_failure_threshold: int = Field(
        default=5, alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD"
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60, alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT"
    )

    # Timeouts (seconds)
    llm_timeout: int = Field(default=120, alias="LLM_TIMEOUT")
    agent_workflow_timeout: int = Field(default=300, alias="AGENT_WORKFLOW_TIMEOUT")

    # Retries
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_backoff_base: float = Field(default=2.0, alias="RETRY_BACKOFF_BASE")

    # Cost limits
    max_cost_per_incident: float = Field(default=0.10, alias="MAX_COST_PER_INCIDENT")


class RAGSettings(BaseSettings):
    """RAG pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")

    # Chunking
    chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(default=5, alias="RAG_TOP_K")
    reranker_top_k: int = Field(default=3, alias="RERANKER_TOP_K")

    # Hybrid search weights
    dense_weight: float = Field(default=0.6)
    sparse_weight: float = Field(default=0.4)

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v


class MemorySettings(BaseSettings):
    """Episodic memory configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Mem0
    mem0_collection_name: str = Field(default="sre_incidents", alias="MEM0_COLLECTION_NAME")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")


class SecuritySettings(BaseSettings):
    """Security configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Feature flags
    enable_prompt_injection_detection: bool = Field(
        default=True, alias="ENABLE_PROMPT_INJECTION_DETECTION"
    )
    enable_pii_redaction: bool = Field(default=True, alias="ENABLE_PII_REDACTION")
    enable_output_validation: bool = Field(default=True, alias="ENABLE_OUTPUT_VALIDATION")


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    workers: int = Field(default=1, alias="API_WORKERS")


class Settings(BaseSettings):
    """Main settings class aggregating all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # Sub-configurations
    llm: LLMSettings = Field(default_factory=LLMSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api: APISettings = Field(default_factory=APISettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
