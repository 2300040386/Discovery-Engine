from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    environment: str = "development"
    log_level: str = "INFO"
    redis_url: str = "redis://localhost:6379/0"
    pg_dsn: str = "postgresql://localhost:5432/discovery"
    retrieval_service_url: str = "http://localhost:8000"
    anthropic_api_key: str | None = None
    model_version: str = "dev"

    embedding_dim: int = 128
    intent_heads: int = 4
    intent_head_collapse_cosine: float = 0.8
    session_length: int = 20
    retrieval_k_per_query: int = 150
    retrieval_merge_size: int = 500
    diversity_cap: float = 0.35
    latency_budget_total_ms: float = 80.0
    latency_budget_retrieval_ms: float = 12.0
    latency_budget_ranking_ms: float = 25.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
