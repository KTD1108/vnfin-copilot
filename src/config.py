"""Cấu hình tập trung cho toàn bộ hệ thống VnFinCopilot."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


# Hằng số cấu hình cho Giai đoạn 1 (RAG)
DEFAULT_DENSE_MODEL: str = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
DENSE_VECTOR_SIZE: int = 1024
QDRANT_COLLECTION_NAME: str = "vnfin_financial_docs"

# Hằng số Chunking & Retrieval
CHILD_CHUNK_MAX_TOKENS: int = 250
PARENT_CHUNK_MAX_TOKENS: int = 1000
TABLE_KEEP_WHOLE_MAX_TOKENS: int = 800
CHARS_PER_TOKEN_ESTIMATE: int = 4
RRF_K_CONSTANT: int = 60
STAGE1_TOP_K: int = 25
STAGE2_TOP_K: int = 5
UPSERT_BATCH_SIZE: int = 64


class Settings(BaseSettings):
    """Đọc cấu hình từ biến môi trường / file .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    dense_model_name: str = DEFAULT_DENSE_MODEL
    reranker_model_name: str = DEFAULT_RERANKER_MODEL
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Trả về singleton Settings instance."""
    return Settings()
