# core/__init__.py

"""
app/core/config.py
Single source of truth for all configuration.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:5500,null"

    # ChromaDB
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "glossary"

    # Embeddings
    embed_model: str = "all-MiniLM-L6-v2"
    embed_batch_size: int = 64

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.3

    # LLM — Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: int = 60

    # Rate limiting
    rate_limit: str = "30/minute"

    # Data
    data_file: str = "./data/glossary_data.json"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
