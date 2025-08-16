"""
Contains all the settings needed to run the application as well as
environment variables.
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

DOTENV = Path(__file__).resolve().parent.parent.parent.parent / ".env"


class AppSettings(BaseSettings):
    """All application settings, env variables and static configurations."""

    # General
    ENV: str = Field(default="development", env="ENV")
    APP_NAME: str = Field(default="RAG Chatbot")

    # Application
    RAW_DOCS_DIRECTORY: str = Field(default="../data/raw_docs")
    ALLOWED_FILE_EXTENSIONS: List[str] = Field(default=[".pdf", ".docx", ".txt", ".md"])
    MD_FILE_EXT: str = Field(default=".md")
    TXT_FILE_EXT: str = Field(default=".txt")
    PDF_FILE_EXT: str = Field(default=".pdf")
    DOCX_FILE_EXT: str = Field(default=".docx")

    # Logging
    LOG_LEVEL: str = Field(default="DEBUG")
    LOG_DIRECTORY: str = Field(default="../logs")
    LOG_WEB_SEARCHES: bool = Field(default=False)
    IS_FILE_LOGGING_ENABLED: bool = Field(default=False)

    # LLM
    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-3-small")
    LLM_MODEL_NAME: str = Field(default="gpt-3.5-turbo")
    LLM_TEMPERATURE: float = Field(default=0.7)  # Changed from int to float
    RETRIEVAL_TOP_K: int = Field(default=3)
    RESPONSE_PROMPT_FILEPATH: str = Field(  # Removed SecretStr - not sensitive
        default="../data/prompts/default_response_prompt.yaml")

    # Vector DB & Processing
    VECTOR_DB_FILE_PATH: str = Field(default="../../../data/db/vector_index.faiss")
    METADATA_DB_FILE_PATH: str = Field(default="../../../data/db/metadata.pkl")
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=20)

    # API Configuration
    OPENAI_API_RATE_LIMIT: int = Field(default=60)
    OPENAI_API_TIMEOUT_SEC: int = Field(default=30)
    MAX_API_RETRIES: int = Field(default=3)
    API_RETRY_DELAY_SEC: int = Field(default=1)

    # Web Search
    IS_WEB_SEARCH_ENABLED: bool = Field(default=True)
    MAX_WEB_SEARCH_RESULTS: int = Field(default=5)
    WEB_REQUEST_TIMEOUT_SECS: int = Field(default=15)
    WEB_REQUEST_DELAY_SECS: int = Field(default=1)
    MAX_WEB_CONTENT_LENGTH: int = Field(default=10000)
    MIN_WEB_CONTENT_LENGTH: int = Field(default=100)
    WEB_USER_AGENT: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    SEARCH_API_KEY: Optional[SecretStr] = Field(default=None, env="SEARCH_API_KEY")
    SEARCH_ENGINE_ID: Optional[SecretStr] = Field(default=None, env="SEARCH_ENGINE_ID")

    # Performance Configuration
    MAX_VECTORS_IN_MEMORY: int = Field(default=10000)
    VECTOR_BATCH_SIZE: int = Field(default=100)
    EMBEDDING_CACHE_ENABLED: bool = Field(default=True)
    EMBEDDING_CACHE_DIR: str = Field(default="../data/cache/embeddings")
    MAX_CACHE_SIZE_MB: int = Field(default=500)

    # Evaluation
    QA_SQLITE_DB_PATH: str = Field(default="../data/db/qa_log.db")

    model_config = SettingsConfigDict(env_file=DOTENV, extra="ignore")

    def __init__(self, **kwargs):
        """Initialize settings with validation for web search dependencies."""
        super().__init__(**kwargs)

        # Validate web search configuration
        if self.IS_WEB_SEARCH_ENABLED:
            if not self.SEARCH_API_KEY or not self.SEARCH_ENGINE_ID:
                print("⚠️  Warning: Web search is enabled but API keys are missing.")
                print("   Web search will use fallback methods.")


settings = AppSettings()
