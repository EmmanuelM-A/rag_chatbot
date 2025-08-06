"""
Contains all the settings needed to run the application as well as
environment variables.
"""

from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


class AppSettings(BaseSettings):
    """All application settings, env variables and static configurations."""

    # General
    ENV: str = Field(default="development", env="ENV")
    APP_NAME: str = Field(default="RAG Chatbot")

    # Application
    RAW_DOCS_DIRECTORY: str = Field(default="../../../data/raw_docs")
    ALLOWED_FILE_EXTENSIONS: List[str] = Field(default=[".pdf", ".docx", ".txt", ".md"])
    MD_FILE_EXT: str = Field(default=".md")
    TXT_FILE_EXT: str = Field(default=".txt")
    PDF_FILE_EXT: str = Field(default=".pdf")
    DOCX_FILE_EXT: str = Field(default=".docx")

    # Logging
    LOG_LEVEL: str = Field(default="DEBUG")
    LOG_DIRECTORY: str = Field(default="../../../logs")
    LOG_WEB_SEARCHES: bool = Field(default=False)
    IS_FILE_LOGGING_ENABLED: bool = Field(default=False)

    # LLM
    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-3-small")
    LLM_MODEL_NAME: str = Field(default="gpt-3.5-turbo")
    LLM_TEMPERATURE: float = Field(default=0.7)  # Changed from int to float
    RETRIEVAL_TOP_K: int = Field(default=3)
    RESPONSE_PROMPT_FILEPATH: str = Field(  # Removed SecretStr - not sensitive
        default="../../../data/prompts/default_response_prompt.yaml")

    # Vector DB & Processing
    VECTOR_DB_FILE_PATH: str = Field(default="../../../data/db/vector_index.faiss")
    METADATA_DB_FILE_PATH: str = Field(default="../../../data/db/metadata.pkl")
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=20)

    # Web Search (Optional - only required if web search is enabled)
    IS_WEB_SEARCH_ENABLED: bool = Field(default=False)
    MAX_WEB_SEARCH_RESULTS: int = Field(default=5)
    SEARCH_API_KEY: Optional[SecretStr] = Field(default=None, env="SEARCH_API_KEY")
    SEARCH_ENGINE_ID: Optional[SecretStr] = Field(default=None, env="SEARCH_ENGINE_ID")

    # Evaluation
    QA_SQLITE_DB_PATH: str = Field(default="../../../data/db/qa_log.db")

    class Config:
        """Configurations for AppSettings"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow extra fields and don't validate assignment
        extra = "ignore"

    def __init__(self, **kwargs):
        """Initialize settings with validation for web search dependencies."""
        super().__init__(**kwargs)

        # Validate web search configuration
        if self.IS_WEB_SEARCH_ENABLED:
            if not self.SEARCH_API_KEY or not self.SEARCH_ENGINE_ID:
                print("⚠️  Warning: Web search is enabled but API keys are missing.")
                print("   Web search will use fallback methods.")


settings = AppSettings()
