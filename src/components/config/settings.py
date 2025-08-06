"""
Contains all the settings needed to run the application as well as
environment variables.
"""

import os
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


class AppSettings(BaseSettings):
    """All application settings, env variables and static configurations."""

    # General
    environment: str = Field(default="development", env="ENVIRONMENT")
    app_name: str = Field("RAG Chatbot",)

    # Application
    RAW_DOCS_DIRECTORY: str = Field("../../../data/raw_docs")
    ALLOWED_FILE_EXTENSIONS: List[str] = Field([".pdf", ".docx", ".txt", ".md"])
    MD_FILE_EXT: str = ".md"
    TXT_FILE_EXT: str = ".txt"
    PDF_FILE_EXT: str = ".pdf"
    DOCX_FILE_EXT: str = ".docx"

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_directory: str = Field("../../../logs")
    log_web_searches: bool = Field(False)
    is_file_logging_enabled: bool = Field(False, env="LOG_TO_FILE")

    # OpenAI API
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    embedding_model_name: str = Field("text-embedding-3-small")

    # LLM
    llm_model_name: str = Field("gpt-3.5-turbo")
    llm_temperature: int = Field(0.7)
    retrieval_top_k: int = Field(3)
    response_prompt_file_path: SecretStr = Field(
        "../../../data/prompts/default_response_prompt.yaml")

    # Vector DB & Processing
    vector_db_path: str = Field("../../../data/db/vector_index.faiss")
    metadata_db_path: str = Field("../../../data/db/metadata.pkl")
    chunk_size: int = Field(1000)
    chunk_overlap: int = Field(20)

    # Web Search
    is_web_search_enabled: bool = Field(False)
    search_api_key: SecretStr = Field(..., env="SEARCH_API_KEY")
    search_engine_id: SecretStr = Field(..., env="SEARCH_ENGINE_ID")

    # Evaluation
    qa_sqlite_db_path: str = Field("../../../data/db/qa_log.db")

    class Config:
        """Configurations for AppSettings"""
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = AppSettings()
