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
    ENV: str = Field(default="development", env="ENVIRONMENT")
    APP_NAME: str = Field("RAG Chatbot",)

    # Application
    RAW_DOCS_DIRECTORY: str = Field("../../../data/raw_docs")
    ALLOWED_FILE_EXTENSIONS: List[str] = Field([".pdf", ".docx", ".txt", ".md"])
    MD_FILE_EXT: str = Field(".md")
    TXT_FILE_EXT: str = Field(".txt")
    PDF_FILE_EXT: str = Field(".pdf")
    DOCX_FILE_EXT: str = Field(".docx")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_DIRECTORY: str = Field("../../../logs")
    LOG_WEB_SEARCHES: bool = Field(False)
    IS_FILE_LOGGING_ENABLED: bool = Field(False, env="LOG_TO_FILE")

    # OpenAI API
    OPEN_API_KEY: SecretStr = Field(..., env="OPENAI_API_KEY")
    EMBEDDING_MODEL_NAME: str = Field("text-embedding-3-small")

    # LLM
    LLM_MODEL_NAME: str = Field("gpt-3.5-turbo")
    LLM_TEMPERATURE: int = Field(0.7)
    RETRIEVAL_TOP_K: int = Field(3)
    RESPONSE_PROMPT_FILEPATH: SecretStr = Field(
        "../../../data/prompts/default_response_prompt.yaml")

    # Vector DB & Processing
    VECTOR_DB_FILE_PATH: str = Field("../../../data/db/vector_index.faiss")
    METADATA_DB_FILE_PATH: str = Field("../../../data/db/metadata.pkl")
    CHUNK_SIZE: int = Field(1000)
    CHUNK_OVERLAP: int = Field(20)

    # Web Search
    IS_WEB_SEARCH_ENABLED: bool = Field(False)
    SEARCH_API_KEY: SecretStr = Field(..., env="SEARCH_API_KEY")
    SEARCH_ENGINE_ID: SecretStr = Field(..., env="SEARCH_ENGINE_ID")

    # Evaluation
    QA_SQLITE_DB_PATH: str = Field("../../../data/db/qa_log.db")

    class Config:
        """Configurations for AppSettings"""
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = AppSettings()
