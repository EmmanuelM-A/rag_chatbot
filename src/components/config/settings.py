"""
Contains all the settings needed to run the application as well as
environment variables.
"""

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

from src.components.config.logger import logger
from src.utils.exceptions import SettingConfigError

ENV_FILE = Path(__file__).resolve().parent.parent.parent.parent / ".env"


class AppSettings(BaseSettings):
    """
    Settings and configurations for the general use.
    """

    ENV: str = Field(default="development", env="ENV")
    APP_NAME: str = Field(default="RAG Chatbot")
    RAW_DOCS_DIRECTORY: str = Field(default="../data/raw_docs")
    ALLOWED_FILE_EXTENSIONS: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md"])
    MD_FILE_EXT: str = Field(default=".md")
    TXT_FILE_EXT: str = Field(default=".txt")
    PDF_FILE_EXT: str = Field(default=".pdf")
    DOCX_FILE_EXT: str = Field(default=".docx")

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


class LogSettings(BaseSettings):
    """
    Settings and configurations for the logging mechanism.
    """

    LOG_LEVEL: str = Field(default="DEBUG")
    LOG_DIRECTORY: str = Field(default="../logs")
    LOG_FORMAT: str = Field(default="%(asctime)s [%(levelname)s]: %(message)s")
    LOG_DATE_FORMAT: str = Field(default="%Y-%m-%d %H:%M:%S")
    LOG_WEB_SEARCHES: bool = Field(default=False)
    IS_FILE_LOGGING_ENABLED: bool = Field(default=False)

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


class LLMSettings(BaseSettings):
    """
    Settings and configurations for using the LLM and OPENAI models.
    """

    OPEN_API_KEY: SecretStr = Field(
        default=..., env="OPEN_API_KEY"
    )
    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-3-small")
    LLM_MODEL_NAME: str = Field(default="gpt-3.5-turbo")
    LLM_TEMPERATURE: float = Field(default=0.7)
    RETRIEVAL_TOP_K: int = Field(default=3)
    RESPONSE_PROMPT_FILEPATH: str = Field(
        default="../data/prompts/default_response_prompt.yaml"
    )

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


class VectorSettings(BaseSettings):
    """
    Settings and configurations for vector storage and processing.
    """

    VECTOR_DB_FILE_PATH: str = Field(
        default="../../../data/db/vector_index.faiss")
    METADATA_DB_FILE_PATH: str = Field(default="../../../data/db/metadata.pkl")
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=20)
    MAX_VECTORS_IN_MEMORY: int = Field(default=10000)
    VECTOR_BATCH_SIZE: int = Field(default=100)
    EMBEDDING_CACHE_ENABLED: bool = Field(default=True)
    EMBEDDING_CACHE_DIR: str = Field(default="../data/cache/embeddings")
    MAX_CACHE_SIZE_MB: int = Field(default=500)

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


class APISettings(BaseSettings):
    """
    Settings and configurations for the llm and openapi models.
    """

    OPENAI_API_RATE_LIMIT: int = Field(default=60)
    OPENAI_API_TIMEOUT_SEC: int = Field(default=30)
    MAX_API_RETRIES: int = Field(default=3)
    API_RETRY_DELAY_SEC: int = Field(default=1)

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


class WebSearchSettings(BaseSettings):
    """
    Settings and configurations for the web search mechanism.
    """

    IS_WEB_SEARCH_ENABLED: bool = Field(default=True)
    MAX_WEB_SEARCH_RESULTS: int = Field(default=5)
    WEB_REQUEST_TIMEOUT_SECS: int = Field(default=15)
    WEB_REQUEST_DELAY_SECS: int = Field(default=1)
    MAX_WEB_CONTENT_LENGTH: int = Field(default=10000)
    MIN_WEB_CONTENT_LENGTH: int = Field(default=100)

    WEB_USER_AGENT: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(HTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    SEARCH_API_KEY: Optional[SecretStr] = Field(
        default=None, env="SEARCH_API_KEY"
    )
    SEARCH_ENGINE_ID: Optional[SecretStr] = Field(
        default=None, env="SEARCH_ENGINE_ID"
    )

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


class EvalSettings(BaseSettings):
    """
    Settings and configurations for evaluation mechanism.
    """

    QA_SQLITE_DB_PATH: str = Field(default="../data/db/qa_log.db")

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


class Settings(BaseSettings):
    """
    All settings and configurations used in the application.
    """

    app: AppSettings = AppSettings()
    logging: LogSettings = LogSettings()
    llm: LLMSettings = LLMSettings()
    vector: VectorSettings = VectorSettings()
    api: APISettings = APISettings()
    web: WebSearchSettings = WebSearchSettings()
    eval: EvalSettings = EvalSettings()

    def __init__(self, **kwargs):
        """
        Setting validations for required settings/configs.
        """

        super().__init__(**kwargs)

        if (self.web.IS_WEB_SEARCH_ENABLED and
            (not self.web.SEARCH_API_KEY or not self.web.SEARCH_ENGINE_ID)
        ):
            logger.warn("Web search is enabled but API keys are missing.")
            raise SettingConfigError(
                setting_name="WEB_SEARCH_SETTINGS",
                issue="Missing API keys detected!"
            )

        if not self.llm.OPEN_API_KEY:
            logger.warn("OPEN_API_KEY is missing in your '.env' file")
            raise SettingConfigError(
                setting_name="LLM_SETTINGS",
                issue="Missing API keys detected!"
            )


settings = Settings()
