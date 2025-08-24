"""
Contains all the custom exceptions for the application.
"""


class RAGChatbotError(Exception):
    """Base exception for RAG chatbot errors."""
    def __init__(
        self,
        message: str = "An error occurred during the RAG Chatbot Pipeline."
    ) -> None:
        super().__init__(message)


class DocumentProcessingError(RAGChatbotError):
    """Raised when document processing fails."""


class EmbeddingError(RAGChatbotError):
    """Raised when embedding creation fails."""


class VectorStoreError(RAGChatbotError):
    """Raised when vector store operations fail."""


class QueryProcessingError(RAGChatbotError):
    """Raised when query processing fails."""


class DirectoryNotFoundError(RAGChatbotError):
    """Raised when the directory does not exist."""
    def __init__(self, path: str) -> None:
        super().__init__(f"The directory '{path}' does not exist.")


class InvalidDirectoryError(RAGChatbotError):
    """Raised when the path provided is not a directory."""
    def __init__(self, path: str) -> None:
        super().__init__(f"The path '{path}' is not a directory.")


class FileTypeNotSupported(RAGChatbotError):
    """Raised when a file type is not supported"""
    def __init__(self, file_type: str) -> None:
        super().__init__(f"The file type '{file_type}' is not supported.")


class FileDoesNotExist(RAGChatbotError, OSError):
    """Raised when a file cannot be found or does not exist"""
    def __init__(
            self,
            message: str = "The file does not exist or cannot be found."
    ) -> None:
        super().__init__(message)

class SettingConfigError(RAGChatbotError, ValueError):
    """Raised when an error occurs during setting/config initialization."""
