"""
Contains all the custom exceptions for the application.
"""

class RAGPipelineException(Exception):
    """Raised when an error or exception occurs during the RAG pipeline."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DirectoryNotFoundError(RAGPipelineException):
    """Raised when the directory does not exist."""
    def __init__(self, path: str) -> None:
        super().__init__(f"The directory '{path}' does not exist.")


class InvalidDirectoryError(RAGPipelineException):
    """Raised when the path provided is not a directory."""
    def __init__(self, path: str) -> None:
        super().__init__(f"The path '{path}' is not a directory.")


class FileTypeNotSupported(RAGPipelineException):
    """Raised when a file type is not supported"""
    def __init__(self, file_type: str) -> None:
        super().__init__(f"The file type '{file_type}' is not supported.")





