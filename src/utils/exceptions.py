"""
Contains all the custom exceptions for the application.
"""

class RAGPipelineException(Exception):
    """Raised when an error or exception occurs during the RAG pipeline."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DocumentDirectoryNotFoundError(RAGPipelineException):
    """Raised when the document directory does not exist."""
    def __init__(self, path: str) -> None:
        super().__init__(f"The directory '{path}' does not exist.")


class InvalidDocumentDirectoryError(RAGPipelineException):
    """Raised when the path provided is not a directory."""
    def __init__(self, path: str) -> None:
        super().__init__(f"The path '{path}' is not a directory.")
