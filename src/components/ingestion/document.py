"""
Contains all the dataclasses that store the information about the documents.
"""


from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class FileDocumentMetadata:
    """
    Represents metadata associated with a file-based document.

    This class is used to capture contextual information about the
    document, which can help with filtering, tracing source documents,
    or debugging during document processing in the RAG pipeline.

    Attributes:
        filename (str): The original name of the file.
        file_extension (str): The file's extension (e.g., '.pdf', '.txt').
        author (Optional[str]): The author of the document, if known.
        created_at (Optional[datetime]): Timestamp when the document was created.
        modified_at (Optional[datetime]): Timestamp when the document was last modified.
        source (Optional[str]): Where the file came from (e.g., URL, uploader).
        document_id (Optional[str]): An internal unique ID, if available.
    """

    filename: str
    file_extension: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    source: Optional[str] = None
    document_id: Optional[str] = None


@dataclass
class FileDocument:
    """
    Represents a document with its content and metadata.

    It contains the main content of the document (e.g., text from a file) and
    a dictionary or object containing metadata about the document (e.g.,
    filename, file extension, author, etc.).
    """

    content: str
    metadata: FileDocumentMetadata

    def __str__(self):
        """Returns the name of the document"""

        return self.metadata.filename
