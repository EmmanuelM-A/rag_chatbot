"""
Handles document ingestion, cleaning, splitting, and preparation of source
documents before embedding and storage.
"""

import os
from typing import List

import docx
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from components.config import ALLOWED_FILE_EXTENSIONS, CHUNK_SIZE, \
    CHUNK_OVERLAP
from utils.exceptions import DocumentDirectoryNotFoundError, \
    InvalidDocumentDirectoryError
from utils.logger import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class FileDocumentMetadata:
    """
    Represents metadata associated with a file-based document.

    This class is used to capture contextual information about the
    document, which can help with filtering, tracing source documents,
    or debugging during document processing in a RAG pipeline.

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



class DocumentProcessor:
    """
    Prepares raw documents for indexing into the vector store.
    """

    def __init__(self, path_to_directory: str) -> None:
        self.path_to_directory = path_to_directory
        self.documents: List[FileDocument] = []

    def load_documents(self)-> List[FileDocument]:
        """
        Loads supported documents from the directory into FileDocument objects.
        """

        if not os.path.exists(self.path_to_directory):
            raise DocumentDirectoryNotFoundError(self.path_to_directory)

        if not os.path.isdir(self.path_to_directory):
            raise InvalidDocumentDirectoryError(self.path_to_directory)

        logger.debug(f"Loading documents from: {self.path_to_directory}")

        documents = []

        for root, _, files in os.walk(self.path_to_directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()

                if ext not in ALLOWED_FILE_EXTENSIONS:
                    logger.warning(f"Skipping unsupported file: {file}")
                    continue

                file_path = os.path.join(root, file)

                try:
                    if ext in [".txt", ".md"]: # Handles txt and md
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                content = f.read()

                    elif ext == ".pdf": # Handles PDFs
                        content = ""
                        with fitz.open(file_path) as doc:
                            for page in doc:
                                content += page.get_text("text")

                    elif ext == ".docx": # Handles DOCX
                        doc = docx.Document(str(file_path))
                        content = "\n".join(
                            [para.text for para in doc.paragraphs])

                    else:
                        continue

                    file_metadata = FileDocumentMetadata(
                        filename=file,
                        file_extension=ext,
                        author=None,
                        source=file_path
                    )

                    documents.append(
                        FileDocument(
                            content=content,
                            metadata=file_metadata
                        )
                    )

                except Exception as e: # TODO: CHANGE THIS
                    logger.error(f"Failed to read {file_path}: {e}")

        logger.info("Document loading completed successfully.")

        return documents

    def _clean_documents(self):
        """
        Cleans loaded documents by stripping whitespace and removing empties.
        """

        cleaned = []

        for doc in self.documents:
            cleaned_content = doc.content.strip()
            if cleaned_content:
                cleaned.append(FileDocument(content=cleaned_content,
                                            metadata=doc.metadata))

        logger.debug(f"Cleaned {len(cleaned)} documents "
                     f"(out of {len(self.documents)}).")

        return cleaned

    def _chunk_documents(self) -> List[FileDocument]:
        """
        Splits documents into smaller chunks using a recursive character
        splitter.
        """

        logger.debug("Chunking documents...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = []
        for doc in self.documents:
            for chunk in splitter.split_text(doc.content):
                if chunk.strip():
                    chunks.append(
                        FileDocument(content=chunk, metadata=doc.metadata))

        logger.info(f"Chunking produced {len(chunks)} chunks from "
                    f"{len(self.documents)} documents.")

        return chunks

    def process_documents(self):
        """
        Full processing pipeline: load, clean, chunk.
        """

        self.documents = self.load_documents()

        self.documents = self._clean_documents()

        return self._chunk_documents()
