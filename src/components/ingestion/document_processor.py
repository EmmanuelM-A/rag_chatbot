"""
Handles document ingestion, cleaning, splitting, and preparation of source
documents before embedding and storage.
"""

import os
from abc import ABC
from typing import List, override

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.components.config.settings import settings
from src.components.ingestion.document import FileDocument
from src.components.ingestion.document_loader import PDFDocumentLoader, \
    MarkdownDocumentLoader, TxTDocumentLoader, DocxDocumentLoader, \
    DocumentLoader
from src.utils.exceptions import FileTypeNotSupported, DirectoryNotFoundError, \
    InvalidDirectoryError, RAGPipelineException
from src.components.config.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor(ABC):
    """
    Abstract class responsible defining the operations a document processor
    can complete.
    """

    def __init__(
            self,
            path_to_directory: str
    ) -> None:
        self.path_to_directory = path_to_directory
        self.documents = []
        self.document_loader_mappings = {
            settings.PDF_FILE_EXT: PDFDocumentLoader(),
            settings.MD_FILE_EXT: MarkdownDocumentLoader(),
            settings.TXT_FILE_EXT: TxTDocumentLoader(),
            settings.DOCX_FILE_EXT: DocxDocumentLoader()
        }

    def load_documents(self) -> None:
        """
        Loads supported documents from the directory into FileDocument objects
        and stores it in the self.documents list.
        """

        if not os.path.exists(self.path_to_directory):
            raise DirectoryNotFoundError(self.path_to_directory)

        if not os.path.isdir(self.path_to_directory):
            raise InvalidDirectoryError(self.path_to_directory)

        logger.debug(f"Loading documents from: {self.path_to_directory}")

        documents = []

        for root, _, files in os.walk(self.path_to_directory):
            for file in files:

                file_extension = os.path.splitext(file)[1].lower()

                if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
                    logger.warning(f"Skipping unsupported file: {file}")
                    continue

                file_path = os.path.join(root, file)

                try:
                    file_loader = self._get_loader(file_extension)

                    document = file_loader.load_data(file_path)

                    documents.append(document)
                except (ValueError, OSError, RAGPipelineException) as e:
                    logger.error(f"Failed to read {file_path}: {e}")

        logger.debug("Document loading completed successfully.")

        self.documents = documents

    def _get_loader(self, file_extension) -> DocumentLoader:
        """
        Gets the correct document loader based on the document's file
        extension.
        """

        file_extension = file_extension.lower()

        if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
            raise FileTypeNotSupported(file_extension)

        return self.document_loader_mappings.get(file_extension, None)

    def _clean_documents(self) -> List[FileDocument]:
        """
        Cleans the loaded documents via the implemented cleaning process.
        """

        raise NotImplementedError(
            "This method must be implemented by subclasses!"
        )

    def _chunk_documents(self, documents):
        """
        Splits documents into smaller chunks using the implemented character
        splitter.
        """

        raise NotImplementedError(
            "This method must be implemented by subclasses!"
        )

    def process_documents(self):
        """
        Full processing pipeline: load, clean, chunk.
        """

        self.load_documents()

        cleaned_documents = self._clean_documents()

        chunked_documents = self._chunk_documents(cleaned_documents)

        logger.debug(
            "Documents have been loaded, processed and chunked successfully!"
        )

        return chunked_documents


class DefaultDocumentProcessor(DocumentProcessor):
    """
    Prepares raw documents for indexing into the vector store.
    """

    def __init__(self, path_to_directory: str) -> None:
        super().__init__(path_to_directory)

    @override
    def _chunk_documents(self, documents):
        """
        Splits documents into smaller chunks using a recursive character
        splitter and returns a list of chunked documents.
        """

        logger.debug("Chunking documents...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        chunks = []
        for doc in self.documents:
            for chunk in splitter.split_text(doc.content):
                if chunk.strip():
                    chunks.append(
                        FileDocument(content=chunk, metadata=doc.metadata))

        logger.debug(f"Chunking produced {len(chunks)} chunks from "
                    f"{len(self.documents)} documents.")

        return chunks

    @override
    def _clean_documents(self) -> List[FileDocument]:
        """
        Cleans the loaded documents by stripping whitespace and removing
        empties and returns a list of cleaned file documents.
        """

        cleaned = []

        for doc in self.documents:
            cleaned_content = doc.content.strip()
            if cleaned_content:
                cleaned.append(FileDocument(
                    content=cleaned_content,
                    metadata=doc.metadata)
                )

        logger.debug(f"Cleaned {len(cleaned)} documents "
                     f"(out of {len(self.documents)}).")

        return cleaned
