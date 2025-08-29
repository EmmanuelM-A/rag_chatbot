"""
Handles document ingestion, cleaning, splitting, and preparation of source
documents before embedding and storage.
"""

import os
from abc import ABC
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.components.config.settings import settings
from src.components.ingestion.document import FileDocument
from src.components.ingestion.document_loader import PDFDocumentLoader, \
    MarkdownDocumentLoader, TxTDocumentLoader, DocxDocumentLoader, \
    DocumentLoader
from src.components.config.logger import get_logger
from src.utils.exceptions import InvalidDirectoryError, FileSystemError, \
    DocumentLoadError, FileTypeNotSupported, FileDoesNotExist

logger = get_logger(__name__)


class DocumentProcessor(ABC):
    """
    Prepares raw documents for indexing into the vector store.
    """

    def __init__(self) -> None:
        """
        Initializes the DocumentProcessor instance.
        """

        self.path_to_directory = settings.app.RAW_DOCS_DIRECTORY
        self.documents = []
        self.documents_skipped = []
        self.document_loader_mappings = {
            settings.app.PDF_FILE_EXT: PDFDocumentLoader(),
            settings.app.MD_FILE_EXT: MarkdownDocumentLoader(),
            settings.app.TXT_FILE_EXT: TxTDocumentLoader(),
            settings.app.DOCX_FILE_EXT: DocxDocumentLoader()
        }

    def __load_documents(self) -> None:
        """
        Loads supported documents from the directory into FileDocument objects
        and stores it in the self.documents list.

        Raises:
            InvalidDirectoryError:

        """

        if not os.path.exists(self.path_to_directory):
            raise InvalidDirectoryError(self.path_to_directory)

        if not os.path.isdir(self.path_to_directory):
            raise InvalidDirectoryError(self.path_to_directory)

        logger.debug(f"Loading documents from: {self.path_to_directory}")

        documents = []

        for root, _, files in os.walk(self.path_to_directory):
            for file in files:

                file_extension = os.path.splitext(file)[1].lower()

                if file_extension not in settings.app.ALLOWED_FILE_EXTENSIONS:
                    logger.warning(f"Skipping unsupported file: {file}")
                    self.documents_skipped.append(file)
                    continue

                file_path = os.path.join(root, file)

                try:
                    file_loader = self.__get_loader(file_extension)

                    document = file_loader.load_data(file_path)

                    documents.append(document)
                except (FileDoesNotExist, DocumentLoadError) as e:
                    self.documents_skipped.append(file)
                    logger.error(e.as_log())

        logger.debug(
            "Report: %d documents loaded and %d documents skipped.",
            len(documents), len(self.documents_skipped)
        )

        self.documents = documents

    def __get_loader(self, file_extension) -> DocumentLoader | None:
        """
        Gets the correct document loader based on the document's file
        extension.
        """

        return self.document_loader_mappings.get(file_extension.lower(), None)

    @staticmethod
    def __chunk_documents(documents):
        """
        Splits documents into smaller chunks using a recursive character
        splitter and returns a list of chunked documents.
        """

        logger.debug("Chunking documents...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.vector.CHUNK_SIZE,
            chunk_overlap=settings.vector.CHUNK_OVERLAP
        )

        chunks = []
        for doc in documents:
            for chunk in splitter.split_text(doc.content):
                if chunk.strip():
                    chunks.append(
                        FileDocument(content=chunk, metadata=doc.metadata))

        logger.debug(f"Chunking produced {len(chunks)} chunks from "
                     f"{len(documents)} documents.")

        return chunks

    @staticmethod
    def __clean_documents(documents) -> List[FileDocument]:
        """
        Cleans the loaded documents by stripping whitespace and removing
        empties and returns a list of cleaned file documents.
        """

        cleaned = []

        for doc in documents:
            cleaned_content = doc.content.strip()
            if cleaned_content:
                cleaned.append(FileDocument(
                    content=cleaned_content,
                    metadata=doc.metadata)
                )

        logger.debug(f"Cleaned {len(cleaned)} documents "
                     f"(out of {len(documents)}).")

        return cleaned

    def process_documents(self):
        """
        Full processing pipeline: load, clean, chunk.
        """

        self.__load_documents()

        cleaned_documents = self.__clean_documents(self.documents)

        chunked_documents = self.__chunk_documents(cleaned_documents)

        if not chunked_documents:
            logger.error("No documents have been processed!")
            return None

        logger.debug(
            "Documents have been loaded, processed and chunked successfully!"
        )

        return chunked_documents
