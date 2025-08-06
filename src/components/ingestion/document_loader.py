"""
Responsible for loading documents of different formats.
"""
import os
from abc import abstractmethod, ABC

import docx
import fitz

from src.components.config.settings import settings
from src.components.ingestion.document import FileDocument, \
    FileDocumentMetadata


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    This class defines the interface for loading data of various file
    formats.
    """

    @abstractmethod
    def load_data(self, file_path: str) -> FileDocument:
        """
        Load data from the specified file path.

        :param file_path: The file path from which to load data (e.g., file path, URL).
        :return: The FileDocument representation of the file.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def __str__(self):
        """
        String representation of the DocumentLoader.
        :return: A string indicating the type of document loader.
        """

        return f"{self.__class__.__name__}"


class TxTDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading txt files.
    """

    def load_data(self, file_path: str) -> FileDocument:
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()

            filename = os.path.splitext(file_path)[0].lower()

            file_metadata = FileDocumentMetadata(
                filename=filename,
                file_extension=settings.TXT_FILE_EXT,
                author=None,
                source=file_path
            )

            return FileDocument(
                content=content,
                metadata=file_metadata
            )
        except ValueError as e:
            raise e


class MarkdownDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading markdown files.
    """

    def load_data(self, file_path: str) -> FileDocument:
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()

            filename = os.path.splitext(file_path)[0].lower()

            file_metadata = FileDocumentMetadata(
                filename=filename,
                file_extension=settings.MD_FILE_EXT,
                author=None,
                source=file_path
            )

            return FileDocument(
                content=content,
                metadata=file_metadata
            )
        except ValueError as e:
            raise e


class PDFDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading pdf files.
    """

    def load_data(self, file_path: str) -> FileDocument:
        try:
            content = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    content += page.get_text("text")

            filename = os.path.splitext(file_path)[0].lower()

            file_metadata = FileDocumentMetadata(
                filename=filename,
                file_extension=settings.PDF_FILE_EXT,
                author=None,
                source=file_path
            )

            return FileDocument(
                content=content,
                metadata=file_metadata
            )
        except ValueError as e:
            raise e


class DocxDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading docx files.
    """

    def load_data(self, file_path: str) -> FileDocument:
        try:
            doc = docx.Document(str(file_path))
            content = "\n".join(
                [para.text for para in doc.paragraphs]
            )

            filename = os.path.splitext(file_path)[0].lower()

            file_metadata = FileDocumentMetadata(
                filename=filename,
                file_extension=settings.DOCX_FILE_EXT,
                author=None,
                source=file_path
            )

            return FileDocument(
                content=content,
                metadata=file_metadata
            )
        except ValueError as e:
            raise e
