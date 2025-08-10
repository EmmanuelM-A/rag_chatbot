"""
Responsible for loading documents of different formats.
"""
import os
from abc import abstractmethod, ABC
from pathlib import Path

import docx
import fitz

from src.components.ingestion.document import FileDocument, \
    FileDocumentMetadata
from src.utils.exceptions import FileDoesNotExist


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    This class defines the interface for loading data of various file
    formats.
    """

    @abstractmethod
    def read_data(self, file_path: str) -> str:
        """
        Reads data from the specified file path.

        :param file_path: The file path to read from.
        :return: The contents of the file.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def load_data(self, file_path: str) -> FileDocument:
        """
        Load data from the specified file path and convert it to a FileDocument.

        :param file_path: The file path from which to load data (e.g., file path, URL).
        :return: The FileDocument representation of the file.
        """

        try:
            path = Path(file_path)

            if not path.exists():
                raise FileDoesNotExist(
                    f"The file '{file_path}' does not exist or cannot be found."
                )

            if not path.is_file():
                raise FileDoesNotExist(f"Path is not a file: {file_path}")

            content = self.read_data(file_path)

            filename = os.path.splitext(file_path)[0].lower()

            file_extension = os.path.splitext(file_path)[1].lower()

            file_metadata = FileDocumentMetadata(
                filename=filename,
                file_extension=file_extension,
                author=None,
                source=file_path
            )

            return FileDocument(
                content=content,
                metadata=file_metadata
            )
        except OSError as e:
            raise e

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

    def read_data(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
            raise e

        return content


class MarkdownDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading markdown files.
    """

    def read_data(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()

        return content


class PDFDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading pdf files.
    """

    def read_data(self, file_path: str) -> str:
        content = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                content += page.get_text("text")

        return content


class DocxDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading docx files.
    """

    def read_data(self, file_path: str) -> str:
        doc = docx.Document(str(file_path))
        content = "\n".join(
            [para.text for para in doc.paragraphs]
        )

        return content
