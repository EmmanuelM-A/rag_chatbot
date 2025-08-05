"""
Responsible for loading documents of different formats.
"""

from abc import abstractmethod, ABC

from src.components.ingestion.document import FileDocument


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    This class defines the interface for loading data of various file
    formats.
    """

    @abstractmethod
    def load_data(self, source: str) -> FileDocument:
        """
        Load data from the specified source.

        :param source: The source from which to load data (e.g.,
        file path, URL).
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

    def load_data(self, source: str) -> FileDocument:
        pass


class MarkdownDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading markdown files.
    """

    def load_data(self, source: str) -> FileDocument:
        pass


class PDFDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading pdf files.
    """

    def load_data(self, source: str) -> FileDocument:
        pass


class DocxDocumentLoader(DocumentLoader):
    """
    Document loader responsible for loading docx files.
    """

    def load_data(self, source: str) -> FileDocument:
        pass
