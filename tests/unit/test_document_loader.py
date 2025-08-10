"""
Contains all the unit test for the document loader classes and their
functionality.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock

from src.components.ingestion.document_loader import (
    DocumentLoader,
    TxTDocumentLoader,
    MarkdownDocumentLoader,
    PDFDocumentLoader,
    DocxDocumentLoader
)
from src.utils.exceptions import FileDoesNotExist
from src.components.ingestion.document import FileDocument, \
    FileDocumentMetadata


class TestDocumentLoader(unittest.TestCase):
    """Test suite for the DocumentLoader"""

    def test_should_raise_file_does_not_exist_if_file_path_doesnt_exist(self):
        """Test that FileDoesNotExist is raised when file path doesn't exist."""
        loader = TxTDocumentLoader()
        non_existent_path = "/path/that/does/not/exist.txt"

        with self.assertRaises(FileDoesNotExist):
            loader.load_data(non_existent_path)

    def test_should_raise_file_does_not_exist_if_file_is_not_file(self):
        """Test that FileDoesNotExist is raised when path is not a file (e.g., directory)."""
        loader = TxTDocumentLoader()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileDoesNotExist):
                loader.load_data(temp_dir)

    def test_should_read_data_successfully(self):
        """Test that data is read successfully from a valid file."""
        loader = TxTDocumentLoader()
        test_content = "This is test content"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name

        try:
            result = loader.read_data(temp_path)
            self.assertEqual(result, test_content)
        finally:
            os.unlink(temp_path)

    def test_should_load_data_successfully(self):
        """Test that FileDocument is created successfully from a valid file."""
        loader = TxTDocumentLoader()
        test_content = "This is test content"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name

        try:
            result = loader.load_data(temp_path)

            self.assertIsInstance(result, FileDocument)
            self.assertEqual(result.content, test_content)
            self.assertIsInstance(result.metadata, FileDocumentMetadata)
            self.assertEqual(result.metadata.source, temp_path)
            self.assertEqual(result.metadata.file_extension, '.txt')
            self.assertEqual(result.metadata.filename,
                             os.path.splitext(temp_path)[0].lower())
        finally:
            os.unlink(temp_path)

    def test_should_raise_error_if_load_data_unsuccessful(self):
        """Test that appropriate error is raised when load_data fails."""
        loader = TxTDocumentLoader()

        # Test with non-existent file
        with self.assertRaises(FileDoesNotExist):
            loader.load_data("/non/existent/path.txt")

    # Additional comprehensive tests
    def test_txt_loader_functionality(self):
        """Test TXT loader specific functionality."""
        self._test_text_based_loader_functionality(TxTDocumentLoader, '.txt',
                                                   'Text file content')

    def test_markdown_loader_functionality(self):
        """Test Markdown loader specific functionality."""
        self._test_text_based_loader_functionality(MarkdownDocumentLoader,
                                                   '.md',
                                                   '# Markdown Header\nContent')

    def _test_text_based_loader_functionality(self, loader_class,
                                              file_extension, mock_content):
        """Helper method to test functionality of text-based loaders (TXT and Markdown)."""
        loader = loader_class()

        with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension,
                                         delete=False) as temp_file:
            temp_file.write(mock_content)
            temp_path = temp_file.name

        try:
            # Test read_data
            content = loader.read_data(temp_path)
            self.assertEqual(content, mock_content)

            # Test load_data
            document = loader.load_data(temp_path)
            self.assertIsInstance(document, FileDocument)
            self.assertEqual(document.content, mock_content)
            self.assertEqual(document.metadata.file_extension, file_extension)
            self.assertEqual(document.metadata.source, temp_path)
        finally:
            os.unlink(temp_path)

    def test_pdf_loader_functionality(self):
        """Test PDF loader specific functionality."""
        loader = PDFDocumentLoader()

        # Mock fitz.open and document structure
        with patch('fitz.open') as mock_fitz:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "PDF page content"
            mock_doc.__iter__.return_value = [mock_page]
            mock_doc.__enter__.return_value = mock_doc
            mock_doc.__exit__.return_value = None
            mock_fitz.return_value = mock_doc

            # Test read_data
            content = loader.read_data("test.pdf")
            self.assertEqual(content, "PDF page content")

            # Test load_data with actual file
            with tempfile.NamedTemporaryFile(suffix='.pdf',
                                             delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                document = loader.load_data(temp_path)
                self.assertIsInstance(document, FileDocument)
                self.assertEqual(document.content, "PDF page content")
                self.assertEqual(document.metadata.file_extension, '.pdf')
            finally:
                os.unlink(temp_path)

    def test_docx_loader_functionality(self):
        """Test DOCX loader specific functionality."""
        loader = DocxDocumentLoader()

        # Mock docx.Document
        with patch('docx.Document') as mock_docx:
            mock_doc = MagicMock()
            mock_para1 = MagicMock()
            mock_para1.text = "First paragraph"
            mock_para2 = MagicMock()
            mock_para2.text = "Second paragraph"
            mock_doc.paragraphs = [mock_para1, mock_para2]
            mock_docx.return_value = mock_doc

            # Test read_data
            content = loader.read_data("test.docx")
            self.assertEqual(content, "First paragraph\nSecond paragraph")

            # Test load_data with actual file
            with tempfile.NamedTemporaryFile(suffix='.docx',
                                             delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                document = loader.load_data(temp_path)
                self.assertIsInstance(document, FileDocument)
                self.assertEqual(document.content,
                                 "First paragraph\nSecond paragraph")
                self.assertEqual(document.metadata.file_extension, '.docx')
            finally:
                os.unlink(temp_path)

    def test_unicode_handling_text_loaders(self):
        """Test that text loaders handle Unicode and encoding issues properly."""
        loader = TxTDocumentLoader()

        # Test UTF-8 content
        utf8_content = "Test content with unicode: ñáéíóú 中文 🎉"
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         suffix='.txt',
                                         delete=False) as temp_file:
            temp_file.write(utf8_content)
            temp_path = temp_file.name

        try:
            content = loader.read_data(temp_path)
            self.assertEqual(content, utf8_content)
        finally:
            os.unlink(temp_path)

        # Test fallback to latin-1 encoding
        with patch('builtins.open') as mock_file:
            # First call raises UnicodeDecodeError, second succeeds
            mock_file.side_effect = [
                UnicodeDecodeError('utf-8', b'', 0, 1, 'mock error'),
                mock_open(read_data='latin-1 content').return_value
            ]

            content = loader.read_data("test.txt")
            self.assertEqual(content, 'latin-1 content')

    def test_document_loader_string_representation(self):
        """Test that document loaders have proper string representations."""
        loaders = [
            TxTDocumentLoader(),
            MarkdownDocumentLoader(),
            PDFDocumentLoader(),
            DocxDocumentLoader()
        ]

        expected_names = [
            "TxTDocumentLoader",
            "MarkdownDocumentLoader",
            "PDFDocumentLoader",
            "DocxDocumentLoader"
        ]

        for loader, expected_name in zip(loaders, expected_names):
            self.assertEqual(str(loader), expected_name)

    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that the abstract DocumentLoader class cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            DocumentLoader()

    def test_filename_extraction_and_metadata_creation(self):
        """Test that filenames are extracted correctly and metadata is properly created."""
        loader = TxTDocumentLoader()
        test_content = "Test content"

        # Create file with specific name to test filename extraction
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                         prefix='TestFile_') as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name

        try:
            document = loader.load_data(temp_path)

            expected_filename = os.path.splitext(temp_path)[0].lower()

            self.assertEqual(document.metadata.filename, expected_filename)
            self.assertEqual(document.metadata.file_extension, '.txt')
            self.assertEqual(document.metadata.source, temp_path)
            self.assertIsNone(document.metadata.author)
        finally:
            os.unlink(temp_path)
