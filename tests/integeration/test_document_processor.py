"""
Contains all the tests for the DocumentProcessor.
"""

import unittest
from unittest.mock import patch

from src.components.ingestion.document import FileDocument, \
    FileDocumentMetadata
from src.components.ingestion.document_loader import PDFDocumentLoader, \
    MarkdownDocumentLoader, TxTDocumentLoader, DocxDocumentLoader
from src.components.ingestion.document_processor import \
    DefaultDocumentProcessor
from src.utils.exceptions import FileTypeNotSupported

TEST_DIRECTORY = "tests/data/raw_docs"


class TestDocumentProcessor(unittest.TestCase):
    """Test suite for the document processor"""

    def setUp(self):
        self.processor = DefaultDocumentProcessor(TEST_DIRECTORY)


class TestGetLoader(unittest.TestCase):
    """Test suite to test the get_loader functionality"""

    def setUp(self):
        self.processor = DefaultDocumentProcessor(TEST_DIRECTORY)

    def test_should_raise_a_file_not_supported_error_for_invalid_extensions(self):
        with self.assertRaises(FileTypeNotSupported):
            self.processor._get_loader(".xyz")

    def test_should_get_pdf_loader_for_pdf_files(self):
        loader = self.processor._get_loader(".pdf")
        self.assertIsInstance(loader, PDFDocumentLoader)

    def test_should_get_markdown_loader_for_markdown_files(self):
        loader = self.processor._get_loader(".md")
        self.assertIsInstance(loader, MarkdownDocumentLoader)

    def test_should_get_txt_loader_for_txt_files(self):
        loader = self.processor._get_loader(".txt")
        self.assertIsInstance(loader, TxTDocumentLoader)

    def test_should_get_docx_loader_for_docx_files(self):
        loader = self.processor._get_loader(".docx")
        self.assertIsInstance(loader, DocxDocumentLoader)

    def test_should_return_correct_loader_type_not_just_existence(self):
        pdf_loader = self.processor._get_loader(".pdf")
        txt_loader = self.processor._get_loader(".txt")
        self.assertNotEqual(type(pdf_loader), type(txt_loader))

    @patch('src.components.config.settings.settings')
    def test_should_handle_case_insensitive_extensions(self, mock_settings):
        mock_settings.ALLOWED_FILE_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]
        loader = self.processor._get_loader(".PDF")
        self.assertIsNotNone(loader)


class TestLoadDocuments(unittest.TestCase):
    """Test suite to test the load_documents functionality"""

    def setUp(self):
        self.processor = DefaultDocumentProcessor(TEST_DIRECTORY)

    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    @patch('os.path.splitext')
    @patch('os.path.join')
    def test_should_load_documents_from_directory_successfully(
            self, mock_join, mock_splitext, mock_walk, mock_isdir, mock_exists
    ):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_walk.return_value = [("/test", [], ["file1.txt", "file2.pdf"])]
        mock_splitext.side_effect = [("file1", ".txt"), ("file2", ".pdf")]
        mock_join.side_effect = ["/test/file1.txt", "/test/file2.pdf"]

        mock_document = FileDocument(
            content="content",
            metadata=FileDocumentMetadata("test", "txt")
        )

        with patch.object(
                self.processor._get_loader(".txt"),
                'load_data',
                return_value=mock_document
        ):
            with patch.object(
                self.processor._get_loader(".pdf"),
                'load_data',
                    return_value=mock_document
            ):
                self.processor.load_documents()
                self.assertEqual(len(self.processor.documents), 2)

    def test_should_raise_directory_not_found_error(self):
        pass

    def test_should_raise_invalid_directory_error(self):
        pass

    def test_should_skip_unsupported_files(self):
        pass

    def test_should_get_correct_document_loader_for_file_extension(self):
        pass

    def test_should_continue_processing_after_individual_file_errors(self):
        pass

    def test_should_load_multiple_file_types_from_directory(self):
        pass


class TestCleanDocuments:
    """Test suite to test the clean_documents functionality"""

    def test_should_remove_empty_documents_from_list(
            self):  # Not raise error
        pass

    def test_should_strip_leading_and_trailing_whitespace(self):
        pass

    def test_should_preserve_internal_whitespace(self):
        pass

    def test_should_preserve_document_metadata_during_cleaning(self):
        pass


class TestChunkDocuments:
    """Test suite to test the chunk_documents functionality"""

    def test_should_create_chunks_with_specified_size_limits(self):
        pass

    def test_should_apply_chunk_overlap_correctly(self):
        pass

    def test_should_filter_out_empty_chunks(self):
        pass

    def test_should_preserve_metadata_in_all_chunks(self):
        pass

    def test_should_use_recursive_character_text_splitter(self):
        pass


class TestProcessDocuments:
    """Test suite to test the process_documents functionality"""

    def test_should_succeed_in_processing_documents_with_no_errors(self):
        pass

    def test_should_fail_in_processing_documents_with_errors(self):
        pass
