"""
Contains all the tests for the DocumentProcessor.
"""

import unittest
from unittest.mock import patch, Mock

from src.components.ingestion.document import FileDocument, \
    FileDocumentMetadata
from src.components.ingestion.document_loader import PDFDocumentLoader, \
    MarkdownDocumentLoader, TxTDocumentLoader, DocxDocumentLoader
from src.components.ingestion.document_processor import \
    DocumentProcessor
from src.utils.exceptions import FileTypeNotSupported, DirectoryNotFoundError, \
    InvalidDirectoryError

TEST_DIRECTORY = "tests/data/raw_docs"


class TestDocumentProcessor(unittest.TestCase):
    """Test suite for the document processor"""

    def setUp(self):
        self.processor = DocumentProcessor(TEST_DIRECTORY)


class TestGetLoader(unittest.TestCase):
    """Test suite to test the get_loader functionality"""

    def setUp(self):
        self.processor = DocumentProcessor(TEST_DIRECTORY)

    def test_should_raise_file_not_supported_error_for_invalid_extensions(self):
        """Should fail if the file type is not supported"""
        with self.assertRaises(FileTypeNotSupported):
            self.processor._get_loader(".xyz")

    def test_should_get_pdf_loader_for_pdf_files(self):
        """Should return a pdf loader if a pdf file"""
        loader = self.processor._get_loader(".pdf")
        self.assertIsInstance(loader, PDFDocumentLoader)

    def test_should_get_markdown_loader_for_markdown_files(self):
        """Should return a markdown loader if a markdown file """
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
        self.processor = DocumentProcessor(TEST_DIRECTORY)

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

    @patch('os.path.exists')
    def test_should_raise_directory_not_found_error(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(DirectoryNotFoundError):
            self.processor.load_documents()

    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_should_raise_invalid_directory_error(self, mock_isdir,
                                                  mock_exists):
        mock_exists.return_value = True
        mock_isdir.return_value = False
        with self.assertRaises(InvalidDirectoryError):
            self.processor.load_documents()

    @patch('src.components.ingestion.document_processor.logger')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    @patch('os.path.splitext')
    @patch('src.components.config.settings.settings')
    def test_should_skip_unsupported_files(self, mock_settings, mock_splitext,
                                           mock_walk, mock_isdir, mock_exists,
                                           mock_logger):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_walk.return_value = [(TEST_DIRECTORY, [], ["IT CV.docx", "file2.xyz"])]
        mock_splitext.side_effect = [("IT CV", ".docx"), ("file2", ".xyz")]
        mock_settings.ALLOWED_FILE_EXTENSIONS = [".txt", ".pdf", ".md",
                                                 ".docx"]

        self.processor.load_documents()

        # Verify the warning was logged
        mock_logger.warning.assert_called_with(
            "Skipping unsupported file: file2.xyz")
        self.assertEqual(len(self.processor.documents), 0)

    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    @patch('os.path.splitext')
    @patch('os.path.join')
    def test_should_get_correct_document_loader_for_file_extension(
            self, mock_join, mock_splitext, mock_walk, mock_isdir, mock_exists
    ):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_walk.return_value = [("/test", [], ["file1.pdf"])]
        mock_splitext.return_value = ("file1", ".pdf")
        mock_join.return_value = "/test/file1.pdf"

        with patch.object(
                self.processor,
        '_get_loader'
        ) as mock_get_loader:
            mock_loader = Mock()
            mock_loader.load_data.return_value = FileDocument(
                "content",
                FileDocumentMetadata("test", ".pdf")
            )
            mock_get_loader.return_value = mock_loader

            self.processor.load_documents()
            mock_get_loader.assert_called_with(".pdf")

    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    @patch('os.path.splitext')
    @patch('os.path.join')
    def test_should_load_multiple_file_types_from_directory(
            self, mock_join, mock_splitext, mock_walk, mock_isdir, mock_exists
    ):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_walk.return_value = [
            ("/test", [], ["file1.txt", "file2.pdf", "file3.md"])]
        mock_splitext.side_effect = [("file1", ".txt"), ("file2", ".pdf"),
                                     ("file3", ".md")]
        mock_join.side_effect = ["/test/file1.txt", "/test/file2.pdf",
                                 "/test/file3.md"]

        mock_document = FileDocument("content",
                                     FileDocumentMetadata("test", ".txt"))

        with patch.object(
            self.processor,
            '_get_loader'
        ) as mock_get_loader:
            mock_loader = Mock()
            mock_loader.load_data.return_value = mock_document
            mock_get_loader.return_value = mock_loader

            self.processor.load_documents()
            self.assertEqual(len(self.processor.documents), 3)


class TestCleanDocuments(unittest.TestCase):
    """Test suite to test the clean_documents functionality"""

    def setUp(self):
        self.processor = DocumentProcessor(TEST_DIRECTORY)

    def test_should_remove_empty_documents_from_list(self):
        self.processor.documents = [
            FileDocument("  ", FileDocumentMetadata("empty", ".txt")),
            FileDocument("content", FileDocumentMetadata("valid", ".txt"))
        ]
        cleaned = self.processor._clean_documents()
        self.assertEqual(len(cleaned), 1)

    def test_should_strip_leading_and_trailing_whitespace(self):
        self.processor.documents = [
            FileDocument("  content  ", FileDocumentMetadata("test", ".txt"))
        ]
        cleaned = self.processor._clean_documents()
        self.assertEqual(cleaned[0].content, "content")

    def test_should_preserve_internal_whitespace(self):
        content_with_spaces = "word1 word2  word3"
        self.processor.documents = [
            FileDocument(f"  {content_with_spaces}  ",
                         FileDocumentMetadata("test", ".txt"))
        ]
        cleaned = self.processor._clean_documents()
        self.assertEqual(cleaned[0].content, content_with_spaces)

    def test_should_preserve_document_metadata_during_cleaning(self):
        metadata = FileDocumentMetadata("test", ".txt", author="Test Author")
        self.processor.documents = [
            FileDocument("  content  ", metadata)
        ]
        cleaned = self.processor._clean_documents()
        self.assertEqual(cleaned[0].metadata, metadata)


class TestChunkDocuments(unittest.TestCase):
    """Test suite to test the chunk_documents functionality"""

    def setUp(self):
        self.processor = DocumentProcessor(TEST_DIRECTORY)

    @patch('src.components.ingestion.document_processor.RecursiveCharacterTextSplitter')
    @patch('src.components.config.settings.settings')
    def test_should_create_chunks_with_specified_size_limits(
        self, mock_settings, mock_splitter_class
    ): # FIXME
        mock_settings.CHUNK_SIZE = 100
        mock_settings.CHUNK_OVERLAP = 20

        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        mock_splitter_class.return_value = mock_splitter

        self.processor.documents = [
            FileDocument("long content", FileDocumentMetadata("test", ".txt"))]
        chunks = self.processor._chunk_documents([])

        mock_splitter_class.assert_called_with(chunk_size=100,
                                               chunk_overlap=20)

    @patch('src.components.ingestion.document_processor.RecursiveCharacterTextSplitter')
    def test_should_apply_chunk_overlap_correctly(self, mock_splitter_class):
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        mock_splitter_class.return_value = mock_splitter

        self.processor.documents = [
            FileDocument("content", FileDocumentMetadata("test", ".txt"))]
        self.processor._chunk_documents([])

        mock_splitter_class.assert_called()

    @patch('src.components.ingestion.document_processor.RecursiveCharacterTextSplitter')
    def test_should_filter_out_empty_chunks(self, mock_splitter_class):
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk1", "  ", "chunk2", ""]
        mock_splitter_class.return_value = mock_splitter

        self.processor.documents = [
            FileDocument("content", FileDocumentMetadata("test", ".txt"))]
        chunks = self.processor._chunk_documents([])

        self.assertEqual(len(chunks), 2)

    @patch('src.components.ingestion.document_processor.RecursiveCharacterTextSplitter')
    def test_should_preserve_metadata_in_all_chunks(self, mock_splitter_class):
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        mock_splitter_class.return_value = mock_splitter

        metadata = FileDocumentMetadata("test", ".txt", author="Test Author")
        self.processor.documents = [FileDocument("content", metadata)]
        chunks = self.processor._chunk_documents([])

        for chunk in chunks:
            self.assertEqual(chunk.metadata, metadata)

    @patch('src.components.ingestion.document_processor.RecursiveCharacterTextSplitter')
    def test_should_use_recursive_character_text_splitter(
        self, mock_splitter_class
    ):
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk1"]
        mock_splitter_class.return_value = mock_splitter

        self.processor.documents = [
            FileDocument("content", FileDocumentMetadata("test", ".txt"))]
        self.processor._chunk_documents([])

        mock_splitter_class.assert_called()


class TestProcessDocuments(unittest.TestCase):
    """Test suite to test the process_documents functionality"""

    def setUp(self):
        self.processor = DocumentProcessor(TEST_DIRECTORY)

    @patch.object(DocumentProcessor, 'load_documents')
    @patch.object(DocumentProcessor, '_clean_documents')
    @patch.object(DocumentProcessor, '_chunk_documents')
    def test_should_succeed_in_processing_documents_with_no_errors(
        self, mock_chunk, mock_clean, mock_load
    ):
        mock_load.return_value = None
        mock_clean.return_value = []
        mock_chunk.return_value = []

        result = self.processor.process_documents()

        mock_load.assert_called_once()
        mock_clean.assert_called_once()
        mock_chunk.assert_called_once()
        self.assertEqual(result, [])

    @patch.object(DocumentProcessor, 'load_documents')
    def test_should_fail_in_processing_documents_with_errors(self, mock_load):
        mock_load.side_effect = DirectoryNotFoundError("Directory not found")

        with self.assertRaises(DirectoryNotFoundError):
            self.processor.process_documents()


if __name__ == '__main__':
    unittest.main()
