"""
Contains all the tests for the DocumentProcessor.
"""

import unittest

import pytest

from src.components.ingestion.document_processor import \
    DefaultDocumentProcessor
from src.utils.exceptions import FileTypeNotSupported

TEST_DIRECTORY = "/test/documents"


class TestDocumentProcessor(unittest.TestCase):
    """Test suite for the document processor"""

    def setUp(self):
        self.processor = DefaultDocumentProcessor(TEST_DIRECTORY)

    class TestGetLoader:
        """Test suite to test the get_loader functionality"""

        def setUp(self):
            self.processor = DefaultDocumentProcessor(TEST_DIRECTORY)

        def test_should_raise_a_file_not_supported_error_for_invalid_extensions(
                self):
            with pytest.raises(FileTypeNotSupported):
                self.processor.__get_loader()

        def test_should_get_pdf_loader_for_pdf_files(self):
            pass

        def test_should_get_markdown_loader_for_markdown_files(self):
            pass

        def test_should_get_txt_loader_for_txt_files(self):
            pass

        def test_should_get_docx_loader_for_docx_files(self):
            pass

        def test_should_return_correct_loader_type_not_just_existence(self):
            pass

        def test_should_handle_case_insensitive_extensions(self):
            pass

    class TestLoadDocuments:
        """Test suite to test the load_documents functionality"""

        def test_should_load_documents_from_directory_successfully(self):
            pass

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
