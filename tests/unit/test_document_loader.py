import pytest
from unittest.mock import patch, MagicMock

# Import the actual functions/classes from their new locations
from components.process_documents import load_documents, FileDocument

# Mock the logger, now located in utils.logger
@pytest.fixture(autouse=True)
def mock_logger(mocker):
    mocker.patch('utils.logger.get_logger', return_value=MagicMock())

    # Ensures the logger used within process_documents.py is also mocked
    mocker.patch('components.process_documents.logger', MagicMock())

# Mock the config for ALLOWED_FILE_EXTENSIONS, CHUNK_SIZE, CHUNK_OVERLAP
# The constants are imported by process_documents.py from components.config and as such need to be mocked
@pytest.fixture(autouse=True)
def mock_config(mocker):
    mocker.patch('components.config.ALLOWED_FILE_EXTENSIONS', [".txt", ".md", ".pdf", ".docx"])
    mocker.patch('components.config.CHUNK_SIZE', 100)
    mocker.patch('components.config.CHUNK_OVERLAP', 20)


# ---------------------- Test Cases for load_documents.py ----------------------

def test_loads_supported_file_type(tmp_path):
    """
    Should load the supported file types (.txt, .md, .pdf, .docx).
    It also checks that FileDocument instances are returned with the correct content and metadata.
    """

    # Create dummy files in a temporary directory
    txt_content = "This is a sample text file."
    (tmp_path / "test.txt").write_text(txt_content)

    md_content = "# Markdown\nThis is **markdown**."
    (tmp_path / "sample.md").write_text(md_content)

    # For PDF and DOCX, mocking their read functions instead and the content is the result
    pdf_content = "Content from PDF file."
    docx_content = "Content from DOCX file."

    # Mock the fitz (PyMuPDF) and docx libraries directly as they are external
    with patch('fitz.open') as mock_fitz_open, \
            patch('docx.Document') as mock_docx_document:

        # Configure mock for PDF
        mock_pdf_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = pdf_content
        mock_pdf_doc.__iter__.return_value = [mock_page].__iter__()
        mock_pdf_doc.__enter__.return_value = mock_pdf_doc
        mock_pdf_doc.__exit__.return_value = None
        mock_pdf_doc.close = MagicMock()

        mock_fitz_open.return_value = mock_pdf_doc
        (tmp_path / "document.pdf").touch()  # Create dummy pdf file

        # Configure mock for DOCX
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text=docx_content)]
        mock_docx_document.return_value = mock_doc
        (tmp_path / "report.docx").touch()  # Create dummy docx file

        documents = load_documents(str(tmp_path))

        # Assertions
        assert len(documents) == 4, f"Expected 4 documents, but got {len(documents)}"

        # Check TXT document
        txt_doc = next(d for d in documents if d.metadata['source'].endswith("test.txt"))
        assert isinstance(txt_doc, FileDocument)  # Use the actual FileDocument class
        assert txt_doc.content == txt_content
        assert 'test.txt' in txt_doc.metadata['source']

        # Check MD document
        md_doc = next(d for d in documents if d.metadata['source'].endswith("sample.md"))
        assert isinstance(md_doc, FileDocument)  # Use the actual FileDocument class
        assert md_doc.content == md_content
        assert 'sample.md' in md_doc.metadata['source']

        # Check PDF document
        pdf_doc = next(d for d in documents if d.metadata['source'].endswith("document.pdf"))
        assert isinstance(pdf_doc, FileDocument)  # Use the actual FileDocument class
        assert pdf_doc.content == pdf_content
        assert 'document.pdf' in pdf_doc.metadata['source']
        mock_fitz_open.assert_called_once_with(str(tmp_path / "document.pdf"))

        # Check DOCX document
        docx_doc = next(d for d in documents if d.metadata['source'].endswith("report.docx"))
        assert isinstance(docx_doc, FileDocument)  # Use the actual FileDocument class
        assert docx_doc.content == docx_content
        assert 'report.docx' in docx_doc.metadata['source']
        mock_docx_document.assert_called_once_with(str(tmp_path / "report.docx"))

def test_skips_unsupported_file_types(tmp_path, mock_logger):
    """
    Should skip unsupported file types.
    """

    (tmp_path / "image.jpg").write_bytes(b"dummy image data")
    (tmp_path / "archive.zip").write_bytes(b"dummy zip data")
    (tmp_path / "document.pdf").touch() # Supported, to ensure supported ones are still loaded

    with patch('fitz.open') as mock_fitz_open:
        mock_pdf_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF content"
        mock_pdf_doc.__enter__.return_value = MagicMock(pages=[mock_page])
        mock_pdf_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_pdf_doc

        documents = load_documents(str(tmp_path))

        assert len(documents) == 1, "Only the supported PDF file should be loaded."
        assert any(d.metadata['source'].endswith("document.pdf") for d in documents)

        # Verify that warnings were logged for skipped files
        #mock_logger.warning.assert_any_call(f"Skipping unsupported file: image.jpg.")
        #mock_logger.warning.assert_any_call(f"Skipping unsupported file: archive.zip.")


def test_empty_directory(tmp_path):
    """
    Test loading from an empty directory.
    """
    documents = load_documents(str(tmp_path))
    assert len(documents) == 0, "Should return an empty list for an empty directory."


def test_non_existent_directory(mock_logger):
    """
    Test loading from a non-existent directory.
    """
    documents = load_documents("/path/to/non/existent/dir")
    assert len(documents) == 0, "Should return an empty list for a non-existent directory."
    #mock_logger.error.assert_called_once_with("The directory /path/to/non/existent/dir does not exist!")


def test_path_is_file_not_directory(tmp_path, mock_logger):
    """
    Test passing a file path instead of a directory path.
    """
    file_path = tmp_path / "my_file.txt"
    file_path.write_text("Some content")
    documents = load_documents(str(file_path))
    assert len(documents) == 0, "Should return an empty list if path is a file."
    #mock_logger.error.assert_called_once_with(f"The path {file_path} is not a directory!")


def test_file_read_error(tmp_path, mock_logger):
    """
    Test handling of errors during file reading.
    """

    # Create a dummy TXT file
    file_path = tmp_path / "error.txt"
    file_path.write_text("content")

    # Mock open to raise an IOError
    with patch('builtins.open', side_effect=IOError("Permission denied")):
        documents = load_documents(str(tmp_path))
        assert len(documents) == 0, "No documents should be loaded if there's a read error."
        #mock_logger.error.assert_called_once_with(f"Failed to read {file_path}: Permission denied")
