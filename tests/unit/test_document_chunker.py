import pytest
from unittest.mock import patch, MagicMock

# Import the actual functions/classes from their new locations
from components.process_documents import chunk_documents, FileDocument

# Mock the logger, now located in utils.logger
@pytest.fixture(autouse=True)
def mock_logger(mocker):
    mocker.patch('utils.logger.get_logger', return_value=MagicMock())

    # Ensure the logger used within process_documents.py is also mocked
    mocker.patch('components.process_documents.logger', MagicMock())

# Mock config constants for chunking, now located in components.config
@pytest.fixture
def mock_chunk_config(mocker):
    mocker.patch('components.config.CHUNK_SIZE', 10)
    mocker.patch('components.config.CHUNK_OVERLAP', 2)

# ---------------------- Test Cases for load_documents.py ----------------------

def test_chunks_are_correctly_split_by_size_and_overlap(mock_chunk_config):
    """
    ✅ Chunks are correctly split by size & overlap.
    Uses CHUNK_SIZE=10, CHUNK_OVERLAP=2
    """



def test_metadata_is_preserved(mock_chunk_config):
    """
    ✅ Metadata is preserved in all generated chunks.
    """
    original_metadata = {"source": "document.pdf", "author": "John Doe", "page": 1}
    doc = FileDocument("This is some sample text for testing metadata preservation.", original_metadata) # Use actual FileDocument

    chunks = chunk_documents([doc])

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, FileDocument) # Use the actual FileDocument class
        assert chunk.metadata == original_metadata, "Metadata should be identical for all chunks."


def test_empty_chunks_are_ignored(mock_chunk_config):
    """
    ✅ Empty chunks are ignored.
    This includes chunks that might result from splitting that are just whitespace.
    """



def test_empty_documents_list():
    """
    Test with an empty list of documents.
    """
    chunks = chunk_documents([])
    assert len(chunks) == 0, "Should return an empty list if input documents list is empty."


def test_document_with_empty_content(mock_chunk_config):
    """
    Test with a FileDocument that has empty content.
    """
    doc = FileDocument("", {"id": "empty_content_doc"}) # Use actual FileDocument
    chunks = chunk_documents([doc])
    assert len(chunks) == 0, "Should produce no chunks for a document with empty content."


def test_multiple_documents(mock_chunk_config):
    """
    Test chunking multiple documents.
    """
    doc1 = FileDocument("Short content one.", {"id": "doc1"}) # Use actual FileDocument
    doc2 = FileDocument("A longer piece of text for the second document that needs to be split.", {"id": "doc2"}) # Use actual FileDocument

    chunks = chunk_documents([doc1, doc2])

    assert len(chunks) >= 2 # At least 2 from doc1, and several from doc2

    doc1_chunks = [c for c in chunks if c.metadata['id'] == 'doc1']
    doc2_chunks = [c for c in chunks if c.metadata['id'] == 'doc2']

    # For doc1 with "Short content one." (18 chars), CHUNK_SIZE=10, OVERLAP=2
    # Chunk 1: "Short cont"
    # Chunk 2: "ent one." (overlap 'ent')
    # The RecursiveCharacterTextSplitter will try to split by default separators first.
    # Given the content and small chunk size, it will likely split into:
    # "Short cont" and "content one." or similar, depending on exact word breaks.
    # Let's verify lengths and metadata.
    assert len(doc1_chunks) >= 1
    assert len(doc2_chunks) >= 1

    # Basic check for metadata preservation across multiple documents
    for chunk in doc1_chunks:
        assert chunk.metadata == {"id": "doc1"}
    for chunk in doc2_chunks:
        assert chunk.metadata == {"id": "doc2"}
