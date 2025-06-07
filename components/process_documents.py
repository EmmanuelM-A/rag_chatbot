import os

from components.config import ALLOWED_FILE_EXTENSIONS, CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

import docx
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger("process_documents_logger")


class FileDocument:
    def __init__(self, content, metadata):
        """
        Represents a document with its content and metadata.

        :param content: The main content of the document (e.g., text from a file).
        :param metadata: A dictionary or object containing metadata about the document (e.g., filename, file extension,
        author, etc.).
        """
        self.content = content
        self.metadata = metadata


def load_documents(path_to_directory):
    """
    Reads all files in the directory that match the allowed extensions and converts them into Document objects.

    :param path_to_directory: The path to the folder containing the document files.

    :returns list[FileDocument]: A list of loaded FileDocument objects
    """

    if not os.path.exists(path_to_directory):
        logger.error(f"The directory {path_to_directory} does not exist!")
        return []

    if not os.path.isdir(path_to_directory):
        logger.error(f"The path {path_to_directory} is not a directory!")
        return []

    logger.debug(f"Loading documents from: {path_to_directory}.")

    documents = []

    # Iterate through all files in the directory and subdirectories
    for root, _, files in os.walk(path_to_directory):
        for file in files:
            # Get the file extension (in lowercase)
            ext = os.path.splitext(file)[1].lower()

            if ext not in ALLOWED_FILE_EXTENSIONS:
                logger.warning(f"Skipping unsupported file: {file}.")
                continue

            file_path = os.path.join(root, file)

            logger.debug(f"Reading the file: {file_path}")

            try:
                # Handle plain text and markdown files
                if ext in [".txt", ".md"]:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()

                # Handles PDF
                elif ext == ".pdf":
                    content = ""
                    with fitz.open(file_path) as doc:
                        for page in doc:
                            content += page.get_text("text")

                    doc.close()

                # Handles DOCX
                elif ext == ".docx":
                    doc = docx.Document(str(file_path))
                    # Combine all paragraph texts into a single string
                    content = "\n".join([para.text for para in doc.paragraphs])

                else:
                    logger.warning(f"Unknown extension: {ext}.")
                    continue  # Skip unknown file types

                # Add the document to the list with metadata
                documents.append(FileDocument(content, {'source': file_path}))

            except Exception as e:
                # Log an error if reading fails
                logger.error(f"Failed to read {file_path}: {e}")

    logger.info("Document ingestion completed successfully.")

    return documents


def chunk_documents(documents):
    """
    Splits a list of FileDocument objects into smaller, manageable chunks using a RecursiveCharacterTextSplitter.

    This method iterates through each FileDocument, extracts its content, and then uses a
    RecursiveCharacterTextSplitter to break down the content into chunks of a specified size
    with a defined overlap. Empty or whitespace-only chunks are filtered out.
    Each generated chunk is then wrapped back into a FileDocument object, inheriting the
    original document's metadata.

    param:
        documents (list[FileDocument]): A list of FileDocument objects, each containing
                                       'content' (str) and 'metadata' (dict).

    Returns:
        list[FileDocument]: A new list of FileDocument objects, where each object
                            represents a chunk of the original documents' content.
    """

    logger.debug(f"Chunking the documents: {documents}.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = []

    for doc in documents:
        for chunk in splitter.split_text(doc.content):
            if chunk.strip():  # skip empty or whitespace-only chunks
                chunks.append(FileDocument(chunk, doc.metadata))

    logger.info(f"Chunking completed for the documents: {documents}")

    return chunks
