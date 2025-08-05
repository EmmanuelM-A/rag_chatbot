"""
Responsible for wrapping the embedding model client to encode text into
vectors.
"""

from langchain_openai import OpenAIEmbeddings
from process_documents import DocumentProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Handles text embedding by processing documents and converting them into
    embedding vectors using a specified embedding model.
    """

    def __init__(self, directory: str, embedding_model_name: str) -> None:
        """
        Initializes the embedder with a document directory.

        Args:
            directory (str): Path to the folder containing raw documents.
        """
        self.document_processor = DocumentProcessor(directory)
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)

    def create_embedded_chunks(self):
        """
        Processes documents (load, clean, chunk) and returns embedded vectors
        with associated metadata.

        Returns:
            tuple:
                - vectors (List[List[float]]): List of embedding vectors.
                - metadata (dict): Mapping from index to (text + metadata).
        """

        logger.debug("Starting full embedding pipeline.")

        # Step 1: Process the documents into chunks
        chunked_documents = self.document_processor.process_documents()

        if not chunked_documents:
            logger.error("No documents were chunked. Aborting embedding.")
            raise ValueError("No documents to embed.")

        logger.debug(f"{len(chunked_documents)} chunks ready for embedding.")

        # Step 2: Extract content
        texts = [doc.content for doc in chunked_documents]

        # Step 3: Embed the text
        vectors = self.embedding_model.embed_documents(texts)

        # Step 4: Map metadata for reference
        metadata = {
            idx: {"text": doc.content, "meta": doc.metadata}
            for idx, doc in enumerate(chunked_documents)
        }

        logger.info("Successfully created document embeddings and metadata.")

        return vectors, metadata
