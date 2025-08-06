"""
Responsible for wrapping the embedding model client to encode text into
vectors.
"""

from langchain_openai import OpenAIEmbeddings
from src.components.config.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Handles text embedding by processing documents and converting them into
    embedding vectors using a specified embedding model.
    """

    def __init__(self, embedding_model_name: str) -> None:
        """
        Initializes the embedder with a document directory.

        Args:
            embedding_model_name (str): The name of the embedding model to be used.
        """

        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)

    def create_embedded_chunks(self, chunked_documents):
        """
        Embeds the document chunks and returns embedded vectors with the
        associated metadata.

        Args:
            chunked_documents (List[FileDocument]): The processed and chunked documents to be embedded.

        Returns:
            tuple:
                - vectors (List[List[float]]): List of embedding vectors.
                - metadata (dict): Mapping from index to (text + metadata).
        """

        logger.debug("Starting full embedding pipeline.")

        if not chunked_documents:
            logger.error("No documents were chunked. Aborting embedding.")
            raise ValueError("No documents to embed.")

        logger.debug(f"{len(chunked_documents)} chunks ready for embedding.")

        # Step 1: Extract content
        texts = [doc.content for doc in chunked_documents]

        # Step 2: Embed the text
        vectors = self.embedding_model.embed_documents(texts)

        # Step 3: Map metadata for reference
        metadata = {
            idx: {"text": doc.content, "meta": doc.metadata}
            for idx, doc in enumerate(chunked_documents)
        }

        logger.info("Successfully created document embeddings and metadata.")

        return vectors, metadata
