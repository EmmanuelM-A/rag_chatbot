"""
Responsible for wrapping the embedding model client to encode text into
vectors.
"""
from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings
from src.components.config.logger import get_logger
from src.components.retrieval.embedding_cache import EmbeddingCache

logger = get_logger(__name__)


class Embedder:
    """
    Handles text embedding by processing documents and converting them into
    embedding vectors using a specified embedding model with caching support.
    """

    def __init__(self, embedding_model_name: str) -> None:
        """
        Initializes the embedder with a document directory.

        Args:
            embedding_model_name (str): The name of the embedding model to be used.
        """

        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.cache = EmbeddingCache()

    def create_embedded_chunks(self, chunked_documents):
        """
        Embeds the document chunks and returns embedded vectors with the
        associated metadata.

        Args:
            chunked_documents (List[FileDocument]): The processed and
            chunked documents to be embedded.

        Returns:
            tuple:
                - vectors (List[List[float]]): List of embedding vectors.
                - metadata (dict): Mapping from index to (text + metadata).
        """

        if not chunked_documents:
            logger.error("No documents were chunked. Aborting embedding.")
            raise ValueError("No documents to embed.")

        logger.debug(f"{len(chunked_documents)} chunks ready for embedding.")

        # Step 1: Extract content and sources for caching
        texts = [doc.content for doc in chunked_documents]
        sources = [doc.metadata.source for doc in chunked_documents]

        # Step 2: Try to get embeddings from cache
        cached_embeddings = []
        texts_to_embed = []
        cache_map = {}  # Maps original index to position in texts_to_embed

        for i, (text, source) in enumerate(zip(texts, sources)):
            cached_embedding = self.cache.get_embedding(text, source)
            if cached_embedding is not None:
                cached_embeddings.append((i, cached_embedding))
            else:
                cache_map[i] = len(texts_to_embed)
                texts_to_embed.append((text, source))

        logger.info(f"Cache hits: {len(cached_embeddings)}/{len(texts)} "
                    f"({len(cached_embeddings) / len(texts) * 100:.1f}%)")

        # Step 3: Embed the remaining texts
        new_embeddings = []
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} new documents...")

            # Extract just the text content for embedding
            texts_only = [text for text, source in texts_to_embed]
            new_embeddings = self.embedding_model.embed_documents(texts_only)

            # Store new embeddings in cache
            for (text, source), embedding in zip(texts_to_embed,
                                                 new_embeddings):
                self.cache.store_embedding(text, embedding, source)

        # Step 4: Combine cached and new embeddings in correct order
        vectors = [None] * len(texts)

        # Place cached embeddings
        for original_idx, embedding in cached_embeddings:
            vectors[original_idx] = embedding

        # Place new embeddings
        for original_idx, texts_to_embed_idx in cache_map.items():
            vectors[original_idx] = new_embeddings[texts_to_embed_idx]

        # Step 5: Create metadata mapping (same as your original)
        metadata = {
            idx: {"text": doc.content, "meta": doc.metadata}
            for idx, doc in enumerate(chunked_documents)
        }

        logger.info("Successfully created document embeddings and metadata.")

        return vectors, metadata

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.cache.get_cache_stats()

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear_cache()
