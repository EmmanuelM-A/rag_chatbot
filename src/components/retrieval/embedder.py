"""
Responsible for wrapping the embedding model client to encode text into
vectors.
"""

from typing import Dict, Any, List, Tuple

from langchain_openai import OpenAIEmbeddings
from src.components.config.logger import get_logger
from src.components.config.settings import settings
from src.components.retrieval.embedding_cache import EmbedderCache
from src.utils.exceptions import EmptyDocumentError, EmbeddingError

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

        try:
            self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
            self.cache = EmbedderCache()
            logger.debug(
                f"Initialized embedder with the model: {embedding_model_name}")
        except Exception as e:
            raise EmbeddingError(
                operation="initialization",
                reason=f"Failed to initialize embedding model: {e}"
            ) from e

    def embed_documents(
            self,
            processed_documents
    ) -> Tuple[List[List[float]], Dict[int, Dict]]:
        """
        Embeds the document chunks and returns embedded vectors with the
        associated metadata.

        Args:
            processed_documents: The processed and chunked documents to be embedded.

        Returns:
            tuple: (vectors, metadata)
                - vectors: List of embedding vectors.
                - metadata: Mapping from index to (text + metadata).
        """

        if not processed_documents:
            raise EmptyDocumentError(document_path="Embedder")

        logger.debug(
            f"Starting embedding process for {len(processed_documents)} documents")

        try:
            # Separate cached and uncached content
            cached_embeddings, uncached_items = self._get_cached_embeddings(
                processed_documents)

            # Get new embeddings for uncached items
            new_embeddings = self._get_new_embeddings(uncached_items)

            # Combine cached and new embeddings
            vectors = self._combine_embeddings(
                cached_embeddings, new_embeddings, len(processed_documents))

            # Create metadata
            metadata = self._create_metadata(processed_documents)

            logger.info(
                f"Successfully embedded {len(processed_documents)} documents")
            return vectors, metadata

        except Exception as e:
            raise EmbeddingError(
                operation="document embedding",
                reason=f"Failed to embed documents: {e}",
                context={"document_count": len(processed_documents)},
                original_error=e
            ) from e

    def _get_cached_embeddings(self, documents) -> Tuple[
        List[Tuple[int, List[float]]], List[Tuple[int, str, str]]]:
        """Get cached embeddings and identify uncached items."""
        cached_embeddings = []
        uncached_items = []

        for i, doc in enumerate(documents):
            cached = self.cache.get_embedding(
                content=doc.content,
                source=doc.metadata.source,
                kind=settings.vector.DOCUMENT
            )

            if cached is not None:
                cached_embeddings.append((i, cached))
            else:
                uncached_items.append((i, doc.content, doc.metadata.source))

        cache_hit_rate = len(cached_embeddings) / len(
            documents) * 100 if documents else 0
        logger.debug(
            f"Cache hit rate: {cache_hit_rate:.1f}% ({len(cached_embeddings)}/{len(documents)})")

        return cached_embeddings, uncached_items

    def _get_new_embeddings(self, uncached_items) -> List[
        Tuple[int, List[float]]]:
        """Get embeddings for uncached items."""
        if not uncached_items:
            return []

        logger.debug(
            f"Computing embeddings for {len(uncached_items)} new items")

        # Extract texts for batch embedding
        texts = [item[1] for item in uncached_items]

        try:
            # Get embeddings in batch
            new_embeddings = self.embedding_model.embed_documents(texts)

            # Store in cache and create result list
            result = []
            for (original_idx, text, source), embedding in zip(uncached_items,
                                                               new_embeddings):
                # Cache the embedding
                self.cache.store_embedding(
                    content=text,
                    embedding=embedding,
                    source=source,
                    kind=settings.vector.DOCUMENT
                )
                result.append((original_idx, embedding))

            return result

        except Exception as e:
            raise EmbeddingError(
                operation="batch embedding",
                reason=f"Failed to get embeddings from model: {e}",
                context={"batch_size": len(texts)},
                original_error=e
            ) from e

    @staticmethod
    def _combine_embeddings(cached_embeddings, new_embeddings,
                            total_count) -> List[List[float]]:
        """Combine cached and new embeddings in correct order."""
        vectors = [None] * total_count

        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            vectors[idx] = embedding

        # Place new embeddings
        for idx, embedding in new_embeddings:
            vectors[idx] = embedding

        # Verify all positions are filled
        if None in vectors:
            raise EmbeddingError(
                operation="embedding combination",
                reason="Some embeddings are missing after combination"
            )

        return vectors

    @staticmethod
    def _create_metadata(documents) -> Dict[int, Dict]:
        """Create metadata dictionary."""
        return {
            idx: {"text": doc.content, "meta": doc.metadata}
            for idx, doc in enumerate(documents)
        }

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string with caching.

        Args:
            query: The query string to embed

        Returns:
            List[float]: The embedding vector
        """
        if not query or not query.strip():
            raise EmbeddingError(
                operation="query embedding",
                reason="Query is empty or contains only whitespace"
            )

        try:
            # Check cache first
            cached = self.cache.get_embedding(
                content=query.strip(),
                source="query",
                kind=settings.vector.QUERY
            )

            if cached is not None:
                logger.debug("Cache hit found! Using cached data.")
                return cached

            # Get new embedding
            logger.debug("Computing new embedding for query")
            embedding = self.embedding_model.embed_query(query.strip())

            # Cache the result
            self.cache.store_embedding(
                content=query.strip(),
                embedding=embedding,
                source="query",
                kind=settings.vector.QUERY
            )

            return embedding

        except Exception as e:
            raise EmbeddingError(
                operation="query embedding",
                reason=f"Failed to embed query: {e}",
                context={"query_length": len(query)},
                original_error=e
            ) from e

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = self.cache.get_cache_stats()

        # Add additional stats
        base_stats.update({
            "model_name": getattr(self.embedding_model, 'model', 'unknown'),
            "cache_enabled": settings.vector.EMBEDDING_CACHE_ENABLED
        })

        return base_stats

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        logger.info("Clearing embedding cache")
        self.cache.clear_cache()

    def validate_cache(self) -> Dict[str, Any]:
        """Validate cache integrity."""
        return self.cache.validate_cache()

    def cleanup_cache(self) -> None:
        """Perform cache cleanup if needed."""
        if self.cache._should_cleanup():
            logger.info("Performing cache cleanup")
            self.cache._cleanup_cache()
