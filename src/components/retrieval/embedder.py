"""
Responsible for wrapping the embedding model client to encode text into
vectors.
"""

from typing import Dict, Any, List

from langchain_openai import OpenAIEmbeddings
from src.components.config.logger import logger
from src.components.config.settings import settings
from src.components.retrieval.embedding_cache import EmbedderCache
from src.utils.exceptions import EmptyDocumentError


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
        self.cache = EmbedderCache()

    def embed_documents(self, processed_documents):
        """
        Embeds the document chunks and returns embedded vectors with the
        associated metadata.

        Args:
            processed_documents (List[FileDocument]): The processed and
                chunked documents to be embedded.

        Returns:
            - tuple: Tuple[Optional[vectors], Optional[metadata]]
                - vectors (List[List[float]]): List of embedding vectors.
                - metadata (dict): Mapping from index to (text + metadata).

        Raises:
            EmptyDocumentError:
        """

        if not processed_documents:
            logger.error("No documents provided for embedding.")
            raise EmptyDocumentError(document_path="Embedder")

        texts = [doc.content for doc in processed_documents]
        sources = [doc.metadata.source for doc in processed_documents]

        cached_embeddings = []
        uncached_texts = []
        index_map = {}

        for i, (text, source) in enumerate(zip(texts, sources)):
            cached = self.cache.get_embedding(
                content=text,
                source=source,
                kind=settings.vector.DOCUMENT
            )
            if cached is not None:
                cached_embeddings.append((i, cached))
            else:
                index_map[i] = len(uncached_texts)
                uncached_texts.append((text, source))

        new_embeddings = []
        if uncached_texts:
            texts_only = [t for t, _ in uncached_texts]
            new_embeddings = self.embedding_model.embed_documents(texts_only)

            for (text, source), embedding in zip(uncached_texts,
                                                 new_embeddings):
                self.cache.store_embedding(text, embedding, source,
                                           kind="document")

        vectors = [None] * len(texts)
        for i, emb in cached_embeddings:
            vectors[i] = emb
        for i, uncached_idx in index_map.items():
            vectors[i] = new_embeddings[uncached_idx]

        metadata = {
            idx: {"text": doc.content, "meta": doc.metadata}
            for idx, doc in enumerate(processed_documents)
        }
        return vectors, metadata

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string with caching.
        """

        cached = self.cache.get_embedding(
            content=query,
            source="query",
            kind=settings.vector.QUERY
        )

        if cached is not None:
            return cached

        embedding = self.embedding_model.embed_query(query)

        self.cache.store_embedding(
            content=query,
            embedding=embedding,
            source="query",
            kind=settings.vector.QUERY
        )

        return embedding

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache.get_cache_stats()

    def clear_cache(self):
        self.cache.clear_cache()
