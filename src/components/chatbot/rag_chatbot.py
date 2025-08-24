"""
Orchestrates the rag chatbot pipeline.
"""

import os
import sys
from typing import Optional, List

import numpy as np

from src.components.config.settings import settings
from src.components.ingestion.document_processor import \
    DefaultDocumentProcessor
from src.components.retrieval.embedder import Embedder
from src.components.chatbot.query_handler import QueryHandler
from src.components.retrieval.vector_store import VectorStore
from src.components.retrieval.web_searcher import WebSearcher
from dotenv import load_dotenv

from src.components.config.logger import get_logger
from src.utils.exceptions import DocumentProcessingError, EmbeddingError, \
    VectorStoreError, QueryProcessingError, RAGChatbotError
from src.utils.helper import does_file_exist

load_dotenv()

logger = get_logger(__name__)


class RAGChatbotApp:
    """
    Processes a user query by leveraging a RAG (Retrieval Augmented Generation)
    pipeline. It checks for an existing vector store, builds it if necessary,
    retrieves relevant document chunks, and generates an LLM-based response.
    """

    def __init__(
            self,
            raw_docs_directory: str,
            index_path: str,
            metadata_path: str,
            embedding_model_name: str,
            llm_model_name: str
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Core components
        self.vector_store = VectorStore(index_path, metadata_path)
        self.document_processor = DefaultDocumentProcessor(raw_docs_directory)
        self.embedder = Embedder(embedding_model_name)
        self.query_handler = QueryHandler(
            embedding_model_name, llm_model_name
        )

        # Enhanced components
        self.web_searcher = (
            WebSearcher() if settings.IS_WEB_SEARCH_ENABLED else None
        )

        # Initialize database for logging
        # init_db()


    def _search_documents(self, query: str) -> Optional[List[dict]]:
        """
        Search the document vector store for relevant chunks.

        Args:
            query: User's query string

        Returns:
            List of relevant document chunks or None if no results
        """

        if not does_file_exist(self.index_path) or not does_file_exist(self.metadata_path):
            logger.info(
                "FAISS index or metadata not found! Creating new ones..."
            )

            try:
                processed_documents = self.document_processor.process_documents()

                logger.debug(f"Processed {len(processed_documents)} documents")

                if not processed_documents:
                    logger.warning("No documents were processed!")
                    return None
            except Exception as e:
                logger.error(
                    f"Document processing failed: {e}",
                    exc_info=True
                )
                raise DocumentProcessingError(f"Failed to process documents: {e}") from e

            try:
                vectors, metadata = self.embedder.create_embedded_chunks(
                    processed_documents
                )

                logger.debug(f"Created {len(vectors)} embeddings")
            except Exception as e:
                logger.error(
                    f"Embedding creation failed: {e}",
                    exc_info=True
                )
                raise EmbeddingError(f"Failed to create embeddings: {e}")

            try:
                self.vector_store.save_faiss_index(vectors, metadata)

                logger.info("Saved FAISS index and metadata")
            except Exception as e:
                logger.error(f"Vector store save failed: {e}",
                             exc_info=True)
                raise VectorStoreError(f"Failed to save vector store: {e}")

        try:
            logger.info("Loading FAISS index and metadata...")

            index, metadata = self.vector_store.load_faiss_index()

            logger.debug(f"Loaded index with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Vector store load failed: {e}", exc_info=True)

            raise VectorStoreError(f"Failed to load vector store: {e}")

        try:
            results = self.query_handler.search(query, index, metadata)

            logger.debug(
                f"Search returned {len(results) if results else 0} results"
            )

            return results
        except Exception as e:
            logger.error(f"Query search failed: {e}", exc_info=True)

            raise QueryProcessingError(f"Failed to search query: {e}")

    def _search_web_and_add_to_store(self, query: str) -> Optional[List[dict]]:
        """
        Search the web for information and add results to vector store.

        Args:
            query: User's query string

        Returns:
            List of web search results formatted for response generation
        """

        if not self.web_searcher:
            logger.info("Web search is disabled!")
            return None

        try:
            logger.debug(f"Performing web search for: {query}")

            web_documents = self.web_searcher.search_and_retrieve_content(
                query)

            if not web_documents:
                logger.warning("No web content retrieved!")
                return None

            web_chunks = self.web_searcher.chunk_web_documents(web_documents)

            if not web_chunks:
                logger.warning("No web chunks created!")
                return None

            logger.info(
                f"Retrieved {len(web_chunks)} chunks from web search"
            )

            # Create embeddings for web content
            texts = [doc.content for doc in web_chunks]
            web_vectors = self.embedder.embedding_model.embed_documents(texts)

            # Create metadata for web chunks
            web_metadata = {
                idx: {"text": doc.content, "meta": doc.metadata}
                for idx, doc in enumerate(web_chunks)
            }

            # Load existing index and metadata
            try:
                index, existing_metadata = self.vector_store.load_faiss_index()

                # Add new vectors to existing index
                index.add(np.array(web_vectors).astype("float32"))

                # Merge metadata (adjust indices for new vectors)
                offset = len(existing_metadata)
                for idx, data in web_metadata.items():
                    existing_metadata[offset + idx] = data

                # Save updated index and metadata
                self.vector_store.save_faiss_index(None, existing_metadata,
                                                   existing_index=index)

                logger.info("Web content added to existing vector stores!")

            except FileNotFoundError:
                # If no existing index, create new one with just web content
                self.vector_store.save_faiss_index(
                    vectors=web_vectors,
                    metadata=web_metadata
                )

                logger.info("Created new vector store with web content")

            # Format results for response generation
            web_results = []
            for doc in web_chunks:
                web_results.append({
                    "text": doc.content,
                    "metadata": doc.metadata
                })

            return web_results

        except Exception as e:
            logger.error(f"Error in web search and storage: {e}")
            return None

    def process_query(self, user_query: str) -> dict:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: The user's question/query

        Returns:
            Dictionary containing answer, sources, and metadata
        """

        logger.info(f"Processing query: {user_query}")

        try:
            # Step 1: Search documents first
            document_results = self._search_documents(user_query)

            if document_results:
                try:
                    response_data = self.query_handler.generate_responses(
                        user_query, document_results)

                    if response_data:
                        response_data["source_type"] = "documents"
                        logger.info("Response generated from documents")
                        return response_data
                except Exception as e:
                    logger.error(
                        f"Response generation failed: {e}",
                        exc_info=True
                    )
                    raise QueryProcessingError(
                        f"Failed to generate response: {e}")

            # Fall back to web search if query is relevant but no document results
            web_results = self._search_web_and_add_to_store(user_query)

            if web_results:
                # Generate response from web content
                response_data = self.query_handler.generate_responses(
                    user_query, web_results)
                response_data["source_type"] = "web_search"

                # Log the QA pair
                # log_qa_pair(user_query, response_data["answer"], response_data["sources"])

                logger.info("Response generated from web search")
                return response_data

            # No results found anywhere
            response_data = {
                "answer": "I couldn't find relevant information to answer your "
                          "question in my documents or through web search. Please "
                          "try rephrasing your question or ask about a different "
                          "topic.",
                "sources": [],
                "source_type": "none"
            }

            # Log the QA pair
            # log_qa_pair(user_query, response_data["answer"], response_data["sources"])

            logger.info("No relevant information found")
            return response_data
        except RAGChatbotError:
            # Re-raise critical RAG errors to be handled by terminal_usage
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error processing query: {e}",
                exc_info=True
            )
            raise RAGChatbotError(f"Unexpected error in query processing: {e}")

    def start_interactive(self):
        """
        Start the chatbot in interactive terminal mode.
        """

        print("🤖 Hi, I am Bob. Your RAG assistant!")
        print("I can answer questions from my documents and search "
              "the web when needed.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                query = input("🔍 Ask me: ").strip().lower()

                if query in ['quit', 'exit', 'bye']:
                    print("👋 Happy to be of service! Goodbye!")
                    self.shutdown()

                if not query:
                    print("Please enter a question.")
                    continue

                # Process the query
                response_data = self.process_query(query)

                # Display response
                print(f"\n📝 Response: {response_data['answer']}")

                if response_data['sources']:
                    print(f"\n📚 Sources ({response_data['source_type']}):")
                    for i, source in enumerate(response_data['sources'], 1):
                        print(f"  {i}. {source}")

                print("\n" + "=" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")

                self.shutdown()
            except RAGChatbotError as e:
                # Critical error - propagate to terminal_usage to exit
                logger.error(f"Critical RAG error: {e}")
                raise
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")

                print("❌ An error occurred. Please try again.")

    def shutdown(self):
        """
        Gracefully shut down the application.
        """

        logger.info("Shutting down enhanced RAG chatbot...")

        try:
            # Clean up resources if needed
            if hasattr(self, 'vector_store'):
                # Any cleanup for vector store
                pass

            logger.info("Shutdown completed successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        sys.exit(0)
