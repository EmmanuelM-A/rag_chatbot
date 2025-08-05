"""
Orchestrates the rag chatbot pipeline.
"""

import os
import sys
from typing import Optional, List

import numpy as np

from components.config import WEB_SEARCH_ENABLED
from components.embedder import Embedder
from components.query_handler import QueryHandler
from components.relevance_checker import RelevanceChecker
from components.vector_store import VectorStore
from components.web_searcher import WebSearcher
from config import (RELEVANCE_CHECK_ENABLED)
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()


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
        self.logger = get_logger(__name__)
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Core components
        self.vector_store = VectorStore(index_path, metadata_path)
        self.embedder = Embedder(raw_docs_directory, embedding_model_name)
        self.query_handler = QueryHandler(
            embedding_model_name, llm_model_name
        )

        # Enhanced components
        self.relevance_checker = (
            RelevanceChecker(
                embedding_model_name,
                llm_model_name
            ) if RELEVANCE_CHECK_ENABLED else None
        )
        self.web_searcher = (
            WebSearcher() if WEB_SEARCH_ENABLED else None
        )

        # Initialize database for logging
        # init_db()

        # Document topics for relevance checking
        self.document_topics_initialized = False

    def _initialize_document_topics(self):
        """Initialize document topics for relevance checking."""
        if not self.relevance_checker or self.document_topics_initialized:
            return

        try:
            # Load existing topics or extract from documents
            if not self.relevance_checker.document_topics:
                self.logger.info("Extracting topics from document corpus...")
                documents = self.embedder.document_processor.load_documents()
                if documents:
                    self.relevance_checker.extract_document_topics(documents)
                    self.document_topics_initialized = True
                    self.logger.info(
                        "Document topics initialized successfully")
                else:
                    self.logger.warning(
                        "No documents found for topic extraction")
            else:
                self.document_topics_initialized = True
                self.logger.info("Document topics loaded from existing data")

        except Exception as e:
            self.logger.error(f"Error initializing document topics: {e}")
            self.document_topics_initialized = False

    def _is_query_relevant_to_documents(self, query: str) -> bool:
        """
        Check if query is relevant to the document corpus.

        Args:
            query: User's query string

        Returns:
            True if query is relevant to documents, False otherwise
        """
        if not RELEVANCE_CHECK_ENABLED or not self.relevance_checker:
            self.logger.debug(
                "Relevance checking disabled, assuming query is relevant")
            return True

        if not self.document_topics_initialized:
            self._initialize_document_topics()

        if not self.document_topics_initialized:
            self.logger.warning(
                "Document topics not initialized, assuming query is relevant")
            return True

        try:
            is_relevant = self.relevance_checker.is_query_relevant(query)
            self.logger.info(
                f"Query relevance check: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
            return is_relevant
        except Exception as e:
            self.logger.error(f"Error in relevance checking: {e}")
            return True  # Default to allowing the query

    def _search_documents(self, query: str) -> Optional[List[dict]]:
        """
        Search the document vector store for relevant chunks.

        Args:
            query: User's query string

        Returns:
            List of relevant document chunks or None if no results
        """
        try:
            if not os.path.exists(self.index_path) or not os.path.exists(
                    self.metadata_path):
                self.logger.info(
                    "FAISS index or metadata not found. Creating new ones...")
                vectors, metadata = self.embedder.create_embedded_chunks()
                self.vector_store.save_faiss_index(vectors, metadata)

            self.logger.info("Loading FAISS index and metadata...")
            index, metadata = self.vector_store.load_faiss_index()

            results = self.query_handler.search(query, index, metadata)

            if results:
                # Check if results meet relevance threshold
                relevant_results = self._filter_relevant_results(results,
                                                                 query)
                if relevant_results:
                    self.logger.info(
                        f"Found {len(relevant_results)} relevant document chunks")
                    return relevant_results
                else:
                    self.logger.info(
                        "No document chunks meet relevance threshold")
                    return None
            else:
                self.logger.info("No results found in document search")
                return None

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return None

    def _filter_relevant_results(self, results: List[dict], query: str) -> \
    List[dict]:
        """
        Filter search results based on relevance threshold.

        Args:
            results: List of search results
            query: Original query string

        Returns:
            Filtered list of relevant results
        """
        # For now, we'll use a simple approach based on the existing similarity search
        # In a more sophisticated implementation, you could add additional relevance scoring
        return results

    def _search_web_and_add_to_store(self, query: str) -> Optional[List[dict]]:
        """
        Search the web for information and add results to vector store.

        Args:
            query: User's query string

        Returns:
            List of web search results formatted for response generation
        """
        if not WEB_SEARCH_ENABLED or not self.web_searcher:
            self.logger.info("Web search is disabled")
            return None

        try:
            self.logger.info(f"Performing web search for: {query}")

            # Get web documents
            web_documents = self.web_searcher.search_and_retrieve_content(
                query)

            if not web_documents:
                self.logger.warning("No web content retrieved")
                return None

            # Chunk web documents
            web_chunks = self.web_searcher.chunk_web_documents(web_documents)

            if not web_chunks:
                self.logger.warning("No web chunks created")
                return None

            self.logger.info(
                f"Retrieved {len(web_chunks)} chunks from web search")

            # Create embeddings for web content
            texts = [doc.content for doc in web_chunks]
            vectors = self.embedder.embedding_model.embed_documents(texts)

            # Create metadata for web chunks
            web_metadata = {
                idx: {"text": doc.content, "meta": doc.metadata}
                for idx, doc in enumerate(web_chunks)
            }

            # Load existing index and metadata
            try:
                index, existing_metadata = self.vector_store.load_faiss_index()

                # Add new vectors to existing index
                index.add(np.array(vectors).astype("float32"))

                # Merge metadata (adjust indices for new vectors)
                offset = len(existing_metadata)
                for idx, data in web_metadata.items():
                    existing_metadata[offset + idx] = data

                # Save updated index and metadata
                self.vector_store.save_faiss_index(None, existing_metadata,
                                                   existing_index=index)

                self.logger.info("Web content added to vector store")

            except FileNotFoundError:
                # If no existing index, create new one with just web content
                self.vector_store.save_faiss_index(vectors, web_metadata)
                self.logger.info("Created new vector store with web content")

            # Format results for response generation
            web_results = []
            for doc in web_chunks:
                web_results.append({
                    "text": doc.content,
                    "metadata": doc.metadata
                })

            return web_results

        except Exception as e:
            self.logger.error(f"Error in web search and storage: {e}")
            return None

    def process_query(self, user_query: str) -> dict:
        """
        Process a user query through the enhanced RAG pipeline.

        Args:
            user_query: The user's question/query

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        self.logger.info(f"Processing query: {user_query}")

        try:
            # Step 1: Search documents first
            document_results = self._search_documents(user_query)

            if document_results:
                # Generate response from documents
                response_data = self.query_handler.generate_responses(
                    user_query, document_results)
                response_data["source_type"] = "documents"

                # Log the QA pair
                # log_qa_pair(user_query, response_data["answer"], response_data["sources"])

                self.logger.info("Response generated from documents")
                return response_data

            # Step 2: Check if query is relevant to document corpus before web search
            if not self._is_query_relevant_to_documents(user_query):
                response_data = {
                    "answer": "I don't have information about that topic in my knowledge base. This query appears to be outside the scope of the documents I have access to.",
                    "sources": [],
                    "source_type": "none"
                }

                # Log the QA pair
                # log_qa_pair(user_query, response_data["answer"], response_data["sources"])

                self.logger.info(
                    "Query deemed not relevant to document corpus")
                return response_data

            # Step 3: Fall back to web search if query is relevant but no document results
            web_results = self._search_web_and_add_to_store(user_query)

            if web_results:
                # Generate response from web content
                response_data = self.query_handler.generate_responses(
                    user_query, web_results)
                response_data["source_type"] = "web_search"

                # Log the QA pair
                # log_qa_pair(user_query, response_data["answer"], response_data["sources"])

                self.logger.info("Response generated from web search")
                return response_data

            # Step 4: No results found anywhere
            response_data = {
                "answer": "I couldn't find relevant information to answer your question in my documents or through web search. Please try rephrasing your question or ask about a different topic.",
                "sources": [],
                "source_type": "none"
            }

            # Log the QA pair
            # log_qa_pair(user_query, response_data["answer"], response_data["sources"])

            self.logger.info("No relevant information found")
            return response_data

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            error_response = {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "source_type": "error"
            }
            return error_response

    def start_interactive(self):
        """
        Start the chatbot in interactive terminal mode.
        """
        print("🤖 Hi, I am Bob. Your enhanced RAG assistant!")
        print(
            "I can answer questions from my documents and search the web when needed.")
        print("Type 'quit' to exit.\n")

        # Initialize document topics if relevance checking is enabled
        if RELEVANCE_CHECK_ENABLED:
            print("📚 Initializing document analysis for relevance checking...")
            self._initialize_document_topics()
            print("✅ Ready to answer your questions!\n")

        while True:
            try:
                query = input("🔍 Ask me: ").strip()

                if query.lower() in ['quit', 'exit', 'bye']:
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
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                print("❌ An error occurred. Please try again.")

    def shutdown(self):
        """
        Gracefully shut down the application.
        """
        self.logger.info("Shutting down enhanced RAG chatbot...")

        try:
            # Clean up resources if needed
            if hasattr(self, 'vector_store'):
                # Any cleanup for vector store
                pass

            self.logger.info("Shutdown completed successfully")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

        sys.exit(0)
