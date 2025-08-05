"""
Coordinates the overall flow of the RAG Chatbot and starts the interactive CLI
chatbot loop.
"""

import os

from components.embedder import Embedder
from components.vector_store import VectorStore
from query_handler import search, generate_response
from config import RAW_DOCS_DIRECTORY, METADATA_PATH, INDEX_PATH
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class RAGChatbotApp:
    """
    Processes a user query by leveraging a RAG (Retrieval Augmented Generation)
    pipeline. It checks for an existing vector store, builds it if necessary,
    retrieves relevant document chunks, and generates an LLM-based response.
    """

    def __init__(self, index_path: str, metadata_path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.vector_store = VectorStore(index_path, metadata_path)
        self.embedder = Embedder(RAW_DOCS_DIRECTORY)

    def start_interactive(self):
        """
        Starts the chatbot in a terminal interface.
        Initializes the chatbot and enters a loop to accept user input.
        """

        print("Hi, I am Bob. Your personal assistant chatbot. If you have any "
              "question, feel free to ask me. If you wish to exit enter quit")

        # Start an infinite loop to continuously accept user queries.
        while True:
            # Prompt the user for input.
            query = input("Ask me: ")

            # Check for a 'quit' command to exit the chatbot.
            if query.lower() == "quit":
                print("Happy to be of service! Goodbye for now.")

                self.shutdown()

            # Process the user's query
            self._terminal_usage(query)

    def _terminal_usage(self, user_query: str):
        """
        Runs the RAG pipeline to generate the response from the user query.
        """

        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.info("FAISS index or metadata not found. Creating new ones...")

            vectors, metadata = self.embedder.create_embedded_chunks()

            self.vector_store.save_faiss_index(vectors, metadata)

        logger.info("Existing FAISS index and metadata found. Loading now...")

        index, meta = self.vector_store.load_faiss_index()

        results = search(user_query, index, meta)

        if results:
            llm_response_data = generate_response(user_query, results)

            print(f"Response: {llm_response_data["answer"]}")
        else:
            print("\nNo relevant information found in the documents.")

    def shutdown(self):
        """
        Gracefully shutdowns the program.
        """

        return


if __name__ == "__main__":
    chatbot = RAGChatbotApp(INDEX_PATH, METADATA_PATH)

    chatbot.start_interactive()
