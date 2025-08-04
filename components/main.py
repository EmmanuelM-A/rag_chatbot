"""
Entry point to run the RAG Chatbot
"""

import os

from components.embedder import prepare_document_chunks, create_embedded_chunks
from vector_store import load_faiss_index, save_faiss_index
from query_handler import search, generate_response
from config import RAW_DOCS_DIRECTORY, METADATA_PATH, INDEX_PATH
from dotenv import load_dotenv

from utils.logger import get_logger
from evaluation import init_db, log_qa_pair

load_dotenv()

logger = get_logger("main_logger")


def terminal_usage(user_query):
    """
    Processes a user query by leveraging a RAG (Retrieval Augmented Generation) pipeline.
    It checks for an existing vector store, builds it if necessary,
    retrieves relevant document chunks, and generates an LLM-based response.

    Args:
        user_query (str): The question or query provided by the user.
    """

    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        logger.info("Existing FAISS index and metadata found. Loading now...")
    else:
        logger.info("FAISS index or metadata not found. Creating new ones.")

        documents = prepare_document_chunks(RAW_DOCS_DIRECTORY)

        vectors, metadata = create_embedded_chunks(documents)

        save_faiss_index(vectors, metadata)

    index, meta = load_faiss_index()

    results = search(user_query, index, meta)

    if results:
        llm_response_data = generate_response(user_query, results)

        log_qa_pair(user_query, llm_response_data["answer"], llm_response_data["sources"])

        print(f"Response: {llm_response_data["answer"]}")
    else:
        print("\nNo relevant information found in the documents.")


def main():
    """
    Main function to run the chatbot in a terminal interface.
    Initializes the chatbot and enters a loop to accept user input.
    """

    init_db()

    print("Hi, I am Bob. Your personal assistant chatbot. If you have any "
          "question, feel free to ask me.")

    # Start an infinite loop to continuously accept user queries.
    while True:
        # Prompt the user for input.
        query = input("Ask me: ")

        # Check for a 'quit' command to exit the chatbot.
        if query.lower() == "quit":
            print("Happy to be of service! Goodbye for now.")
            return

        # Process the user's query
        terminal_usage(query)


if __name__ == "__main__":
    main()
