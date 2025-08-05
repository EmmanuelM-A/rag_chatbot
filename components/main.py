"""
Coordinates the overall flow of the RAG Chatbot and starts the interactive CLI
chatbot loop.
"""
import signal

from dotenv import load_dotenv

from components.rag_chatbot import RAGChatbotApp
from config import (VECTOR_METADATA_FILE_PATH, FAISS_INDEX_FILE_PATH)

load_dotenv()

if __name__ == "__main__":
    chatbot = RAGChatbotApp(FAISS_INDEX_FILE_PATH, VECTOR_METADATA_FILE_PATH)

    def handle_shutdown_signal(signum, frame):
        """Handles program exists via Ctrl+C"""
        chatbot.shutdown()

    # Handle SIGINT (Ctrl+C) and SIGTERM (container shutdown)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    chatbot.start_interactive()
