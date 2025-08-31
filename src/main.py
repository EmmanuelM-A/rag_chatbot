"""
The main entry point for the RAG Chatbot.
"""

from dotenv import load_dotenv

from components.chatbot.query_handler import QueryHandler
from components.ingestion.document_processor import DocumentProcessor
from components.retrieval.embedder import Embedder
from components.retrieval.vector_store import VectorStore
from components.retrieval.web_searcher import WebSearcher
from src.components.chatbot.rag_chatbot import RAGChatbotApp
from src.components.chatbot.terminal_usage import TerminalUsage

load_dotenv("../.env")


if __name__ == "__main__":

    embedder = Embedder()

    chatbot = RAGChatbotApp(
        vector_store=VectorStore(),
        document_processor=DocumentProcessor(),
        embedder=embedder,
        query_handler=QueryHandler(embedder=embedder),
        web_searcher=WebSearcher()
    )

    usage = TerminalUsage(app=chatbot)

    usage.run()
