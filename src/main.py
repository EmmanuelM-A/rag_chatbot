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
from src.components.config.settings import settings

load_dotenv("../.env")


if __name__ == "__main__":

    vector_store = VectorStore(
        index_path=settings.vector.VECTOR_DB_FILE_PATH,
        metadata_path=settings.vector.METADATA_DB_FILE_PATH
    )
    document_processor = DocumentProcessor(
        path_to_directory=settings.app.RAW_DOCS_DIRECTORY)
    embedder = Embedder()
    query_handler = QueryHandler(
        embedder=embedder,
        llm_model_name=settings.llm.LLM_MODEL_NAME
    )
    web_searcher = WebSearcher()

    chatbot = RAGChatbotApp(
        vector_store=vector_store,
        document_processor=document_processor,
        embedder=embedder,
        query_handler=query_handler,
        web_searcher=web_searcher
    )

    usage = TerminalUsage(app=chatbot)

    usage.run()

 # TODO: FIX CACHE FOR EMBEDDINGS
# TODO: REVIEW AND OPTIMIZE CODE
# TODO: UPDATE DOCUMENTATION
# TODO: LAYOUT PLAN FOR API CONVERSION
# TODO: BEGIN API