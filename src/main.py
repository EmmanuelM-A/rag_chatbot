"""
The main entry point for the RAG Chatbot.
"""

from dotenv import load_dotenv

from src.components.chatbot.rag_chatbot import RAGChatbotApp
from src.components.chatbot.terminal_usage import TerminalUsage
from src.components.config.settings import settings
from src.components.config.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


if __name__ == "__main__":

    chatbot = RAGChatbotApp(
        raw_docs_directory=settings.RAW_DOCS_DIRECTORY,
        index_path=settings.VECTOR_DB_FILE_PATH,
        metadata_path=settings.METADATA_DB_FILE_PATH,
        embedding_model_name=settings.EMBEDDING_MODEL_NAME,
        llm_model_name=settings.LLM_MODEL_NAME
    )

    usage = TerminalUsage(app=chatbot)

    usage.run()


# TODO: IMPLEMENT AND INCORPORATE CUSTOM EXCEPTIONS
# TODO: FIX AND INCORPORATE THE WEBS SEARCH
# TODO: REVIEW AND OPTIMIZE CODE
# TODO: UPDATE DOCUMENTATION
# TODO: LAYOUT PLAN FOR API CONVERSION
# TODO: BEGIN API