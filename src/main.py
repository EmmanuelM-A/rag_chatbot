"""
The main entry point for the RAG Chatbot.
"""

import signal
import sys
import os

from dotenv import load_dotenv

from src.components.chatbot.rag_chatbot import RAGChatbotApp
from src.components.config.config import (
    VECTOR_METADATA_FILE_PATH,
    FAISS_INDEX_FILE_PATH,
    WEB_SEARCH_ENABLED,
    RELEVANCE_CHECK_ENABLED,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME, RAW_DOCS_DIRECTORY
)
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def validate_environment():
    """
    Validate that all required environment variables and configurations are set.
    """
    required_env_vars = [
        "OPENAI_API_KEY"  # Required for embeddings and LLM
    ]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("❌ Error: Please set the following environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        sys.exit(1)

    # Optional: Check for web search API keys if web search is enabled
    if WEB_SEARCH_ENABLED:
        web_search_vars = ["SEARCH_API_KEY", "SEARCH_ENGINE_ID"]
        missing_web_vars = [var for var in web_search_vars if not os.getenv(var)]

        if missing_web_vars:
            logger.warning(f"Web search enabled but missing API keys: "
                           f"{', '.join(missing_web_vars)}")

            print("⚠️  Warning: Web search is enabled but missing API keys:")
            for var in missing_web_vars:
                print(f"   - {var}")
            print("   Web search will fall back to alternative methods.")


def print_startup_info():
    """
    Print startup information about the chatbot configuration.
    """
    print("🚀 Starting Enhanced RAG Chatbot")
    print("=" * 50)
    print(f"📊 Web Search: {'✅ Enabled' if WEB_SEARCH_ENABLED else '❌ Disabled'}")
    print(f"🔍 Relevance Check: {'✅ Enabled' if RELEVANCE_CHECK_ENABLED else '❌ Disabled'}")
    print(f"📁 Index Path: {FAISS_INDEX_FILE_PATH}")
    print(f"📋 Metadata Path: {VECTOR_METADATA_FILE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        validate_environment()

        print_startup_info()

        logger.debug("Initializing RAG Chatbot...")

        chatbot = RAGChatbotApp(
            raw_docs_directory=RAW_DOCS_DIRECTORY,
            index_path=FAISS_INDEX_FILE_PATH,
            metadata_path=VECTOR_METADATA_FILE_PATH,
            embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
            llm_model_name=DEFAULT_LLM_MODEL_NAME
        )

        def handle_shutdown_signal(signum, frame):
            """Handles program exits via Ctrl+C or termination signals"""
            chatbot.shutdown()

        # Handle SIGINT (Ctrl+C) and SIGTERM (container shutdown)
        signal.signal(signal.SIGINT, handle_shutdown_signal)
        signal.signal(signal.SIGTERM, handle_shutdown_signal)

        # Start the interactive chatbot
        logger.info("Starting interactive chatbot session")

        chatbot.start_interactive()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n👋 Goodbye!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        sys.exit(1)
