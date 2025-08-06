"""
Responsible for running the application in the terminal
"""

import signal
import sys
import os

from src.components.config.settings import settings
from src.components.config.logger import get_logger

logger = get_logger(__name__)


class TerminalUsage:
    """Handles running the application in the terminal"""

    def __init__(self, app):
        self.app = app

    def __validate_environment(self):
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

        if settings.IS_WEB_SEARCH_ENABLED:
            web_search_vars = ["SEARCH_API_KEY", "SEARCH_ENGINE_ID"]

            missing_web_vars = [var for var in web_search_vars if not os.getenv(var)]

            if missing_web_vars:
                logger.warning(
                    f"Web search enabled but missing API keys: {', '.join(missing_web_vars)}"
                )

                print("⚠️  Warning: Web search is enabled but missing API keys:")

                for var in missing_web_vars:
                    print(f"   - {var}")

                print("   Web search will fall back to alternative methods.")


    def __print_startup_info(self):
        """
        Print startup information about the chatbot configuration.
        """

        print("🚀 Starting RAG Chatbot")
        print("=" * 50)
        print(f"📊 Web Search: {'✅ Enabled' if settings.IS_WEB_SEARCH_ENABLED else '❌ Disabled'}")
        print(f"📁 Index Path: {settings.VECTOR_DB_FILE_PATH}")
        print(f"📋 Metadata Path: {settings.METADATA_DB_FILE_PATH}")
        print("=" * 50)

    def handle_shutdown_signal(self, signum, frame):
        """Handles program exits via Ctrl+C or termination signals"""

        self.app.shutdown()

    def run(self):
        """Runs the chatbot"""

        try:
            # Handle SIGINT (Ctrl+C) and SIGTERM (container shutdown)
            signal.signal(signal.SIGINT, self.handle_shutdown_signal)
            signal.signal(signal.SIGTERM, self.handle_shutdown_signal)

            self.__validate_environment()

            self.__print_startup_info()

            logger.debug("Initializing RAG Chatbot...")

            # Start the interactive chatbot
            logger.info("Starting interactive chatbot session")

            self.app.start_interactive()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            print("\n👋 Goodbye!")
            sys.exit(0)

        except ValueError as e:
            logger.error(f"Fatal error during startup: {e}", exc_info=True)
            sys.exit(1)
