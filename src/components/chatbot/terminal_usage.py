"""
Responsible for running the application in the terminal
"""

import signal
import sys
import os

from src.components.chatbot.rag_chatbot import RAGChatbotApp
from src.components.config.settings import settings
from src.components.config.logger import get_logger
from src.utils.exceptions import RAGChatbotError

logger = get_logger(__name__)


class TerminalUsage:
    """Handles running the application in the terminal"""

    def __init__(self, app: RAGChatbotApp):
        self.app = app

    @staticmethod
    def __validate_environment():
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

        if settings.web.IS_WEB_SEARCH_ENABLED:
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

    @staticmethod
    def __print_startup_info():
        """
        Print startup information about the chatbot configuration.
        """

        print("🚀 Starting RAG Chatbot")
        print("=" * 50)
        print(f"📊 Web Search: {'✅ Enabled' if settings.web.IS_WEB_SEARCH_ENABLED
                                else '❌ Disabled'}")
        print(f"📁 Index Path: {settings.vector.VECTOR_DB_FILE_PATH}")
        print(f"📋 Metadata Path: {settings.vector.METADATA_DB_FILE_PATH}")
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

            self.start_interactive()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            print("\n👋 Goodbye!")
            sys.exit(0)
        except RAGChatbotError as e:
            # Critical RAG error occurred - exit gracefully
            logger.error(f"Critical RAG chatbot error: {e}", exc_info=True)
            print(f"\n❌ Critical Error: {e}")
            print("The application cannot continue and will now exit.")
            print("Please check the logs for more details.")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Configuration error during startup: {e}",
                         exc_info=True)
            print(f"\n❌ Configuration Error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected fatal error during startup: {e}",
                         exc_info=True)
            print(f"\n❌ Fatal Error: {e}")
            print("An unexpected error occurred. Please check the logs.")
            sys.exit(1)

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
                response_data = self.app.process_query(query)

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
            logger.error("Error during shutdown: %s", e)

        sys.exit(0)
