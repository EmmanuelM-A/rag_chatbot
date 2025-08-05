"""
Contains all the application configurations.
"""

# ------------------------------------ PATHS -----------------------------------

RAW_DOCS_DIRECTORY = "../data/raw_docs"

FAISS_INDEX_FILE_PATH = "../../../data/db/vector_index.faiss"

# Path to save the metadata associated with the vectors
VECTOR_METADATA_FILE_PATH = "../../../data/db/metadata.pkl"

# Path to save document topics/subjects for relevance checking
DOCUMENT_TOPICS_FILE_PATH = "../data/db/document_topics.pkl"

# ------------------------------------------------------------------------------


# --------------------- DOCUMENT PROCESSING ------------------------------------

ALLOWED_FILE_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]

# Chunk size for text splitting
CHUNK_SIZE = 1000

# Overlap between chunks for text splitting
CHUNK_OVERLAP = 20

# ------------------------------------------------------------------------------


# --------------------- EMBEDDING MODEL CONFIGURATION --------------------------

# OpenAI embedding model name
DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# ------------------------------------------------------------------------------


# --------------------- Language Model (LLM) Configuration ---------------------

DEFAULT_LLM_MODEL_NAME = "gpt-3.5-turbo"

# Temperature setting for the LLM (controls randomness)
LLM_TEMPERATURE = 0.7

# Top-k value for retrieving similar chunks from the vector store
RETRIEVAL_TOP_K = 3

# The file path for default prompt used to generate responses based on user query and content
DEFAULT_RESPONSE_PROMPT_FILEPATH = "../../../data/prompts/default_response_prompt.yaml"

# Minimum similarity threshold for considering retrieved chunks relevant
RELEVANCE_THRESHOLD = 0.7

# ------------------------------------------------------------------------------


# --------------------- WEB SEARCH CONFIGURATION -------------------------------

# Enable/disable web search fallback when no relevant documents found
WEB_SEARCH_ENABLED = True

# Enable/disable relevance checking before web search
RELEVANCE_CHECK_ENABLED = True

# Similarity threshold for determining if query is relevant to document corpus
TOPIC_RELEVANCE_THRESHOLD = 0.6

# Maximum number of web search results to retrieve
MAX_WEB_SEARCH_RESULTS = 5

# Prompt file for topic analysis (used in relevance checking)
TOPIC_ANALYSIS_PROMPT_FILEPATH = "../../../data/prompts/topic_analysis_prompt.yaml"

# Timeout for web requests (in seconds)
WEB_REQUEST_TIMEOUT = 15

# Delay between web requests to be respectful to servers (in seconds)
WEB_REQUEST_DELAY = 1

# Maximum content length to extract from web pages (in characters)
MAX_WEB_CONTENT_LENGTH = 10000

# Minimum content length to consider a web page viable (in characters)
MIN_WEB_CONTENT_LENGTH = 100

# User agent string for web requests
WEB_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# ------------------------------------------------------------------------------


# --------------------- RELEVANCE CHECKING CONFIGURATION -----------------------

# Enable semantic relevance checking using LLM (in addition to embedding similarity)
SEMANTIC_RELEVANCE_CHECK_ENABLED = True

# Maximum number of keywords to use in semantic relevance checking
MAX_KEYWORDS_FOR_RELEVANCE = 10

# Maximum number of documents to analyze for topic extraction
MAX_DOCUMENTS_FOR_TOPIC_EXTRACTION = 10

# Maximum content length to analyze per document for topic extraction
MAX_CONTENT_LENGTH_FOR_TOPIC_EXTRACTION = 500

# ------------------------------------------------------------------------------


# --------------------- LOGGING CONFIGURATION ----------------------------------

# The path to save and access the QA logs
QA_SQLITE_DB_PATH = "../../../data/db/qa_log.db"

# The path to the directory that stores the log files
LOG_DIRECTORY = "../logs"

# Log web search activities
LOG_WEB_SEARCH = True

# Log relevance checking activities
LOG_RELEVANCE_CHECKS = True

# ------------------------------------------------------------------------------


# --------------------- PERFORMANCE CONFIGURATION ------------------------------

# Maximum number of vectors to keep in memory at once
MAX_VECTORS_IN_MEMORY = 10000

# Batch size for processing vectors
VECTOR_BATCH_SIZE = 100

# Enable/disable caching of embeddings
EMBEDDING_CACHE_ENABLED = True

# Cache directory for embeddings
EMBEDDING_CACHE_DIR = "../data/cache/embeddings"

# Maximum cache size in MB
MAX_CACHE_SIZE_MB = 500

# ------------------------------------------------------------------------------


# --------------------- API CONFIGURATION --------------------------------------

# Rate limiting for API calls (calls per minute)
OPENAI_API_RATE_LIMIT = 60

# Timeout for OpenAI API calls (in seconds)
OPENAI_API_TIMEOUT = 30

# Maximum retries for failed API calls
MAX_API_RETRIES = 3

# Delay between retries (in seconds)
API_RETRY_DELAY = 1

# ------------------------------------------------------------------------------


# --------------------- DEVELOPMENT/DEBUG CONFIGURATION ------------------------

# Enable debug mode (more verbose logging, additional checks)
DEBUG_MODE = False

# Enable performance profiling
ENABLE_PROFILING = False

# Save intermediate results for debugging
SAVE_INTERMEDIATE_RESULTS = False

# Directory for debug outputs
DEBUG_OUTPUT_DIR = "../debug"

# ------------------------------------------------------------------------------


# --------------------- FEATURE FLAGS ------------------------------------------

# Enable experimental features
ENABLE_EXPERIMENTAL_FEATURES = False

# Enable query preprocessing (spell check, expansion, etc.)
ENABLE_QUERY_PREPROCESSING = False

# Enable response post-processing (fact checking, formatting, etc.)
ENABLE_RESPONSE_POSTPROCESSING = False

# Enable multi-language support
ENABLE_MULTILINGUAL_SUPPORT = False

# ------------------------------------------------------------------------------


# --------------------- SECURITY CONFIGURATION ---------------------------------

# Enable input sanitization
ENABLE_INPUT_SANITIZATION = True

# Maximum query length to prevent abuse
MAX_QUERY_LENGTH = 1000

# Enable output filtering (remove sensitive information)
ENABLE_OUTPUT_FILTERING = True

# List of sensitive patterns to filter from outputs
SENSITIVE_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
    r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card pattern
]

# ------------------------------------------------------------------------------


# --------------------- OTHER CONFIGURATIONS -----------------------------------

# Application version
APP_VERSION = "2.0.0"

# Application name
APP_NAME = "Enhanced RAG Chatbot"

# Support contact information
SUPPORT_EMAIL = "support@example.com"
