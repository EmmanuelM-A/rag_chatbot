"""
Contains all the application configurations
"""

# ------------------------------------ PATHS -----------------------------------

# Directory for raw documents to be processed
RAW_DOCS_DIRECTORY = "../data/raw_docs"

# Path to save the FAISS vector index
INDEX_PATH = "../data/db/vector_index.faiss"

# Path to save the metadata associated with the vectors
METADATA_PATH = "../data/db/metadata.pkl"

# Path to save document topics/subjects for relevance checking
DOCUMENT_TOPICS_PATH = "../data/db/document_topics.pkl"

# ------------------------------------------------------------------------------


# --------------------- DOCUMENT PROCESSING ------------------------------------

# Allowed file extensions for document ingestion
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
DEFAULT_RESPONSE_PROMPT_FILEPATH = "../prompts/default_response_prompt.yaml"

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

TOPIC_ANALYSIS_PROMPT_FILEPATH = "../prompts/topic_analysis_prompt.yaml"

# ------------------------------------------------------------------------------


# --------------------- Logging Configuration ----------------------------------

# The path to save and access the QA logs
QA_SQLITE_DB_PATH = "../data/db/qa_log.db"

# The path to the directory that stores the log files
LOG_DIRECTORY = "../logs"

# ------------------------------------------------------------------------------


# --------------------- OTHER CONFIGURATIONS -----------------------------------

# ------------------------------------------------------------------------------
