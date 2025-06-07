
# ------------------------------------ PATHS -----------------------------------

# Directory for raw documents to be processed
RAW_DOCS_DIRECTORY = "../data/raw_docs"

# Path to save the FAISS vector index
INDEX_PATH = "../data/db/vector_index.faiss"

# Path to save the metadata associated with the vectors
METADATA_PATH = "../data/db/metadata.pkl"

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
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# ------------------------------------------------------------------------------


# --------------------- Language Model (LLM) Configuration ---------------------

# Default LLM model name for generating responses
LLM_MODEL_NAME = "gpt-3.5-turbo"

# Temperature setting for the LLM (controls randomness)
LLM_TEMPERATURE = 0.7

# Top-k value for retrieving similar chunks from the vector store
RETRIEVAL_TOP_K = 3

# The file path for default prompt used to generate responses based on user query and content
DEFAULT_PROMPT_FILEPATH = "../prompts/default_prompt.yaml"

# ------------------------------------------------------------------------------


# --------------------- Logging Configuration ----------------------------------

# The path to save and access the QA logs
QA_SQLITE_DB_PATH = "../data/db/qa_log.db"
LOG_DIRECTORY = "../logs"

# ------------------------------------------------------------------------------


# --------------------- OTHER CONFIGURATIONS -----------------------------------

# ------------------------------------------------------------------------------
