from langchain_openai import OpenAIEmbeddings
from process_documents import load_documents, chunk_documents
from utils.logger import get_logger
from config import EMBEDDING_MODEL_NAME

logger = get_logger("embedder_logger")


def prepare_document_chunks(directory):
    """
    Loads and chunks documents from a directory.

    Returns a list of FileDocument objects with chunked content.
    """

    raw_documents = load_documents(directory)

    chunked_documents = chunk_documents(raw_documents)

    return chunked_documents


def create_embedded_chunks(chunked_documents):
    """
    Takes FileDocument list, returns (vectors, metadata).

    vectors: List of embedding vectors metadata: Mapping of index -> (text, metadata)
    """
    if not chunked_documents:
        logger.error("No chunked documents provided!")
        raise ValueError("No chunked documents provided!")

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    texts = [doc.content for doc in chunked_documents]

    vectors = embedding_model.embed_documents(texts)

    metadata = {
        idx: {"text": doc.content, "meta": doc.metadata}
        for idx, doc in enumerate(chunked_documents)
    }

    logger.info("Document vectors and metadata have been created successfully!")

    return vectors, metadata
