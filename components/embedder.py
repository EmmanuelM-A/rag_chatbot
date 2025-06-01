from langchain_openai import OpenAIEmbeddings
from process_documents import load_documents, chunk_documents


def prepare_document_chunks(directory, chunk_size=1000, chunk_overlap=20):
    """
    Loads and chunks documents from a directory.

    Returns a list of FileDocument objects with chunked content.
    """

    raw_documents = load_documents(directory)

    chunked_documents = chunk_documents(raw_documents, chunk_size, chunk_overlap)

    return chunked_documents


def create_embedded_chunks(chunked_documents, model_name="text-embedding-3-small"):
    """
    Takes FileDocument list, returns (vectors, metadata).

    vectors: List of embedding vectors metadata: Mapping of index -> (text, metadata)
    """
    if not chunked_documents:
        raise ValueError("No chunked documents provided.")

    embedding_model = OpenAIEmbeddings(model=model_name)

    texts = [doc.content for doc in chunked_documents]

    vectors = embedding_model.embed_documents(texts)

    metadata = {
        idx: {"text": doc.content, "meta": doc.metadata}
        for idx, doc in enumerate(chunked_documents)
    }

    return vectors, metadata
