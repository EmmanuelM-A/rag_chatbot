from langchain_openai import OpenAIEmbeddings
import numpy as np
from utils.logger import get_logger

logger = get_logger("query_handler_logger")


def search(query, index, metadata, embedding_model=None, k=3):
    """
        Embeds query, searches vector DB, returns top_k results.
        """
    if embedding_model is None:
        logger.error("The default embedding model: text-embedding-3-small has been set.")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    query_vector = embedding_model.embed_query(query)

    D, I = index.search(np.array([query_vector]).astype("float32"), k)

    results = []

    for i in I[0]:
        entry = metadata[i]
        results.append({
            "text": entry["text"],
            "metadata": entry["meta"]
        })

    logger.info("Query embedded and results found.")

    return results
