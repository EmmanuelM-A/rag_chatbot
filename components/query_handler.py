from langchain_openai import OpenAIEmbeddings
import numpy as np


def search(query, index, metadata, embedding_model=None, k=3):
    """
        Embeds query, searches vector DB, returns top_k results.
        """
    if embedding_model is None:
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

    return results
