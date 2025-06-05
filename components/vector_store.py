import pickle
import os
import faiss
import numpy as np

from config import METADATA_PATH, INDEX_PATH
from utils.logger import get_logger

logger = get_logger("vector_store_logger")


def save_faiss_index(vectors, metadata, index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """
    Saves vectors and metadata to disk using FAISS + Pickle.
    """

    if not vectors:
        logger.error("The vectors list is empty. Cannot save FAISS index.")
        raise ValueError("The vectors list is empty. Cannot save FAISS index.")

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))

    # Save index to disk
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    # Save metadata using pickle
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("The vectors and metadata have been saved to disk.")


def load_faiss_index(index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """
    Loads FAISS index and metadata from disk.
    """
    if not os.path.exists(index_path):
        logger.error(f"FAISS index not found at {index_path}")
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found at {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    logger.info("The FAISS index and metadata have been loaded from disk.")

    return index, metadata
