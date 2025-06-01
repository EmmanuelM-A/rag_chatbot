import pickle
import os
import faiss
import numpy as np

from utils.constants import METADATA_PATH, INDEX_PATH


def save_faiss_index(vectors, metadata, index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """
    Saves vectors and metadata to disk using FAISS + Pickle.
    """

    if not vectors:
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


def load_faiss_index(index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """
    Loads FAISS index and metadata from disk.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata
