import faiss
import pickle
import numpy as np
from utils.constants import METADATA_PATH, INDEX_PATH


def save_faiss_index(vectors, metadata, index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """
    Saves vectors and metadata to disk using FAISS + Pickle.
    """
    dimension = len(vectors[0])

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def load_faiss_index(index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    """
    Loads FAISS index and metadata from disk.
    """
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata
