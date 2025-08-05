"""
Responsible for loading and storing vectors.
"""

import pickle
import os
import faiss
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Represents the communication service to access and store vectors.
    """

    def __init__(self, index_path, metadata_path):
        self.index_path = index_path
        self.metadata_path = metadata_path

    def save_faiss_index(self, vectors, metadata) -> None:
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
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)

        # Save metadata using pickle
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.debug("The vectors and metadata have been saved to disk.")

    def load_faiss_index(self):
        """
        Loads FAISS index and metadata from disk.
        """

        logger.debug("Loading FAISS index and metadata from disk.")

        if not os.path.exists(self.index_path):
            logger.critical(f"FAISS index not found at {self.index_path}")
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")

        if not os.path.exists(self.metadata_path):
            logger.critical(f"Metadata file not found at {self.metadata_path}")
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)

        logger.debug("The FAISS index and metadata have been loaded from disk.")

        return index, metadata
