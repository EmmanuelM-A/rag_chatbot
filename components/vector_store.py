"""
Responsible for loading and storing vectors.
"""

import pickle
import os
from typing import Dict, Any, Optional

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

    def save_faiss_index(self, vectors, metadata,
                         existing_index: Optional[faiss.Index] = None) -> None:
        """
        Saves vectors and metadata to disk using FAISS + Pickle.
        Can either create new index or update existing one.

        Args:
            vectors: List of vectors to save (can be None if using existing_index)
            metadata: Metadata dictionary to save
            existing_index: Optional existing FAISS index to save directly
        """

        if existing_index is not None:
            # Save existing index that was already updated
            index = existing_index
        else:
            # Create new index from vectors
            if not vectors:
                logger.error(
                    "The vectors list is empty. Cannot save FAISS index.")
                raise ValueError(
                    "The vectors list is empty. Cannot save FAISS index.")

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

        Returns:
            Tuple of (index, metadata)
        """

        logger.debug("Loading FAISS index and metadata from disk.")

        if not os.path.exists(self.index_path):
            logger.critical(f"FAISS index not found at {self.index_path}")
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}")

        if not os.path.exists(self.metadata_path):
            logger.critical(f"Metadata file not found at {self.metadata_path}")
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_path}")

        index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)

        logger.debug(
            "The FAISS index and metadata have been loaded from disk.")

        return index, metadata

    def add_vectors_to_index(self, new_vectors,
                             new_metadata: Dict[int, Dict[str, Any]]) -> None:
        """
        Add new vectors to existing index and update metadata.

        Args:
            new_vectors: List of new vectors to add
            new_metadata: Dictionary of new metadata entries
        """

        if not new_vectors:
            logger.warning("No new vectors to add")
            return

        try:
            # Load existing index and metadata
            index, existing_metadata = self.load_faiss_index()

            # Add new vectors to index
            index.add(np.array(new_vectors).astype("float32"))

            # Merge metadata (adjust indices for new vectors)
            offset = len(existing_metadata)
            for idx, data in new_metadata.items():
                existing_metadata[offset + idx] = data

            # Save updated index and metadata
            self.save_faiss_index(None, existing_metadata,
                                  existing_index=index)

            logger.info(
                f"Added {len(new_vectors)} new vectors to existing index")

        except FileNotFoundError:
            # If no existing index, create new one
            logger.info(
                "No existing index found, creating new one with new vectors")
            self.save_faiss_index(new_vectors, new_metadata)

    def index_exists(self) -> bool:
        """
        Check if both index and metadata files exist.

        Returns:
            True if both files exist, False otherwise
        """
        return (os.path.exists(self.index_path) and
                os.path.exists(self.metadata_path))

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dictionary with index statistics
        """
        try:
            if not self.index_exists():
                return {"exists": False, "total_vectors": 0}

            index, metadata = self.load_faiss_index()

            return {
                "exists": True,
                "total_vectors": index.ntotal,
                "dimension": index.d,
                "metadata_entries": len(metadata)
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"exists": False, "total_vectors": 0, "error": str(e)}
