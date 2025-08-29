"""
Embedding cache implementation to optimize RAG chatbot performance.
Supports separate caches for documents and queries, with proper error handling
for corrupted files and metadata.
"""

import json
import pickle
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import time

from src.components.config.settings import settings
from src.components.config.logger import get_logger
from src.utils.helper import generate_content_hash

logger = get_logger(__name__)


class EmbedderCache:
    """
    Manages caching of embeddings with separate subcaches for documents and queries.

    - Stores embeddings as `.pkl` files under `documents/` and `queries/`.
    - Tracks metadata per subcache in `cache_metadata.json`.
    - Prevents redundant embedding computation by reusing cached vectors.
    """

    def __init__(self) -> None:
        """
        Initialize the embedding cache.
        Creates directories and loads metadata for both caches.
        """
        self.base_cache_dir = Path(settings.vector.EMBEDDING_CACHE_DIR)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)

        # Two subdirectories: documents and queries
        self.subcaches = {
            "document": self.base_cache_dir / "documents",
            "query": self.base_cache_dir / "queries",
        }
        for path in self.subcaches.values():
            path.mkdir(parents=True, exist_ok=True)

        # Each kind has its own metadata file
        self.metadata_files = {
            kind: path / "cache_metadata.json"
            for kind, path in self.subcaches.items()
        }

        # Load metadata for both caches
        self.metadata = {
            kind: self._load_metadata(self.metadata_files[kind])
            for kind in self.subcaches
        }

    # --------------------- Metadata helpers ---------------------

    @staticmethod
    def _get_default_metadata() -> Dict[str, Any]:
        """
        Default metadata structure used when creating or resetting a cache.
        """
        return {
            "entries": {},  # hash -> {source, size, timestamp, embedding_file, last_accessed}
            "total_size": 0,
            "last_cleanup": time.time(),
        }

    def _load_metadata(self, metadata_file: Path) -> Dict[str, Any]:
        """
        Load cache metadata from disk with error handling.

        Args:
            metadata_file: Path to metadata JSON file.

        Returns:
            Dict[str, Any]: Metadata dictionary.

        Raises:
            json.JSONDecodeError: If file exists but cannot be parsed.
            OSError: If file operations fail.
        """
        if not metadata_file.exists() or metadata_file.stat().st_size == 0:
            logger.debug(f"Metadata file {metadata_file} missing or empty, creating new.")
            return self._get_default_metadata()

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if not isinstance(metadata, dict) or "entries" not in metadata:
                logger.warning(f"Invalid metadata structure in {metadata_file}, resetting.")
                return self._get_default_metadata()

            # Ensure required keys exist
            default = self._get_default_metadata()
            for k, v in default.items():
                if k not in metadata:
                    metadata[k] = v

            return metadata

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted metadata file {metadata_file}: {e}, resetting.")
            # Backup corrupted file
            try:
                backup_file = metadata_file.with_suffix(".json.corrupted")
                metadata_file.rename(backup_file)
                logger.info(f"Backed up corrupted metadata to {backup_file}")
            except OSError as backup_error:
                logger.error(f"Failed to backup corrupted metadata: {backup_error}")
            return self._get_default_metadata()

        except OSError as e:
            logger.error(f"OS error while loading metadata {metadata_file}: {e}")
            return self._get_default_metadata()

    def _save_metadata(self, kind: str) -> None:
        """
        Save metadata for a given kind atomically.

        Args:
            kind: "document" or "query".
        """
        metadata_file = self.metadata_files[kind]
        temp_file = metadata_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata[kind], f, indent=4)
            temp_file.replace(metadata_file)
            logger.debug(f"{kind} metadata saved successfully.")
        except OSError as e:
            logger.error(f"Failed to save {kind} metadata: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    def _get_cache_file_path(self, kind: str, content_hash: str) -> Path:
        """Return cache file path for a given kind and content hash."""
        return self.subcaches[kind] / f"{content_hash}.pkl"

    # --------------------- Public API ---------------------

    def get_embedding(self, kind: str, content: str) -> Optional[List[float]]:
        """
        Retrieve an embedding from cache if it exists.

        Args:
            kind: Either "document" or "query".
            content: The text or query string.

        Returns:
            The embedding vector if cached, else None.
        """
        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            return None

        try:
            content_hash = generate_content_hash(f"{kind}:{content}")
            cache_file = self._get_cache_file_path(kind, content_hash)

            if content_hash in self.metadata[kind]["entries"] and cache_file.exists():
                with open(cache_file, "rb") as f:
                    embedding = pickle.load(f)
                self.metadata[kind]["entries"][content_hash]["last_accessed"] = time.time()
                return embedding

        except (OSError, pickle.PickleError) as e:
            logger.error(f"Error retrieving {kind} embedding: {e}")

        return None

    def store_embedding(
            self,
            kind: str,
            content: str,
            embedding: List[float],
            source: str = None
    ) -> None:
        """
        Store an embedding in cache.

        Args:
            kind: "document" or "query".
            content: Text or query string.
            embedding: Embedding vector.
            source: Optional identifier.

        Raises:
            OSError: If file writing fails.
            pickle.PickleError: If pickling fails.
        """
        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            return

        try:
            content_hash = generate_content_hash(f"{kind}:{content}")
            cache_file = self._get_cache_file_path(kind, content_hash)

            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)

            file_size = cache_file.stat().st_size
            self.metadata[kind]["entries"][content_hash] = {
                "source": source or "unknown",
                "type": kind,
                "size": file_size,
                "timestamp": time.time(),
                "last_accessed": time.time(),
                "embedding_file": cache_file.name,
            }
            self.metadata[kind]["total_size"] += file_size
            self._save_metadata(kind)

        except (OSError, pickle.PickleError) as e:
            logger.error(f"Failed to store {kind} embedding: {e}")

    def get_multiple_embeddings(
        self, kind: str, contents: List[str], sources: List[str] = None
    ) -> Tuple[List[Optional[List[float]]], List[str]]:
        """
        Retrieve multiple embeddings.

        Args:
            kind: "document" or "query".
            contents: List of text contents.
            sources: Optional list of identifiers.

        Returns:
            Tuple:
              - List of cached embeddings (None if missing).
              - List of contents that were not cached.
        """
        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            return [None] * len(contents), contents

        sources = sources or [None] * len(contents)
        cached, uncached = [], []

        for content, _ in zip(contents, sources):
            embedding = self.get_embedding(kind, content)
            cached.append(embedding)
            if embedding is None:
                uncached.append(content)

        return cached, uncached

    def store_multiple_embeddings(
            self,
            kind: str,
            contents: List[str],
            embeddings: List[List[float]], sources: List[str] = None
    ) -> None:
        """
        Store multiple embeddings in cache.

        Args:
            kind: "document" or "query".
            contents: List of text contents.
            embeddings: List of embedding vectors.
            sources: Optional list of identifiers.
        """
        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            return

        sources = sources or [None] * len(contents)
        for content, embedding, source in zip(contents, embeddings, sources):
            self.store_embedding(kind, content, embedding, source)

    # --------------------- Maintenance ---------------------

    def should_cleanup(self, kind: str) -> bool:
        """Check if a cache needs cleanup (size exceeded or stale)."""
        size_mb = self.metadata[kind]["total_size"] / (1024 * 1024)
        time_since_cleanup = time.time() - self.metadata[kind].get("last_cleanup", 0)
        return size_mb > settings.vector.MAX_CACHE_SIZE_MB or time_since_cleanup > 86400

    def cleanup_cache(self, kind: str) -> None:
        """Clean up oldest entries in a cache until below threshold."""
        logger.info(f"Cleaning up {kind} cache...")
        try:
            entries = list(self.metadata[kind]["entries"].items())
            entries.sort(key=lambda x: x[1].get("last_accessed", 0))

            target_size = settings.vector.MAX_CACHE_SIZE_MB * 0.8 * 1024 * 1024
            current_size = self.metadata[kind]["total_size"]
            removed = 0

            for content_hash, entry in entries:
                if current_size <= target_size:
                    break
                if self._remove_cache_entry(kind, content_hash):
                    current_size -= entry.get("size", 0)
                    removed += 1

            self.metadata[kind]["last_cleanup"] = time.time()
            logger.info(f"{kind} cache cleanup removed {removed} entries.")

        except OSError as e:
            logger.error(f"OS error during {kind} cache cleanup: {e}")

    def _remove_cache_entry(self, kind: str, content_hash: str) -> bool:
        """Remove a single entry from a cache."""
        try:
            cache_file = self._get_cache_file_path(kind, content_hash)
            if cache_file.exists():
                cache_file.unlink()

            if content_hash in self.metadata[kind]["entries"]:
                entry_size = self.metadata[kind]["entries"][content_hash].get("size", 0)
                del self.metadata[kind]["entries"][content_hash]
                self.metadata[kind]["total_size"] -= entry_size
                return True

        except OSError as e:
            logger.error(f"Failed to remove {kind} cache entry {content_hash}: {e}")
        return False

    def clear_cache(self, kind: str = None) -> None:
        """Clear one or both caches completely."""
        kinds = [kind] if kind else self.subcaches.keys()
        for k in kinds:
            logger.info(f"Clearing {k} cache...")
            try:
                for cache_file in self.subcaches[k].glob("*.pkl"):
                    cache_file.unlink(missing_ok=True)
                self.metadata[k] = self._get_default_metadata()
                self._save_metadata(k)
            except OSError as e:
                logger.error(f"Failed to clear {k} cache: {e}")

    def get_cache_stats(self, kind: str = None) -> Dict[str, Any]:
        """Return cache statistics for one or both caches."""
        if kind:
            size_mb = self.metadata[kind]["total_size"] / (1024 * 1024)
            return {
                "kind": kind,
                "total_entries": len(self.metadata[kind]["entries"]),
                "total_size_mb": round(size_mb, 2),
                "max_size_mb": settings.vector.MAX_CACHE_SIZE_MB,
                "cache_directory": str(self.subcaches[kind]),
                "enabled": settings.vector.EMBEDDING_CACHE_ENABLED,
            }
        return {k: self.get_cache_stats(k) for k in self.subcaches}

    def validate_cache(self, kind: str = None) -> Dict[str, Any]:
        """
        Validate cache integrity for one or both caches.

        Args:
            kind: "document", "query", or None for both.

        Returns:
            Report dictionary with counts of valid, missing, corrupted, and orphaned entries.
        """
        kinds = [kind] if kind else self.subcaches.keys()
        overall_report = {}

        for k in kinds:
            logger.info(f"Validating {k} cache...")
            report = {
                "total_entries": len(self.metadata[k]["entries"]),
                "valid_entries": 0,
                "missing_files": 0,
                "corrupted_files": 0,
                "orphaned_files": 0,
                "error": ""
            }

            try:
                valid_hashes = set()

                for content_hash, _ in list(self.metadata[k]["entries"].items()):
                    cache_file = self._get_cache_file_path(k, content_hash)
                    if not cache_file.exists():
                        logger.warning(f"[{k}] Missing file for hash {content_hash[:8]}")
                        report["missing_files"] += 1
                        del self.metadata[k]["entries"][content_hash]
                        continue
                    try:
                        with open(cache_file, "rb") as f:
                            pickle.load(f)
                        report["valid_entries"] += 1
                        valid_hashes.add(content_hash)
                    except (OSError, pickle.PickleError) as e:
                        logger.warning(f"[{k}] Corrupted file {cache_file.name}: {e}")
                        report["corrupted_files"] += 1
                        self._remove_cache_entry(k, content_hash)

                for cache_file in self.subcaches[k].glob("*.pkl"):
                    if cache_file.stem not in valid_hashes:
                        logger.warning(f"[{k}] Orphaned file {cache_file.name}")
                        report["orphaned_files"] += 1
                        try:
                            cache_file.unlink()
                        except OSError as e:
                            logger.error(f"[{k}] Failed to remove orphaned file: {e}")

                if report["missing_files"] > 0 or report["corrupted_files"] > 0:
                    self._save_metadata(k)

                logger.info(f"[{k}] Validation complete: {report}")

            except OSError as e:
                logger.error(f"OS error during {k} validation: {e}")
                report["error"] = str(e)

            overall_report[k] = report

        return overall_report
