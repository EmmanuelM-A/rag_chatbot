"""
Embedding cache implementation to optimize RAG chatbot performance.
Fixed version with proper error handling for corrupted cache files.
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
    Manages caching of document embeddings to avoid redundant API calls.
    Uses content-based hashing for cache keys to detect document changes.
    """

    def __init__(self) -> None:
        """
        Initialize the embedding cache.
        """

        self.base_cache_dir = Path(settings.vector.EMBEDDING_CACHE_DIR)

        # Create cache directory
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)

        # Two subdirectories: documents and queries
        self.subcaches = {
            "document": self.base_cache_dir / "documents",
            "query": self.base_cache_dir / "queries"
        }

        # Make sure both exist
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
        Get default metadata structure.
        """

        return {
            "entries": {}, # hash -> {source, size, timestamp, embedding_file, last_accessed}
            "total_size": 0,
            "last_cleanup": time.time()
        }

    def _load_metadata(self, metadata_file: Path) -> Dict[str, Any]:
        """
        Load cache metadata from disk with proper error handling.
        """

        if not metadata_file.exists() or metadata_file.stat().st_size == 0:
            logger.debug("Cache metadata file doesn't exist or is empty! "
                         "Creating new one...")
            return self._get_default_metadata()

        try:
            with open(metadata_file, 'r', encoding="utf-8") as f:
                metadata = json.load(f)

            # Validate metadata structure
            if not isinstance(metadata, dict) or "entries" not in metadata:
                logger.warning("Invalid cache metadata structure, resetting...")
                return self._get_default_metadata()

            # Ensure all required keys exist
            default_metadata = self._get_default_metadata()
            for key, value in default_metadata.items():
                if key not in metadata:
                    metadata[key] = value

            logger.debug(f"Loaded cache metadata with {len(metadata['entries'])} entries")
            return metadata

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Corrupted cache metadata file: {e}. Creating fresh metadata.")
            # Backup corrupted file
            try:
                backup_file = metadata_file.with_suffix('.json.corrupted')
                metadata_file.rename(backup_file)
                logger.info(f"Backed up corrupted metadata to {backup_file}")
            except Exception as backup_error:
                logger.error(f"Failed to backup corrupted file: {backup_error}")

            return self._get_default_metadata()

        except Exception as e:
            logger.error(f"Unexpected error loading cache metadata: {e}")
            return self._get_default_metadata()

    def _save_metadata(self, kind: str):
        """Save metadata for a given kind (document/query) atomically."""

        metadata_file = self.metadata_files[kind]

        # Write to temporary file first (atomic write)
        temp_file = metadata_file.with_suffix('.tmp')

        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata[kind], f, indent=4)

            # Atomic move
            temp_file.replace(metadata_file)
            logger.debug(f"{kind} metadata saved in cache successfully")
        except Exception as e:
            logger.error(f"Failed to save {kind} metadata: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _get_cache_file_path(self, kind: str, content_hash: str) -> Path:
        """
        Get cache file path for a content hash.
        """

        return self.subcaches[kind] / f"{content_hash}.pkl"

    def get_embedding(
        self,
        kind: str,
        content: str,
    ) -> Optional[List[float]]:
        """
        Retrieve embedding from cache if it exists.

        Args:
            kind: Either "document" or "query"
            content: Text content to check
        """

        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            logger.warning("Embedding cache is disabled!")
            return None

        try:
            content_hash = generate_content_hash(f"{kind}:{content}")
            cache_file = self._get_cache_file_path(kind, content_hash)

            if content_hash in self.metadata[kind][
                "entries"] and cache_file.exists():
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)

                # Update last access time
                self.metadata[kind]["entries"][content_hash][
                    "last_accessed"] = time.time()
                return embedding
        except Exception as e:
            logger.error(f"Error retrieving {kind} embedding from cache: {e}")
        return None

    def store_embedding(self, kind: str, content: str, embedding: List[float],
                        source: str = None) -> None:
        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            return
        try:
            content_hash = generate_content_hash(f"{kind}:{content}")
            cache_file = self._get_cache_file_path(kind, content_hash)

            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)

            file_size = cache_file.stat().st_size
            self.metadata[kind]["entries"][content_hash] = {
                "source": source or "unknown",
                "type": kind,
                "size": file_size,
                "timestamp": time.time(),
                "last_accessed": time.time(),
                "embedding_file": cache_file.name
            }
            self.metadata[kind]["total_size"] += file_size
            self._save_metadata(kind)
        except Exception as e:
            logger.error(f"Failed to cache {kind} embedding: {e}")

    def get_multiple_embeddings(self, kind: str, contents: List[str],
                                sources: List[str] = None):
        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            return [None] * len(contents), contents

        sources = sources or [None] * len(contents)
        cached_embeddings, uncached_contents = [], []

        for content, source in zip(contents, sources):
            embedding = self.get_embedding(kind, content, source)
            cached_embeddings.append(embedding)
            if embedding is None:
                uncached_contents.append(content)

        return cached_embeddings, uncached_contents

    def store_multiple_embeddings(self, kind: str, contents: List[str],
                                  embeddings: List[List[float]],
                                  sources: List[str] = None):
        if not settings.vector.EMBEDDING_CACHE_ENABLED:
            return

        sources = sources or [None] * len(contents)
        for content, embedding, source in zip(contents, embeddings, sources):
            self.store_embedding(kind, content, embedding, source)

    def _should_cleanup(self, kind: str) -> bool:
        """Check if cleanup is needed for a given kind."""
        size_mb = self.metadata[kind]["total_size"] / (1024 * 1024)
        time_since_cleanup = time.time() - self.metadata[kind].get(
            "last_cleanup", 0)
        return (
                    size_mb > settings.vector.MAX_CACHE_SIZE_MB or time_since_cleanup > 86400)

    def _cleanup_cache(self, kind: str):
        """Clean up old cache entries for a given kind to maintain size limit."""
        logger.info(f"Starting {kind} cache cleanup...")
        try:
            entries = list(self.metadata[kind]["entries"].items())
            entries.sort(
                key=lambda x: x[1].get("last_accessed", 0))  # oldest first

            target_size = settings.vector.MAX_CACHE_SIZE_MB * 0.8 * 1024 * 1024
            current_size = self.metadata[kind]["total_size"]
            removed_count = 0

            for content_hash, entry in entries:
                if current_size <= target_size:
                    break
                if self._remove_cache_entry(kind, content_hash):
                    current_size -= entry.get("size", 0)
                    removed_count += 1

            self.metadata[kind]["last_cleanup"] = time.time()
            logger.info(
                f"{kind} cache cleanup completed. Removed {removed_count} entries.")
        except Exception as e:
            logger.error(f"Error during {kind} cache cleanup: {e}")

    def _remove_cache_entry(self, kind: str, content_hash: str) -> bool:
        """Remove a cache entry for a given kind."""
        try:
            cache_file = self._get_cache_file_path(kind, content_hash)
            if cache_file.exists():
                cache_file.unlink()

            if content_hash in self.metadata[kind]["entries"]:
                entry_size = self.metadata[kind]["entries"][content_hash].get(
                    "size", 0)
                del self.metadata[kind]["entries"][content_hash]
                self.metadata[kind]["total_size"] -= entry_size
                return True
        except Exception as e:
            logger.error(
                f"Failed to remove {kind} cache entry {content_hash}: {e}")
        return False

    def clear_cache(self, kind: str = None):
        """
        Clear cache for one kind or all.
        """
        kinds = [kind] if kind else self.subcaches.keys()
        for k in kinds:
            logger.info(f"Clearing {k} cache...")
            try:
                for cache_file in self.subcaches[k].glob("*.pkl"):
                    cache_file.unlink(missing_ok=True)
                self.metadata[k] = self._get_default_metadata()
                self._save_metadata(k)
            except Exception as e:
                logger.error(f"Failed to clear {k} cache: {e}")

    def get_cache_stats(self, kind: str = None) -> Dict[str, Any]:
        """
        Return stats for one kind or all.
        """
        if kind:
            size_mb = self.metadata[kind]["total_size"] / (1024 * 1024)
            return {
                "kind": kind,
                "total_entries": len(self.metadata[kind]["entries"]),
                "total_size_mb": round(size_mb, 2),
                "max_size_mb": self.max_cache_size_mb,
                "cache_directory": str(self.subcaches[kind]),
                "enabled": settings.vector.EMBEDDING_CACHE_ENABLED
            }
        else:
            return {k: self.get_cache_stats(k) for k in self.subcaches}

    def validate_cache(self, kind: str = None) -> Dict[str, Any]:
        """
        Validate cache integrity for a given kind (document/query) or for all.
        Returns a report with counts of valid, missing, corrupted, and orphaned entries.
        """
        kinds = [kind] if kind else self.subcaches.keys()
        overall_report = {}

        for k in kinds:
            logger.info(f"Validating {k} cache integrity...")

            report = {
                "total_entries": len(self.metadata[k]["entries"]),
                "valid_entries": 0,
                "missing_files": 0,
                "corrupted_files": 0,
                "orphaned_files": 0
            }

            try:
                valid_hashes = set()

                # --- Check metadata entries ---
                for content_hash, entry in list(
                        self.metadata[k]["entries"].items()):
                    cache_file = self._get_cache_file_path(k, content_hash)

                    if not cache_file.exists():
                        logger.warning(
                            f"[{k}] Missing file for hash {content_hash[:8]}")
                        report["missing_files"] += 1
                        del self.metadata[k]["entries"][content_hash]
                        continue

                    try:
                        with open(cache_file, 'rb') as f:
                            pickle.load(f)  # ensure it's readable
                        report["valid_entries"] += 1
                        valid_hashes.add(content_hash)
                    except Exception as e:
                        logger.warning(
                            f"[{k}] Corrupted file {cache_file.name}: {e}")
                        report["corrupted_files"] += 1
                        self._remove_cache_entry(k, content_hash)

                # --- Check for orphaned files (files not in metadata) ---
                for cache_file in self.subcaches[k].glob("*.pkl"):
                    if cache_file.stem not in valid_hashes:
                        logger.warning(
                            f"[{k}] Orphaned cache file: {cache_file.name}")
                        report["orphaned_files"] += 1
                        try:
                            cache_file.unlink()
                        except Exception as e:
                            logger.error(
                                f"[{k}] Failed to remove orphaned file: {e}")

                # Save cleaned metadata if necessary
                if report["missing_files"] > 0 or report[
                    "corrupted_files"] > 0:
                    self._save_metadata(k)

                logger.info(f"[{k}] Cache validation complete: {report}")

            except Exception as e:
                logger.error(f"[{k}] Error during cache validation: {e}")
                report["error"] = str(e)

            overall_report[k] = report

        return overall_report if not kind else overall_report[k]
