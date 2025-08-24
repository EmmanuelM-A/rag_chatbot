"""
Embedding cache implementation to optimize RAG chatbot performance.
Fixed version with proper error handling for corrupted cache files.
"""

import json
import hashlib
import pickle
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import time

from src.components.config.settings import settings
from src.components.config.logger import get_logger
from src.utils.helper import generate_content_hash

logger = get_logger(__name__)


class EmbeddingCache:
    """
    Manages caching of document embeddings to avoid redundant API calls.
    Uses content-based hashing for cache keys to detect document changes.
    """

    def __init__(
        self,
        cache_dir: str = settings.EMBEDDING_CACHE_DIR,
        max_cache_size_mb: int = settings.MAX_CACHE_SIZE_MB
    ) -> None:
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache files
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_mb = max_cache_size_mb

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"

        # Load existing metadata
        self.metadata = self._load_metadata()

    def _get_default_metadata(self) -> Dict[str, Any]:
        """Get default metadata structure."""
        return {
            "entries": {},
            # hash -> {source, size, timestamp, embedding_file, last_accessed}
            "total_size": 0,
            "last_cleanup": time.time()
        }

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load cache metadata from disk with proper error handling.
        """
        if not self.metadata_file.exists():
            logger.debug("Cache metadata file doesn't exist, creating new one")
            return self._get_default_metadata()

        try:
            # Check if file is empty
            if self.metadata_file.stat().st_size == 0:
                logger.warning("Cache metadata file is empty, creating new metadata")
                return self._get_default_metadata()

            with open(self.metadata_file, 'r', encoding="utf-8") as f:
                metadata = json.load(f)

            # Validate metadata structure
            if not isinstance(metadata, dict) or "entries" not in metadata:
                logger.warning("Invalid cache metadata structure, resetting")
                return self._get_default_metadata()

            # Ensure all required keys exist
            default_metadata = self._get_default_metadata()
            for key in default_metadata:
                if key not in metadata:
                    metadata[key] = default_metadata[key]

            logger.debug(f"Loaded cache metadata with {len(metadata['entries'])} entries")
            return metadata

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Corrupted cache metadata file: {e}. Creating fresh metadata.")
            # Backup corrupted file
            try:
                backup_file = self.metadata_file.with_suffix('.json.corrupted')
                self.metadata_file.rename(backup_file)
                logger.info(f"Backed up corrupted metadata to {backup_file}")
            except Exception as backup_error:
                logger.error(f"Failed to backup corrupted file: {backup_error}")

            return self._get_default_metadata()

        except Exception as e:
            logger.error(f"Unexpected error loading cache metadata: {e}")
            return self._get_default_metadata()

    def _save_metadata(self):
        """Save cache metadata to disk with atomic write."""
        try:
            # Write to temporary file first (atomic write)
            temp_file = self.metadata_file.with_suffix('.tmp')

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)

            # Atomic move
            temp_file.replace(self.metadata_file)
            logger.debug("Cache metadata saved successfully")

        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            # Clean up temp file if it exists
            temp_file = self.metadata_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass

    def _get_cache_file_path(self, content_hash: str) -> Path:
        """
        Get cache file path for a content hash.
        """
        return self.cache_dir / f"{content_hash}.pkl"

    def get_embedding(self, content: str, source: str = None) -> Optional[List[float]]:
        """
        Retrieve embedding from cache if exists.

        Args:
            content: Text content to check
            source: Source identifier (filename, URL, etc.)

        Returns:
            Cached embedding vector or None if not found
        """
        if not settings.EMBEDDING_CACHE_ENABLED:
            return None

        try:
            content_hash = generate_content_hash(content)
            cache_file = self._get_cache_file_path(content_hash)

            if content_hash in self.metadata["entries"] and cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)

                    # Update access time
                    self.metadata["entries"][content_hash]["last_accessed"] = time.time()

                    # Log with source information for better debugging
                    cached_source = self.metadata["entries"][content_hash].get("source", "unknown")
                    logger.debug(
                        f"Cache hit for content from '{source or 'unknown'}' "
                        f"(originally cached from '{cached_source}'): {content_hash[:8]}..."
                    )
                    return embedding

                except (pickle.PickleError, EOFError) as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
                    # Clean up corrupted cache entry
                    self._remove_cache_entry(content_hash)

        except Exception as e:
            logger.error(f"Error retrieving embedding from cache: {e}")

        return None

    def store_embedding(self, content: str, embedding: List[float], source: str = None):
        """
        Store embedding in cache.

        Args:
            content: Text content
            embedding: Embedding vector
            source: Source identifier (filename, URL, etc.)
        """
        if not settings.EMBEDDING_CACHE_ENABLED:
            return

        try:
            content_hash = generate_content_hash(content)
            cache_file = self._get_cache_file_path(content_hash)

            # Save embedding
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)

            # Get file size
            file_size = cache_file.stat().st_size

            # Update metadata
            self.metadata["entries"][content_hash] = {
                "source": source or "unknown",
                "size": file_size,
                "timestamp": time.time(),
                "last_accessed": time.time(),
                "embedding_file": cache_file.name
            }

            self.metadata["total_size"] += file_size

            logger.debug(
                f"Cached embedding for content from '{source or 'unknown'}': {content_hash[:8]}..."
            )

            # Check if cleanup is needed
            if self._should_cleanup():
                self._cleanup_cache()

            self._save_metadata()

        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")

    def get_multiple_embeddings(self, contents: List[str], sources: List[str] = None) -> Tuple[List[Optional[List[float]]], List[str]]:
        """
        Get multiple embeddings from cache.

        Returns:
            Tuple of (cached_embeddings, uncached_contents)
            - cached_embeddings: List with embeddings or None for cache misses
            - uncached_contents: Contents that need to be embedded
        """
        if not settings.EMBEDDING_CACHE_ENABLED:
            return [None] * len(contents), contents

        sources = sources or [None] * len(contents)
        cached_embeddings = []
        uncached_contents = []

        for content, source in zip(contents, sources):
            embedding = self.get_embedding(content, source)
            cached_embeddings.append(embedding)
            if embedding is None:
                uncached_contents.append(content)

        cache_hits = sum(1 for e in cached_embeddings if e is not None)
        if contents:  # Avoid division by zero
            hit_rate = cache_hits / len(contents) * 100
            logger.info(f"Cache hits: {cache_hits}/{len(contents)} ({hit_rate:.1f}%)")

        return cached_embeddings, uncached_contents

    def store_multiple_embeddings(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        sources: List[str] = None
    ):
        """Store multiple embeddings in cache."""
        if not settings.EMBEDDING_CACHE_ENABLED:
            return

        sources = sources or [None] * len(contents)

        for content, embedding, source in zip(contents, embeddings, sources):
            self.store_embedding(content, embedding, source)

    def _should_cleanup(self) -> bool:
        """Check if cache cleanup is needed."""
        size_mb = self.metadata["total_size"] / (1024 * 1024)
        time_since_cleanup = time.time() - self.metadata.get("last_cleanup", 0)

        return (size_mb > self.max_cache_size_mb or time_since_cleanup > 86400)  # Cleanup daily

    def _cleanup_cache(self):
        """Clean up old cache entries to maintain size limit."""
        logger.info("Starting cache cleanup...")

        try:
            # Sort entries by last access time (oldest first)
            entries = list(self.metadata["entries"].items())
            entries.sort(key=lambda x: x[1].get("last_accessed", 0))

            target_size = self.max_cache_size_mb * 0.8 * 1024 * 1024  # 80% of max
            current_size = self.metadata["total_size"]

            removed_count = 0

            for content_hash, entry in entries:
                if current_size <= target_size:
                    break

                if self._remove_cache_entry(content_hash):
                    current_size -= entry.get("size", 0)
                    removed_count += 1

            self.metadata["last_cleanup"] = time.time()
            logger.info(f"Cache cleanup completed. Removed {removed_count} entries.")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def _remove_cache_entry(self, content_hash: str) -> bool:
        """Remove a cache entry."""
        try:
            cache_file = self._get_cache_file_path(content_hash)
            if cache_file.exists():
                cache_file.unlink()

            if content_hash in self.metadata["entries"]:
                entry_size = self.metadata["entries"][content_hash].get("size", 0)
                del self.metadata["entries"][content_hash]
                self.metadata["total_size"] -= entry_size
                return True

        except Exception as e:
            logger.error(f"Failed to remove cache entry {content_hash}: {e}")

        return False

    def clear_cache(self):
        """Clear all cache entries."""
        logger.info("Clearing embedding cache...")

        try:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")

            # Reset metadata
            self.metadata = self._get_default_metadata()
            self._save_metadata()
            logger.info("Cache cleared successfully.")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            size_mb = self.metadata["total_size"] / (1024 * 1024)

            return {
                "total_entries": len(self.metadata["entries"]),
                "total_size_mb": round(size_mb, 2),
                "max_size_mb": self.max_cache_size_mb,
                "cache_directory": str(self.cache_dir),
                "enabled": settings.EMBEDDING_CACHE_ENABLED
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "total_entries": 0,
                "total_size_mb": 0,
                "max_size_mb": self.max_cache_size_mb,
                "cache_directory": str(self.cache_dir),
                "enabled": settings.EMBEDDING_CACHE_ENABLED,
                "error": str(e)
            }

    def validate_cache(self) -> Dict[str, Any]:
        """Validate cache integrity and return report."""
        logger.info("Validating cache integrity...")

        report = {
            "total_entries": len(self.metadata["entries"]),
            "valid_entries": 0,
            "invalid_entries": 0,
            "missing_files": 0,
            "corrupted_files": 0,
            "orphaned_files": 0
        }

        try:
            # Check metadata entries
            valid_hashes = set()
            for content_hash, entry in list(self.metadata["entries"].items()):
                cache_file = self._get_cache_file_path(content_hash)

                if not cache_file.exists():
                    logger.warning(f"Missing cache file for hash {content_hash[:8]}")
                    report["missing_files"] += 1
                    # Remove from metadata
                    del self.metadata["entries"][content_hash]
                    continue

                try:
                    # Try to load the embedding
                    with open(cache_file, 'rb') as f:
                        pickle.load(f)
                    report["valid_entries"] += 1
                    valid_hashes.add(content_hash)

                except Exception as e:
                    logger.warning(f"Corrupted cache file {content_hash[:8]}: {e}")
                    report["corrupted_files"] += 1
                    self._remove_cache_entry(content_hash)

            # Check for orphaned files
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.stem not in valid_hashes:
                    logger.warning(f"Orphaned cache file: {cache_file.name}")
                    report["orphaned_files"] += 1
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.error(f"Failed to remove orphaned file: {e}")

            # Save cleaned metadata
            if report["missing_files"] > 0 or report["corrupted_files"] > 0:
                self._save_metadata()

            logger.info(f"Cache validation complete: {report}")
            return report

        except Exception as e:
            logger.error(f"Error during cache validation: {e}")
            report["error"] = str(e)
            return report