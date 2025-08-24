"""
Embedding cache implementation to optimize RAG chatbot performance.
"""

import json
import hashlib
import pickle
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import time

from src.components.config.settings import settings
from src.components.config.logger import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """
    Manages caching of document embeddings to avoid redundant API calls.
    Uses content-based hashing for cache keys to detect document changes.
    """

    def __init__(self, cache_dir: str = None, max_cache_size_mb: int = None):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache files
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir or settings.EMBEDDING_CACHE_DIR)
        self.max_cache_size_mb = max_cache_size_mb or settings.MAX_CACHE_SIZE_MB

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"

        # Load existing metadata
        self.metadata = self._load_metadata()

    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content for cache key."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")

        return {
            "entries": {},
            # hash -> {filename, size, timestamp, embedding_file}
            "total_size": 0,
            "last_cleanup": time.time()
        }

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_file_path(self, content_hash: str) -> Path:
        """Get cache file path for a content hash."""
        return self.cache_dir / f"{content_hash}.pkl"

    def get_embedding(self, content: str, source: str = None) -> Optional[
        List[float]]:
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

        content_hash = self._generate_content_hash(content)
        cache_file = self._get_cache_file_path(content_hash)

        if content_hash in self.metadata["entries"] and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)

                # Update access time
                self.metadata["entries"][content_hash][
                    "last_accessed"] = time.time()

                logger.debug(
                    f"Cache hit for content hash: {content_hash[:8]}...")
                return embedding

            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                # Clean up corrupted cache entry
                self._remove_cache_entry(content_hash)

        return None

    def store_embedding(self, content: str, embedding: List[float],
                        source: str = None):
        """
        Store embedding in cache.

        Args:
            content: Text content
            embedding: Embedding vector
            source: Source identifier (filename, URL, etc.)
        """
        if not settings.EMBEDDING_CACHE_ENABLED:
            return

        content_hash = self._generate_content_hash(content)
        cache_file = self._get_cache_file_path(content_hash)

        try:
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
                f"Cached embedding for content hash: {content_hash[:8]}...")

            # Check if cleanup is needed
            if self._should_cleanup():
                self._cleanup_cache()

            self._save_metadata()

        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")

    def get_multiple_embeddings(self, contents: List[str],
                                sources: List[str] = None) -> Tuple[
        List[Optional[List[float]]], List[str]]:
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
        logger.info(
            f"Cache hits: {cache_hits}/{len(contents)} ({cache_hits / len(contents) * 100:.1f}%)")

        return cached_embeddings, uncached_contents

    def store_multiple_embeddings(self, contents: List[str],
                                  embeddings: List[List[float]],
                                  sources: List[str] = None):
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

        return (size_mb > self.max_cache_size_mb or
                time_since_cleanup > 86400)  # Cleanup daily

    def _cleanup_cache(self):
        """Clean up old cache entries to maintain size limit."""
        logger.info("Starting cache cleanup...")

        # Sort entries by last access time (oldest first)
        entries = list(self.metadata["entries"].items())
        entries.sort(key=lambda x: x[1]["last_accessed"])

        target_size = self.max_cache_size_mb * 0.8 * 1024 * 1024  # 80% of max
        current_size = self.metadata["total_size"]

        removed_count = 0

        for content_hash, entry in entries:
            if current_size <= target_size:
                break

            if self._remove_cache_entry(content_hash):
                current_size -= entry["size"]
                removed_count += 1

        self.metadata["last_cleanup"] = time.time()
        logger.info(
            f"Cache cleanup completed. Removed {removed_count} entries.")

    def _remove_cache_entry(self, content_hash: str) -> bool:
        """Remove a cache entry."""
        try:
            cache_file = self._get_cache_file_path(content_hash)
            if cache_file.exists():
                cache_file.unlink()

            if content_hash in self.metadata["entries"]:
                entry_size = self.metadata["entries"][content_hash]["size"]
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
                cache_file.unlink()

            # Reset metadata
            self.metadata = {
                "entries": {},
                "total_size": 0,
                "last_cleanup": time.time()
            }

            self._save_metadata()
            logger.info("Cache cleared successfully.")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        size_mb = self.metadata["total_size"] / (1024 * 1024)

        return {
            "total_entries": len(self.metadata["entries"]),
            "total_size_mb": round(size_mb, 2),
            "max_size_mb": self.max_cache_size_mb,
            "cache_directory": str(self.cache_dir),
            "enabled": settings.EMBEDDING_CACHE_ENABLED
        }
