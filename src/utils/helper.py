"""Helper functions and classes"""

import hashlib
import os

from src.components.config.logger import get_logger

logger = get_logger(__name__)


def does_file_exist(file_path) -> bool:
    """
    Determines if a path exists or not and prints an informative message as well.
    """

    if os.path.exists(file_path):
        logger.debug(f"The file {file_path} exists!")
        return True

    logger.debug(f"The file {file_path} does not exists!")
    return False

def generate_content_hash(content: str) -> str:
    """
    Generate SHA-256 hash of content for cache key.
    """

    return hashlib.sha256(content.encode('utf-8')).hexdigest()
