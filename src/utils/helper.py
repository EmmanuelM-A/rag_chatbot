"""Helper functions and classes"""

import hashlib
import os


def does_file_exist(file_path) -> bool:
    """
    Determines if a path exists or not.
    """

    if os.path.exists(file_path):
        return True

    return False

def generate_content_hash(content: str) -> str:
    """
    Generate SHA-256 hash of content for cache key.
    """

    return hashlib.sha256(content.encode('utf-8')).hexdigest()
