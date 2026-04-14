"""
Privacy-safe content hashing for multi-modal inputs.
"""

import hashlib
from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class ContentHash:
    """A privacy-safe content hash."""
    
    hash_value: str
    algorithm: str = "sha256"
    truncated: bool = True
    original_size: int = 0
    
    def __str__(self) -> str:
        return self.hash_value


def hash_content(
    content: Union[bytes, str],
    algorithm: str = "sha256",
    length: int = 16,
    salt: Optional[str] = None,
) -> ContentHash:
    """
    Create a privacy-safe hash of content.
    
    Args:
        content: Content to hash (bytes or string)
        algorithm: Hash algorithm (sha256, sha1, md5)
        length: Length of truncated hash (0 for full hash)
        salt: Optional salt for additional privacy
        
    Returns:
        ContentHash object
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    original_size = len(content)
    
    if salt:
        content = salt.encode('utf-8') + content
    
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        hasher = hashlib.sha256()
    
    hasher.update(content)
    full_hash = hasher.hexdigest()
    
    if length > 0:
        hash_value = full_hash[:length]
        truncated = True
    else:
        hash_value = full_hash
        truncated = False
    
    return ContentHash(
        hash_value=hash_value,
        algorithm=algorithm,
        truncated=truncated,
        original_size=original_size,
    )


def hash_for_dedup(content: Union[bytes, str]) -> str:
    """
    Create a hash suitable for deduplication.
    
    Args:
        content: Content to hash
        
    Returns:
        32-character hash string
    """
    return hash_content(content, length=32).hash_value


def hash_for_privacy(
    content: Union[bytes, str],
    salt: Optional[str] = None,
) -> str:
    """
    Create a privacy-preserving hash with optional salt.
    
    Args:
        content: Content to hash
        salt: Optional salt (use a secret for better privacy)
        
    Returns:
        16-character hash string
    """
    return hash_content(content, length=16, salt=salt).hash_value


def verify_hash(
    content: Union[bytes, str],
    expected_hash: str,
    algorithm: str = "sha256",
    salt: Optional[str] = None,
) -> bool:
    """
    Verify content matches an expected hash.
    
    Args:
        content: Content to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used
        salt: Salt if one was used
        
    Returns:
        True if hash matches
    """
    computed = hash_content(
        content,
        algorithm=algorithm,
        length=len(expected_hash),
        salt=salt,
    )
    return computed.hash_value == expected_hash
