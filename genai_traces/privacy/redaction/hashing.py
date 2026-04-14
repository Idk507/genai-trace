"""
Hashing utilities for PII anonymization.

Provides SHA-256 based anonymization with optional salt.
"""

import hashlib
import hmac
import secrets
from typing import Optional


def hash_pii(
    value: str,
    salt: Optional[str] = None,
    algorithm: str = "sha256",
    length: Optional[int] = None,
) -> str:
    """
    Hash a PII value for anonymization.
    
    Args:
        value: The PII value to hash
        salt: Optional salt (recommended for security)
        algorithm: Hash algorithm (sha256, sha512, etc.)
        length: Optional length to truncate hash
        
    Returns:
        Hex-encoded hash
    """
    to_hash = value
    if salt:
        to_hash = salt + value
    
    hasher = hashlib.new(algorithm)
    hasher.update(to_hash.encode("utf-8"))
    result = hasher.hexdigest()
    
    if length:
        result = result[:length]
    
    return result


def hash_pii_hmac(
    value: str,
    key: str,
    algorithm: str = "sha256",
    length: Optional[int] = None,
) -> str:
    """
    Hash a PII value using HMAC for better security.
    
    Args:
        value: The PII value to hash
        key: Secret key for HMAC
        algorithm: Hash algorithm
        length: Optional length to truncate
        
    Returns:
        Hex-encoded HMAC
    """
    h = hmac.new(
        key.encode("utf-8"),
        value.encode("utf-8"),
        algorithm,
    )
    result = h.hexdigest()
    
    if length:
        result = result[:length]
    
    return result


def generate_salt(length: int = 32) -> str:
    """
    Generate a random salt for hashing.
    
    Args:
        length: Length of salt in bytes
        
    Returns:
        Hex-encoded salt
    """
    return secrets.token_hex(length)


def consistent_hash(
    value: str,
    namespace: str = "default",
    length: int = 16,
) -> str:
    """
    Generate a consistent hash for a value.
    
    Same value + namespace always produces same hash.
    
    Args:
        value: Value to hash
        namespace: Namespace for isolation
        length: Length of output hash
        
    Returns:
        Consistent hash
    """
    combined = f"{namespace}:{value}"
    return hashlib.sha256(combined.encode()).hexdigest()[:length]


class PIIHasher:
    """
    Hasher for PII anonymization with consistent salting.
    
    Usage:
        hasher = PIIHasher(salt="my-secret-salt")
        
        # Same email always produces same hash
        hash1 = hasher.hash("john@example.com", "email")
        hash2 = hasher.hash("john@example.com", "email")
        assert hash1 == hash2
    """
    
    def __init__(
        self,
        salt: Optional[str] = None,
        algorithm: str = "sha256",
        default_length: int = 16,
    ):
        self._salt = salt or generate_salt()
        self._algorithm = algorithm
        self._default_length = default_length
    
    def hash(
        self,
        value: str,
        pii_type: str = "generic",
        length: Optional[int] = None,
    ) -> str:
        """
        Hash a PII value.
        
        Args:
            value: Value to hash
            pii_type: Type of PII (for namespacing)
            length: Optional output length
            
        Returns:
            Hashed value
        """
        combined_salt = f"{self._salt}:{pii_type}"
        return hash_pii(
            value,
            salt=combined_salt,
            algorithm=self._algorithm,
            length=length or self._default_length,
        )
    
    def hash_dict(
        self,
        data: dict,
        fields: list,
    ) -> dict:
        """
        Hash specific fields in a dictionary.
        
        Args:
            data: Dictionary with data
            fields: List of field names to hash
            
        Returns:
            Dictionary with hashed fields
        """
        result = data.copy()
        
        for field in fields:
            if field in result:
                result[field] = self.hash(str(result[field]), field)
        
        return result
    
    def verify(
        self,
        value: str,
        hashed: str,
        pii_type: str = "generic",
    ) -> bool:
        """
        Verify a value against its hash.
        
        Args:
            value: Original value
            hashed: Hashed value to compare
            pii_type: Type of PII
            
        Returns:
            True if hash matches
        """
        computed = self.hash(value, pii_type, length=len(hashed))
        return hmac.compare_digest(computed, hashed)


def anonymize_email(email: str, hasher: Optional[PIIHasher] = None) -> str:
    """
    Anonymize an email address.
    
    Args:
        email: Email to anonymize
        hasher: Optional PIIHasher instance
        
    Returns:
        Anonymized email
    """
    if hasher is None:
        hasher = PIIHasher()
    
    parts = email.split("@")
    if len(parts) == 2:
        local_hash = hasher.hash(parts[0], "email_local", length=8)
        domain_hash = hasher.hash(parts[1], "email_domain", length=6)
        return f"{local_hash}@{domain_hash}.anon"
    
    return hasher.hash(email, "email")


def anonymize_phone(phone: str, hasher: Optional[PIIHasher] = None) -> str:
    """
    Anonymize a phone number.
    
    Args:
        phone: Phone number to anonymize
        hasher: Optional PIIHasher instance
        
    Returns:
        Anonymized phone
    """
    if hasher is None:
        hasher = PIIHasher()
    
    digits_only = "".join(c for c in phone if c.isdigit())
    return hasher.hash(digits_only, "phone", length=12)
