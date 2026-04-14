"""
AES-256-GCM field-level encryption for sensitive trace data.
"""

import os
import base64
import hashlib
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class EncryptedField:
    """An encrypted field value."""
    
    ciphertext: str
    nonce: str
    tag: str
    
    def to_string(self) -> str:
        """Serialize to string format."""
        return f"ENC:{self.nonce}:{self.tag}:{self.ciphertext}"
    
    @classmethod
    def from_string(cls, s: str) -> Optional["EncryptedField"]:
        """Deserialize from string format."""
        if not s.startswith("ENC:"):
            return None
        
        parts = s[4:].split(":")
        if len(parts) != 3:
            return None
        
        return cls(nonce=parts[0], tag=parts[1], ciphertext=parts[2])


class FieldEncryptor:
    """
    AES-256-GCM encryption for individual fields.
    
    Uses a key derived from a master key for each field type.
    """
    
    def __init__(
        self,
        master_key: Optional[str] = None,
        key_env_var: str = "GENAI_TRACES_ENCRYPTION_KEY",
    ):
        """
        Initialize the field encryptor.
        
        Args:
            master_key: Master encryption key (32 bytes, base64 encoded)
            key_env_var: Environment variable for key if not provided
        """
        if master_key:
            self._master_key = base64.b64decode(master_key)
        else:
            key_str = os.environ.get(key_env_var)
            if key_str:
                self._master_key = base64.b64decode(key_str)
            else:
                self._master_key = None
        
        self._crypto_available = False
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self._crypto_available = True
        except ImportError:
            pass
    
    def is_available(self) -> bool:
        """Check if encryption is available."""
        return self._crypto_available and self._master_key is not None
    
    def encrypt(
        self,
        plaintext: str,
        field_name: str = "default",
    ) -> str:
        """
        Encrypt a field value.
        
        Args:
            plaintext: Value to encrypt
            field_name: Field name for key derivation
            
        Returns:
            Encrypted string in format "ENC:nonce:tag:ciphertext"
        """
        if not self.is_available():
            return plaintext
        
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        field_key = self._derive_key(field_name)
        
        nonce = os.urandom(12)
        
        aesgcm = AESGCM(field_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]
        
        encrypted = EncryptedField(
            ciphertext=base64.b64encode(ciphertext).decode('ascii'),
            nonce=base64.b64encode(nonce).decode('ascii'),
            tag=base64.b64encode(tag).decode('ascii'),
        )
        
        return encrypted.to_string()
    
    def decrypt(
        self,
        encrypted_str: str,
        field_name: str = "default",
    ) -> str:
        """
        Decrypt a field value.
        
        Args:
            encrypted_str: Encrypted string from encrypt()
            field_name: Field name for key derivation
            
        Returns:
            Decrypted plaintext
        """
        if not encrypted_str.startswith("ENC:"):
            return encrypted_str
        
        if not self.is_available():
            return encrypted_str
        
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        encrypted = EncryptedField.from_string(encrypted_str)
        if not encrypted:
            return encrypted_str
        
        field_key = self._derive_key(field_name)
        
        nonce = base64.b64decode(encrypted.nonce)
        tag = base64.b64decode(encrypted.tag)
        ciphertext = base64.b64decode(encrypted.ciphertext)
        
        aesgcm = AESGCM(field_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext + tag, None)
        
        return plaintext.decode('utf-8')
    
    def _derive_key(self, field_name: str) -> bytes:
        """Derive a field-specific key from the master key."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            self._master_key,
            field_name.encode('utf-8'),
            100000,
            dklen=32
        )
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new master key."""
        key = os.urandom(32)
        return base64.b64encode(key).decode('ascii')


_encryptor: Optional[FieldEncryptor] = None


def get_encryptor() -> FieldEncryptor:
    """Get the global field encryptor."""
    global _encryptor
    if _encryptor is None:
        _encryptor = FieldEncryptor()
    return _encryptor


def encrypt_field(value: str, field_name: str = "default") -> str:
    """Convenience function to encrypt a field."""
    return get_encryptor().encrypt(value, field_name)


def decrypt_field(value: str, field_name: str = "default") -> str:
    """Convenience function to decrypt a field."""
    return get_encryptor().decrypt(value, field_name)
