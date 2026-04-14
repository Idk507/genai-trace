"""
PII redaction utilities.
"""

import hashlib
from typing import List

from ..detection.pii_detector import PIIMatch


# Redaction templates by PII type
_TEMPLATES = {
    "credit_card": "****-****-****-****",
    "ssn": "***-**-****",
    "email": "[email redacted]",
    "phone_us": "[phone redacted]",
    "phone_intl": "[phone redacted]",
    "ip_address": "[ip redacted]",
    "ipv6_address": "[ip redacted]",
    "aws_key": "[aws_key redacted]",
    "aws_secret": "[aws_secret redacted]",
    "openai_key": "[api_key redacted]",
    "anthropic_key": "[api_key redacted]",
    "jwt": "[jwt redacted]",
    "api_key_generic": "[api_key redacted]",
    "date_of_birth": "[dob redacted]",
    "passport": "[passport redacted]",
    "iban": "[iban redacted]",
    "bitcoin_address": "[crypto_address redacted]",
    "ethereum_address": "[crypto_address redacted]",
}


class Redactor:
    """
    Redacts PII from text using various strategies.
    
    Strategies:
    - template: Replace with type-specific placeholder
    - partial: Partially mask (e.g., j***@e***.com)
    - hash: Replace with SHA-256 hash prefix
    
    Usage:
        from genai_traces.privacy import PIIDetector, Redactor
        
        detector = PIIDetector()
        redactor = Redactor()
        
        text = "Contact john@example.com or call 555-123-4567"
        matches = detector.detect(text)
        redacted = redactor.redact(text, matches, strategy="template")
        # "Contact [email redacted] or call [phone redacted]"
    """
    
    def __init__(self, salt: str = "genai-traces"):
        """
        Initialize the redactor.
        
        Args:
            salt: Salt for hash-based redaction
        """
        self.salt = salt
    
    def redact(
        self,
        text: str,
        matches: List[PIIMatch],
        strategy: str = "template"
    ) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Original text
            matches: List of PIIMatch objects from detector
            strategy: Redaction strategy (template, partial, hash)
            
        Returns:
            Redacted text
        """
        if not matches:
            return text
        
        if strategy == "hash":
            return self._hash_anonymize(text)
        
        result = text
        # Process in reverse order to preserve indices
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            if strategy == "partial":
                replacement = self._partial_redact(match)
            else:
                replacement = _TEMPLATES.get(match.type, "[redacted]")
            
            result = result[:match.start] + replacement + result[match.end:]
        
        return result
    
    def _partial_redact(self, match: PIIMatch) -> str:
        """Partially redact a PII match."""
        value = match.value
        pii_type = match.type
        
        if pii_type == "email":
            return self._partial_email(value)
        elif pii_type == "phone_us" or pii_type == "phone_intl":
            return self._partial_phone(value)
        elif pii_type == "credit_card":
            return self._partial_credit_card(value)
        elif pii_type == "ssn":
            return "***-**-" + value[-4:]
        else:
            # Generic partial: show first and last char
            if len(value) <= 4:
                return "*" * len(value)
            return value[0] + "*" * (len(value) - 2) + value[-1]
    
    def _partial_email(self, email: str) -> str:
        """Partially redact an email address."""
        parts = email.split("@")
        if len(parts) != 2:
            return "[email]"
        
        user = parts[0]
        domain_parts = parts[1].split(".")
        
        # Mask user part
        if len(user) <= 2:
            masked_user = "*" * len(user)
        else:
            masked_user = user[0] + "*" * (len(user) - 2) + user[-1]
        
        # Mask domain
        if len(domain_parts) >= 2:
            domain = domain_parts[0]
            if len(domain) <= 2:
                masked_domain = "*" * len(domain)
            else:
                masked_domain = domain[0] + "*" * (len(domain) - 1)
            tld = ".".join(domain_parts[1:])
            return f"{masked_user}@{masked_domain}.{tld}"
        
        return f"{masked_user}@***"
    
    def _partial_phone(self, phone: str) -> str:
        """Partially redact a phone number."""
        # Keep last 4 digits
        digits = "".join(c for c in phone if c.isdigit())
        if len(digits) >= 4:
            return "***-***-" + digits[-4:]
        return "[phone]"
    
    def _partial_credit_card(self, cc: str) -> str:
        """Partially redact a credit card number."""
        digits = "".join(c for c in cc if c.isdigit())
        if len(digits) >= 4:
            return "****-****-****-" + digits[-4:]
        return "****-****-****-****"
    
    def _hash_anonymize(self, text: str) -> str:
        """Hash the entire text for anonymization."""
        return hashlib.sha256((text + self.salt).encode()).hexdigest()[:16]
    
    def hash_value(self, value: str) -> str:
        """Hash a single value."""
        return hashlib.sha256((value + self.salt).encode()).hexdigest()[:12]
