"""
PII (Personally Identifiable Information) detection.
"""

import re
from dataclasses import dataclass
from typing import List, Set


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


# Ordered from most specific to least specific to avoid partial matches
_PATTERNS = {
    "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "phone_us": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "phone_intl": r"\b\+[1-9]\d{1,14}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "ipv6_address": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    "aws_key": r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "aws_secret": r"(?i)aws[_\-]?secret[_\-]?(?:access[_\-]?)?key['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})",
    "openai_key": r"sk-[A-Za-z0-9]{20,}",
    "anthropic_key": r"sk-ant-[A-Za-z0-9\-]{20,}",
    "jwt": r"eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+",
    "api_key_generic": r"(?i)(?:api[-_]?key|secret[-_]?key|access[-_]?token)['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9\-_.]{20,})",
    "date_of_birth": r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
    "passport": r"\b[A-Z]{1,2}[0-9]{6,9}\b",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b",
    "bitcoin_address": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
    "ethereum_address": r"\b0x[a-fA-F0-9]{40}\b",
}

_COMPILED = {k: re.compile(v) for k, v in _PATTERNS.items()}


class PIIDetector:
    """
    Detects PII in text using regex patterns.
    
    Usage:
        detector = PIIDetector()
        matches = detector.detect("Contact me at john@example.com")
        if detector.contains_pii("My SSN is 123-45-6789"):
            print("PII detected!")
    """
    
    def __init__(self, patterns: dict = None, sensitivity: str = "medium"):
        """
        Initialize the PII detector.
        
        Args:
            patterns: Custom patterns to use (overrides defaults)
            sensitivity: Detection sensitivity (low, medium, high)
        """
        self._patterns = _COMPILED.copy()
        if patterns:
            self._patterns.update({k: re.compile(v) for k, v in patterns.items()})
        self.sensitivity = sensitivity
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of PIIMatch objects
        """
        if not text:
            return []
        
        matches = []
        for pii_type, pattern in self._patterns.items():
            for m in pattern.finditer(text):
                matches.append(PIIMatch(
                    type=pii_type,
                    value=m.group(),
                    start=m.start(),
                    end=m.end(),
                    confidence=self._get_confidence(pii_type),
                ))
        
        # Sort by position for correct redaction ordering
        return sorted(matches, key=lambda x: x.start)
    
    def detect_types(self, text: str) -> Set[str]:
        """
        Get the types of PII detected in text.
        
        Args:
            text: Text to scan
            
        Returns:
            Set of PII type strings
        """
        return {m.type for m in self.detect(text)}
    
    def contains_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.
        
        Args:
            text: Text to check
            
        Returns:
            True if PII is detected
        """
        if not text:
            return False
        
        for pattern in self._patterns.values():
            if pattern.search(text):
                return True
        return False
    
    def _get_confidence(self, pii_type: str) -> float:
        """Get confidence score for a PII type."""
        # High confidence patterns
        high_confidence = {"credit_card", "ssn", "aws_key", "openai_key", "jwt"}
        if pii_type in high_confidence:
            return 0.95
        
        # Medium confidence
        medium_confidence = {"email", "phone_us", "ip_address"}
        if pii_type in medium_confidence:
            return 0.85
        
        return 0.75
    
    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a custom PII pattern."""
        self._patterns[name] = re.compile(pattern)
    
    def remove_pattern(self, name: str) -> None:
        """Remove a PII pattern."""
        self._patterns.pop(name, None)
