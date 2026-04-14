"""
Redaction strategies for GenAI-Traces.

Provides full, partial, and hash-based redaction strategies.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass


class RedactionStrategy(Enum):
    """Available redaction strategies."""
    FULL = "full"
    PARTIAL = "partial"
    HASH = "hash"
    MASK = "mask"
    TEMPLATE = "template"


@dataclass
class RedactionConfig:
    """Configuration for redaction."""
    strategy: RedactionStrategy = RedactionStrategy.TEMPLATE
    hash_algorithm: str = "sha256"
    hash_length: int = 8
    mask_char: str = "*"
    partial_visible_chars: int = 4
    template: str = "[{type}]"


def redact_full(text: str, pii_type: str = "PII") -> str:
    """
    Fully redact text.
    
    Args:
        text: Text to redact
        pii_type: Type of PII for labeling
        
    Returns:
        Redacted string
    """
    return f"[{pii_type}]"


def redact_partial(
    text: str,
    visible_chars: int = 4,
    mask_char: str = "*",
) -> str:
    """
    Partially redact text, keeping some characters visible.
    
    Args:
        text: Text to redact
        visible_chars: Number of characters to keep visible
        mask_char: Character to use for masking
        
    Returns:
        Partially redacted string
    """
    if len(text) <= visible_chars:
        return mask_char * len(text)
    
    visible = text[:visible_chars]
    masked = mask_char * (len(text) - visible_chars)
    return visible + masked


def redact_hash(
    text: str,
    algorithm: str = "sha256",
    length: int = 8,
    salt: Optional[str] = None,
) -> str:
    """
    Redact text using a hash.
    
    Args:
        text: Text to redact
        algorithm: Hash algorithm
        length: Length of hash to keep
        salt: Optional salt for hashing
        
    Returns:
        Hashed string
    """
    to_hash = text
    if salt:
        to_hash = salt + text
    
    hasher = hashlib.new(algorithm)
    hasher.update(to_hash.encode("utf-8"))
    return hasher.hexdigest()[:length]


def redact_mask(text: str, mask_char: str = "*") -> str:
    """
    Mask text with a character.
    
    Args:
        text: Text to mask
        mask_char: Character to use
        
    Returns:
        Masked string
    """
    return mask_char * len(text)


def redact_template(text: str, pii_type: str, template: str = "[{type}]") -> str:
    """
    Redact using a template.
    
    Args:
        text: Text to redact
        pii_type: Type of PII
        template: Template string with {type} placeholder
        
    Returns:
        Templated redaction
    """
    return template.format(type=pii_type, text=text, length=len(text))


class StrategyRedactor:
    """
    Applies redaction strategies to text.
    
    Usage:
        redactor = StrategyRedactor(RedactionConfig(
            strategy=RedactionStrategy.PARTIAL,
            partial_visible_chars=4,
        ))
        
        result = redactor.redact("john@example.com", "EMAIL")
        # "john************"
    """
    
    def __init__(self, config: Optional[RedactionConfig] = None):
        self._config = config or RedactionConfig()
    
    def redact(self, text: str, pii_type: str = "PII") -> str:
        """
        Redact text using the configured strategy.
        
        Args:
            text: Text to redact
            pii_type: Type of PII
            
        Returns:
            Redacted string
        """
        strategy = self._config.strategy
        
        if strategy == RedactionStrategy.FULL:
            return redact_full(text, pii_type)
        
        elif strategy == RedactionStrategy.PARTIAL:
            return redact_partial(
                text,
                self._config.partial_visible_chars,
                self._config.mask_char,
            )
        
        elif strategy == RedactionStrategy.HASH:
            return redact_hash(
                text,
                self._config.hash_algorithm,
                self._config.hash_length,
            )
        
        elif strategy == RedactionStrategy.MASK:
            return redact_mask(text, self._config.mask_char)
        
        elif strategy == RedactionStrategy.TEMPLATE:
            return redact_template(text, pii_type, self._config.template)
        
        return text
    
    def redact_in_text(
        self,
        text: str,
        matches: List[Dict[str, Any]],
    ) -> str:
        """
        Redact multiple matches in text.
        
        Args:
            text: Full text
            matches: List of {start, end, type} dicts
            
        Returns:
            Text with redactions applied
        """
        matches_sorted = sorted(matches, key=lambda m: m["start"], reverse=True)
        
        result = text
        for match in matches_sorted:
            start = match["start"]
            end = match["end"]
            pii_type = match.get("type", "PII")
            original = text[start:end]
            redacted = self.redact(original, pii_type)
            result = result[:start] + redacted + result[end:]
        
        return result


def create_redactor(
    strategy: str = "template",
    **kwargs,
) -> StrategyRedactor:
    """
    Create a redactor with the specified strategy.
    
    Args:
        strategy: Strategy name
        **kwargs: Additional configuration
        
    Returns:
        Configured StrategyRedactor
    """
    config = RedactionConfig(
        strategy=RedactionStrategy(strategy),
        **kwargs,
    )
    return StrategyRedactor(config)
