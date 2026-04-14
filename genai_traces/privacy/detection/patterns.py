"""
PII detection patterns for GenAI-Traces.

Comprehensive regex patterns for detecting various types of PII.
"""

import re
from typing import Dict, Pattern

PII_PATTERNS: Dict[str, Pattern] = {
    "email": re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    
    "phone_us": re.compile(
        r'\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
    ),
    
    "phone_intl": re.compile(
        r'\b\+?[1-9]\d{1,14}\b'
    ),
    
    "ssn": re.compile(
        r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
    ),
    
    "credit_card": re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
    ),
    
    "credit_card_formatted": re.compile(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    ),
    
    "ip_address": re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    ),
    
    "ipv6": re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
    ),
    
    "date_of_birth": re.compile(
        r'\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12][0-9]|3[01])[/\-](?:19|20)\d{2}\b'
    ),
    
    "passport": re.compile(
        r'\b[A-Z]{1,2}[0-9]{6,9}\b'
    ),
    
    "drivers_license": re.compile(
        r'\b[A-Z]{1,2}\d{5,8}\b'
    ),
    
    "bank_account": re.compile(
        r'\b\d{8,17}\b'
    ),
    
    "routing_number": re.compile(
        r'\b(?:0[1-9]|[1-4][0-9]|5[0-3])[0-9]{7}\b'
    ),
    
    "iban": re.compile(
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b'
    ),
    
    "medical_record": re.compile(
        r'\bMRN[-:\s]?\d{6,10}\b', re.I
    ),
    
    "health_insurance": re.compile(
        r'\b[A-Z]{3}\d{9}\b'
    ),
    
    "vin": re.compile(
        r'\b[A-HJ-NPR-Z0-9]{17}\b'
    ),
    
    "mac_address": re.compile(
        r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'
    ),
    
    "api_key_generic": re.compile(
        r'\b(?:api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.I
    ),
    
    "aws_access_key": re.compile(
        r'\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b'
    ),
    
    "aws_secret_key": re.compile(
        r'\b[A-Za-z0-9/+=]{40}\b'
    ),
    
    "openai_key": re.compile(
        r'\bsk-[a-zA-Z0-9]{48}\b'
    ),
    
    "github_token": re.compile(
        r'\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}\b'
    ),
    
    "stripe_key": re.compile(
        r'\b(?:sk|pk)_(?:test|live)_[a-zA-Z0-9]{24,}\b'
    ),
    
    "jwt_token": re.compile(
        r'\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b'
    ),
    
    "private_key": re.compile(
        r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'
    ),
    
    "address_us": re.compile(
        r'\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\.?\b', re.I
    ),
    
    "zip_code_us": re.compile(
        r'\b\d{5}(?:-\d{4})?\b'
    ),
    
    "postal_code_uk": re.compile(
        r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b', re.I
    ),
}


SENSITIVE_KEYWORDS = {
    "password", "passwd", "pwd", "secret", "token", "credential",
    "api_key", "apikey", "auth", "bearer", "private_key", "ssh_key",
    "access_token", "refresh_token", "session_id", "cookie",
}


def get_all_patterns() -> Dict[str, Pattern]:
    """Get all PII patterns."""
    return PII_PATTERNS.copy()


def get_patterns_by_category(category: str) -> Dict[str, Pattern]:
    """
    Get patterns by category.
    
    Categories: personal, financial, medical, technical, location, credential
    """
    categories = {
        "personal": ["email", "phone_us", "phone_intl", "ssn", "date_of_birth", 
                    "passport", "drivers_license"],
        "financial": ["credit_card", "credit_card_formatted", "bank_account", 
                     "routing_number", "iban"],
        "medical": ["medical_record", "health_insurance"],
        "technical": ["ip_address", "ipv6", "mac_address", "vin"],
        "location": ["address_us", "zip_code_us", "postal_code_uk"],
        "credential": ["api_key_generic", "aws_access_key", "aws_secret_key",
                      "openai_key", "github_token", "stripe_key", "jwt_token",
                      "private_key"],
    }
    
    pattern_names = categories.get(category, [])
    return {name: PII_PATTERNS[name] for name in pattern_names if name in PII_PATTERNS}
