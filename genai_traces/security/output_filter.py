"""
Output filtering and post-generation safety checks.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import re


@dataclass
class FilterResult:
    """Result of output filtering."""
    
    passed: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    filtered_output: Optional[str] = None
    action_taken: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "action_taken": self.action_taken,
        }


class OutputFilter:
    """
    Filters LLM outputs for safety and compliance.
    
    Checks:
    - PII leakage
    - Secret/credential exposure
    - Blocked topics/keywords
    - Format compliance
    """
    
    def __init__(
        self,
        block_pii: bool = True,
        block_secrets: bool = True,
        blocked_topics: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        action: str = "block",
    ):
        """
        Initialize the output filter.
        
        Args:
            block_pii: Whether to block PII in outputs
            block_secrets: Whether to block secrets/credentials
            blocked_topics: List of blocked topic keywords
            blocked_patterns: List of regex patterns to block
            action: Action on violation (block, redact, flag)
        """
        self.block_pii = block_pii
        self.block_secrets = block_secrets
        self.blocked_topics = set(blocked_topics or [])
        self.blocked_patterns = [re.compile(p, re.I) for p in (blocked_patterns or [])]
        self.action = action
        
        self._secret_patterns = [
            re.compile(r'(?:api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})', re.I),
            re.compile(r'(?:password|passwd|pwd)["\s:=]+["\']?([^\s"\']{8,})', re.I),
            re.compile(r'(?:secret|token)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})', re.I),
            re.compile(r'(?:aws_access_key_id)["\s:=]+["\']?([A-Z0-9]{20})', re.I),
            re.compile(r'(?:aws_secret_access_key)["\s:=]+["\']?([a-zA-Z0-9/+=]{40})', re.I),
            re.compile(r'sk-[a-zA-Z0-9]{48}'),
            re.compile(r'ghp_[a-zA-Z0-9]{36}'),
            re.compile(r'gho_[a-zA-Z0-9]{36}'),
        ]
    
    def filter(self, output: str) -> FilterResult:
        """
        Filter an LLM output.
        
        Args:
            output: The output text to filter
            
        Returns:
            FilterResult with pass/fail and any violations
        """
        violations = []
        
        if self.block_secrets:
            secret_violations = self._check_secrets(output)
            violations.extend(secret_violations)
        
        if self.block_pii:
            pii_violations = self._check_pii(output)
            violations.extend(pii_violations)
        
        topic_violations = self._check_blocked_topics(output)
        violations.extend(topic_violations)
        
        pattern_violations = self._check_blocked_patterns(output)
        violations.extend(pattern_violations)
        
        if not violations:
            return FilterResult(passed=True)
        
        if self.action == "block":
            return FilterResult(
                passed=False,
                violations=violations,
                action_taken="blocked",
            )
        
        elif self.action == "redact":
            filtered = self._redact_violations(output, violations)
            return FilterResult(
                passed=True,
                violations=violations,
                filtered_output=filtered,
                action_taken="redacted",
            )
        
        else:
            return FilterResult(
                passed=True,
                violations=violations,
                action_taken="flagged",
            )
    
    def _check_secrets(self, text: str) -> List[Dict[str, Any]]:
        """Check for exposed secrets."""
        violations = []
        
        for pattern in self._secret_patterns:
            matches = pattern.findall(text)
            for match in matches:
                violations.append({
                    "type": "secret_exposure",
                    "pattern": pattern.pattern[:50],
                    "severity": "high",
                })
        
        return violations
    
    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """Check for PII leakage."""
        from ..privacy.detection.pii_detector import PIIDetector
        
        detector = PIIDetector()
        matches = detector.detect(text)
        
        violations = []
        for match in matches:
            violations.append({
                "type": "pii_leakage",
                "pii_type": match.pii_type,
                "severity": "medium",
            })
        
        return violations
    
    def _check_blocked_topics(self, text: str) -> List[Dict[str, Any]]:
        """Check for blocked topics."""
        violations = []
        text_lower = text.lower()
        
        for topic in self.blocked_topics:
            if topic.lower() in text_lower:
                violations.append({
                    "type": "blocked_topic",
                    "topic": topic,
                    "severity": "medium",
                })
        
        return violations
    
    def _check_blocked_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Check for blocked patterns."""
        violations = []
        
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                violations.append({
                    "type": "blocked_pattern",
                    "pattern": pattern.pattern[:50],
                    "severity": "medium",
                })
        
        return violations
    
    def _redact_violations(
        self,
        text: str,
        violations: List[Dict[str, Any]],
    ) -> str:
        """Redact violations from text."""
        result = text
        
        for pattern in self._secret_patterns:
            result = pattern.sub("[REDACTED]", result)
        
        from ..privacy.redaction.redactor import Redactor
        redactor = Redactor(strategy="template")
        
        from ..privacy.detection.pii_detector import PIIDetector
        detector = PIIDetector()
        matches = detector.detect(result)
        result = redactor.redact(result, matches)
        
        return result


class ContentModerator:
    """
    Content moderation for LLM outputs.
    """
    
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the content moderator.
        
        Args:
            categories: Categories to moderate (violence, hate, sexual, etc.)
            threshold: Score threshold for flagging
        """
        self.categories = categories or ["violence", "hate", "sexual", "self-harm"]
        self.threshold = threshold
    
    def moderate(self, text: str) -> Dict[str, Any]:
        """
        Moderate content.
        
        Returns:
            Dict with flagged categories and scores
        """
        results = {
            "flagged": False,
            "categories": {},
        }
        
        text_lower = text.lower()
        
        category_keywords = {
            "violence": ["kill", "murder", "attack", "hurt", "weapon", "gun", "knife"],
            "hate": ["hate", "racist", "sexist", "slur", "discriminate"],
            "sexual": ["explicit", "nude", "sexual", "porn"],
            "self-harm": ["suicide", "self-harm", "cut myself", "end my life"],
        }
        
        for category in self.categories:
            keywords = category_keywords.get(category, [])
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = min(1.0, matches / max(1, len(keywords)) * 2)
            
            results["categories"][category] = {
                "score": score,
                "flagged": score >= self.threshold,
            }
            
            if score >= self.threshold:
                results["flagged"] = True
        
        return results
