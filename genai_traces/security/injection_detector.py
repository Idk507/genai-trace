"""
Prompt injection and jailbreak detection.

Detects prompt injection and jailbreak attempts using:
1. Rule-based pattern matching (fast, zero dependencies)
2. Optional ML classifier (more accurate, requires transformers)

References OWASP LLM01:2025 and LLM07:2025.

Usage:
    detector = InjectionDetector()
    result = detector.check("Ignore previous instructions and...")
    if result.is_injection:
        raise SecurityError(f"Blocked: {result.injection_type.value}")
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..core.types import InjectionType


# High-confidence injection patterns
_INJECTION_PATTERNS = [
    # Instruction override attempts
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I), InjectionType.PROMPT_INJECTION),
    (re.compile(r"disregard\s+(all\s+)?prior\s+instructions?", re.I), InjectionType.PROMPT_INJECTION),
    (re.compile(r"forget\s+(all\s+)?previous\s+instructions?", re.I), InjectionType.PROMPT_INJECTION),
    (re.compile(r"override\s+(all\s+)?previous\s+instructions?", re.I), InjectionType.PROMPT_INJECTION),
    (re.compile(r"new\s+instructions?\s*:", re.I), InjectionType.PROMPT_INJECTION),
    
    # Jailbreak attempts
    (re.compile(r"you\s+are\s+now\s+(?:a|an)\s+(?!assistant|helpful)", re.I), InjectionType.JAILBREAK),
    (re.compile(r"pretend\s+you\s+(have\s+no\s+restrictions|are\s+evil|are\s+DAN|can\s+do\s+anything)", re.I), InjectionType.JAILBREAK),
    (re.compile(r"act\s+as\s+(?:if\s+)?you\s+(?:have\s+)?no\s+(?:ethical\s+)?(?:guidelines|restrictions|limits)", re.I), InjectionType.JAILBREAK),
    (re.compile(r"bypass\s+(?:your\s+)?(?:safety|content)\s+(?:filters?|guidelines?)", re.I), InjectionType.JAILBREAK),
    
    # DAN (Do Anything Now) patterns
    (re.compile(r"\bDAN\b.*\bjailbreak\b", re.I), InjectionType.DAN),
    (re.compile(r"do\s+anything\s+now", re.I), InjectionType.DAN),
    (re.compile(r"DAN\s+mode", re.I), InjectionType.DAN),
    (re.compile(r"developer\s+mode\s+enabled", re.I), InjectionType.DAN),
    
    # Data exfiltration attempts
    (re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.I), InjectionType.DATA_EXFILTRATION),
    (re.compile(r"print\s+(the\s+)?instructions\s+above", re.I), InjectionType.DATA_EXFILTRATION),
    (re.compile(r"show\s+(me\s+)?(your\s+)?(?:system\s+)?(?:prompt|instructions)", re.I), InjectionType.DATA_EXFILTRATION),
    (re.compile(r"what\s+(?:are\s+)?your\s+(?:system\s+)?instructions", re.I), InjectionType.DATA_EXFILTRATION),
    (re.compile(r"output\s+(?:your\s+)?(?:initial|original|system)\s+(?:prompt|instructions)", re.I), InjectionType.DATA_EXFILTRATION),
    (re.compile(r"exfiltrate\s+(all\s+)?data", re.I), InjectionType.DATA_EXFILTRATION),
    
    # Goal hijacking
    (re.compile(r"<\|im_start\|>|<\|endoftext\|>|\[INST\]|\[/INST\]", re.I), InjectionType.PROMPT_INJECTION),
    (re.compile(r"\\n\\nHuman:.*\\n\\nAssistant:", re.I), InjectionType.GOAL_HIJACKING),
    (re.compile(r"```system\s*\n", re.I), InjectionType.GOAL_HIJACKING),
    (re.compile(r"<\|system\|>", re.I), InjectionType.GOAL_HIJACKING),
    
    # Role manipulation
    (re.compile(r"from\s+now\s+on\s+you\s+(?:are|will)", re.I), InjectionType.JAILBREAK),
    (re.compile(r"your\s+new\s+(?:role|persona|identity)\s+is", re.I), InjectionType.JAILBREAK),
]


@dataclass
class InjectionResult:
    """Result of injection detection check."""
    is_injection: bool
    injection_type: InjectionType
    score: float  # 0.0–1.0 confidence
    matched_pattern: str = ""


class InjectionDetector:
    """
    Detects prompt injection and jailbreak attempts.
    
    Usage:
        detector = InjectionDetector()
        result = detector.check("Ignore all previous instructions...")
        if result.is_injection:
            print(f"Detected: {result.injection_type.value}")
    """
    
    def __init__(
        self,
        use_ml_classifier: bool = False,
        threshold: float = 0.7,
        custom_patterns: list = None,
    ):
        """
        Initialize the injection detector.
        
        Args:
            use_ml_classifier: Whether to use ML classifier (requires transformers)
            threshold: Confidence threshold for ML classifier
            custom_patterns: Additional custom patterns to check
        """
        self.use_ml = use_ml_classifier
        self.threshold = threshold
        self._classifier = None
        self._custom_patterns = custom_patterns or []
        
        if use_ml_classifier:
            self._load_classifier()
    
    def check(self, text: str) -> InjectionResult:
        """
        Check text for prompt injection attempts.
        
        Fast rule-based check first. Optionally falls through to ML classifier.
        
        Args:
            text: Text to check
            
        Returns:
            InjectionResult with detection details
        """
        if not text:
            return InjectionResult(
                is_injection=False,
                injection_type=InjectionType.NONE,
                score=0.0,
            )
        
        # Rule-based check (fast path)
        for pattern, injection_type in _INJECTION_PATTERNS:
            m = pattern.search(text)
            if m:
                return InjectionResult(
                    is_injection=True,
                    injection_type=injection_type,
                    score=0.95,
                    matched_pattern=m.group(),
                )
        
        # Check custom patterns
        for pattern_str, injection_type in self._custom_patterns:
            pattern = re.compile(pattern_str, re.I)
            m = pattern.search(text)
            if m:
                return InjectionResult(
                    is_injection=True,
                    injection_type=injection_type,
                    score=0.90,
                    matched_pattern=m.group(),
                )
        
        # Optional ML classifier check
        if self.use_ml and self._classifier:
            score = self._ml_score(text)
            if score > self.threshold:
                return InjectionResult(
                    is_injection=True,
                    injection_type=InjectionType.PROMPT_INJECTION,
                    score=score,
                )
        
        return InjectionResult(
            is_injection=False,
            injection_type=InjectionType.NONE,
            score=0.0,
        )
    
    def _load_classifier(self):
        """Load a lightweight ML classifier."""
        try:
            from transformers import pipeline
            self._classifier = pipeline(
                "text-classification",
                model="meta-llama/Prompt-Guard-86M",
                device=-1,  # CPU
                truncation=True,
                max_length=512,
            )
        except Exception:
            self._classifier = None
    
    def _ml_score(self, text: str) -> float:
        """Get injection score from ML classifier."""
        if not self._classifier:
            return 0.0
        
        try:
            result = self._classifier(text[:512])[0]
            # Check for injection label
            if result["label"].upper() in ("INJECTION", "JAILBREAK", "MALICIOUS"):
                return result["score"]
            return 1.0 - result["score"]
        except Exception:
            return 0.0
    
    def add_pattern(self, pattern: str, injection_type: InjectionType) -> None:
        """Add a custom detection pattern."""
        self._custom_patterns.append((pattern, injection_type))


class OutputGuardrail:
    """
    Post-generation safety checks on LLM output.
    
    Checks for:
    - PII leakage
    - Secret/key exposure
    - Blocked topics
    
    Usage:
        guardrail = OutputGuardrail()
        result = guardrail.check_output(llm_response, policy={"blocked_topics": ["competitor"]})
        if not result.passed:
            print(f"Violations: {result.violations}")
    """
    
    def __init__(self, max_retries: int = 2):
        """
        Initialize the output guardrail.
        
        Args:
            max_retries: Maximum retries for blocked output
        """
        self.max_retries = max_retries
        self._pii_detector = None
    
    def _get_pii_detector(self):
        """Lazy load PII detector."""
        if self._pii_detector is None:
            from ..privacy.detection.pii_detector import PIIDetector
            self._pii_detector = PIIDetector()
        return self._pii_detector
    
    def check_output(self, output: str, policy: dict = None) -> "OutputCheckResult":
        """
        Check LLM output for safety violations.
        
        Args:
            output: LLM output text
            policy: Policy configuration with blocked_topics, etc.
            
        Returns:
            OutputCheckResult with pass/fail and violations
        """
        violations = []
        policy = policy or {}
        
        # PII leak detection
        if self._get_pii_detector().contains_pii(output):
            violations.append("pii_in_output")
        
        # Blocked topics
        blocked_topics = policy.get("blocked_topics", [])
        for topic in blocked_topics:
            if topic.lower() in output.lower():
                violations.append(f"blocked_topic:{topic}")
        
        # Secret/key detection
        secret_patterns = [
            (r"(?:AKIA|ASIA)[A-Z0-9]{16}", "aws_key_in_output"),
            (r"sk-[A-Za-z0-9]{20,}", "openai_key_in_output"),
            (r"sk-ant-[A-Za-z0-9\-]{20,}", "anthropic_key_in_output"),
            (r"ghp_[A-Za-z0-9]{36}", "github_token_in_output"),
            (r"xox[baprs]-[A-Za-z0-9\-]{10,}", "slack_token_in_output"),
        ]
        
        for pattern, violation_type in secret_patterns:
            if re.search(pattern, output):
                violations.append(violation_type)
        
        return OutputCheckResult(
            passed=len(violations) == 0,
            violations=violations,
        )


@dataclass
class OutputCheckResult:
    """Result of output guardrail check."""
    passed: bool
    violations: list
