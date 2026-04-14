"""
Domain enforcement for GenAI-Traces.

Provides topic boundary enforcement to keep LLM responses on-topic.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import re


@dataclass
class DomainRule:
    """A rule for domain enforcement."""
    name: str
    allowed_topics: Set[str] = field(default_factory=set)
    blocked_topics: Set[str] = field(default_factory=set)
    allowed_keywords: Set[str] = field(default_factory=set)
    blocked_keywords: Set[str] = field(default_factory=set)
    allowed_patterns: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "allowed_topics": list(self.allowed_topics),
            "blocked_topics": list(self.blocked_topics),
            "allowed_keywords": list(self.allowed_keywords),
            "blocked_keywords": list(self.blocked_keywords),
            "allowed_patterns": self.allowed_patterns,
            "blocked_patterns": self.blocked_patterns,
        }


@dataclass
class DomainViolation:
    """Represents a domain violation."""
    rule_name: str
    violation_type: str
    matched_content: str
    severity: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "violation_type": self.violation_type,
            "matched_content": self.matched_content,
            "severity": self.severity,
        }


@dataclass
class DomainCheckResult:
    """Result of a domain check."""
    is_valid: bool
    violations: List[DomainViolation] = field(default_factory=list)
    matched_topics: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "matched_topics": list(self.matched_topics),
        }


class DomainEnforcer:
    """
    Enforces topic boundaries for LLM interactions.
    
    Usage:
        enforcer = DomainEnforcer()
        
        # Add a rule for a customer support bot
        enforcer.add_rule(DomainRule(
            name="customer_support",
            allowed_topics={"billing", "orders", "returns", "shipping"},
            blocked_topics={"politics", "religion", "adult_content"},
            blocked_keywords={"competitor_name", "lawsuit"},
        ))
        
        # Check content
        result = enforcer.check("How do I return my order?")
        if not result.is_valid:
            print("Off-topic content detected")
    """
    
    def __init__(self):
        self._rules: Dict[str, DomainRule] = {}
        self._topic_keywords: Dict[str, Set[str]] = {}
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
    
    def add_rule(self, rule: DomainRule) -> None:
        """Add a domain rule."""
        self._rules[rule.name] = rule
        
        self._compiled_patterns[rule.name] = {
            "allowed": [re.compile(p, re.IGNORECASE) for p in rule.allowed_patterns],
            "blocked": [re.compile(p, re.IGNORECASE) for p in rule.blocked_patterns],
        }
    
    def remove_rule(self, name: str) -> None:
        """Remove a domain rule."""
        self._rules.pop(name, None)
        self._compiled_patterns.pop(name, None)
    
    def define_topic(self, topic: str, keywords: List[str]) -> None:
        """Define keywords for a topic."""
        self._topic_keywords[topic] = set(kw.lower() for kw in keywords)
    
    def check(
        self,
        content: str,
        rule_names: Optional[List[str]] = None,
    ) -> DomainCheckResult:
        """
        Check content against domain rules.
        
        Args:
            content: The content to check
            rule_names: Optional list of rule names to check (default: all)
            
        Returns:
            DomainCheckResult with violations
        """
        violations = []
        matched_topics = set()
        
        content_lower = content.lower()
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        for topic, keywords in self._topic_keywords.items():
            if keywords & content_words:
                matched_topics.add(topic)
        
        rules_to_check = (
            [self._rules[n] for n in rule_names if n in self._rules]
            if rule_names
            else list(self._rules.values())
        )
        
        for rule in rules_to_check:
            rule_violations = self._check_rule(
                rule, content, content_lower, content_words, matched_topics
            )
            violations.extend(rule_violations)
        
        return DomainCheckResult(
            is_valid=len(violations) == 0,
            violations=violations,
            matched_topics=matched_topics,
        )
    
    def _check_rule(
        self,
        rule: DomainRule,
        content: str,
        content_lower: str,
        content_words: Set[str],
        matched_topics: Set[str],
    ) -> List[DomainViolation]:
        """Check content against a single rule."""
        violations = []
        
        for keyword in rule.blocked_keywords:
            if keyword.lower() in content_lower:
                violations.append(DomainViolation(
                    rule_name=rule.name,
                    violation_type="blocked_keyword",
                    matched_content=keyword,
                    severity="high",
                ))
        
        for topic in rule.blocked_topics:
            if topic in matched_topics:
                violations.append(DomainViolation(
                    rule_name=rule.name,
                    violation_type="blocked_topic",
                    matched_content=topic,
                    severity="high",
                ))
        
        patterns = self._compiled_patterns.get(rule.name, {})
        for pattern in patterns.get("blocked", []):
            match = pattern.search(content)
            if match:
                violations.append(DomainViolation(
                    rule_name=rule.name,
                    violation_type="blocked_pattern",
                    matched_content=match.group(),
                    severity="high",
                ))
        
        if rule.allowed_topics and matched_topics:
            off_topic = matched_topics - rule.allowed_topics
            for topic in off_topic:
                violations.append(DomainViolation(
                    rule_name=rule.name,
                    violation_type="off_topic",
                    matched_content=topic,
                    severity="medium",
                ))
        
        return violations
    
    def get_rules(self) -> List[DomainRule]:
        """Get all rules."""
        return list(self._rules.values())
    
    def get_rule(self, name: str) -> Optional[DomainRule]:
        """Get a rule by name."""
        return self._rules.get(name)


def create_customer_support_enforcer() -> DomainEnforcer:
    """Create a pre-configured enforcer for customer support."""
    enforcer = DomainEnforcer()
    
    enforcer.define_topic("billing", ["invoice", "payment", "charge", "refund", "price", "cost"])
    enforcer.define_topic("orders", ["order", "purchase", "buy", "cart", "checkout"])
    enforcer.define_topic("shipping", ["ship", "delivery", "track", "package", "arrive"])
    enforcer.define_topic("returns", ["return", "exchange", "refund", "warranty"])
    enforcer.define_topic("politics", ["election", "vote", "democrat", "republican", "president"])
    enforcer.define_topic("religion", ["god", "church", "prayer", "bible", "religious"])
    
    enforcer.add_rule(DomainRule(
        name="customer_support",
        allowed_topics={"billing", "orders", "shipping", "returns"},
        blocked_topics={"politics", "religion"},
    ))
    
    return enforcer
