"""
Guardrail chain for composing multiple security checks.

Usage:
    guardrails = GuardrailChain(
        input_guards=[InjectionDetector(use_ml_classifier=True)],
        output_guards=[OutputGuardrail()],
        action="block",
    )
    
    # Check input
    guardrails.check_input(user_prompt)
    
    # Check output
    guardrails.check_output(llm_response, policy={"blocked_topics": ["competitor"]})
"""

from typing import List, Optional

from .injection_detector import InjectionDetector, OutputGuardrail
from ..core.context_manager import SecurityError


class GuardrailChain:
    """
    Compose multiple input and output guards into a single pipeline.
    
    Actions:
    - block: Raise SecurityError on violation
    - flag: Return result but don't raise
    - log: Log violation and continue
    """
    
    def __init__(
        self,
        input_guards: Optional[List] = None,
        output_guards: Optional[List] = None,
        action: str = "block",  # block | flag | log
    ):
        """
        Initialize the guardrail chain.
        
        Args:
            input_guards: List of input guard instances
            output_guards: List of output guard instances
            action: Action to take on violation
        """
        self.input_guards = input_guards or [InjectionDetector()]
        self.output_guards = output_guards or [OutputGuardrail()]
        self.action = action
    
    def check_input(self, text: str) -> dict:
        """
        Check input text against all input guards.
        
        Args:
            text: Input text to check
            
        Returns:
            Dict with 'passed' bool and 'findings' list
            
        Raises:
            SecurityError: If action is 'block' and violation detected
        """
        findings = []
        
        for guard in self.input_guards:
            if hasattr(guard, "check"):
                result = guard.check(text)
                if result.is_injection:
                    findings.append(result)
                    
                    if self.action == "block":
                        raise SecurityError(
                            f"Input blocked: {result.injection_type.value} "
                            f"(confidence: {result.score:.2f})"
                        )
                    elif self.action == "log":
                        self._log_violation("input", result)
        
        return {"passed": len(findings) == 0, "findings": findings}
    
    def check_output(self, text: str, policy: dict = None) -> dict:
        """
        Check output text against all output guards.
        
        Args:
            text: Output text to check
            policy: Policy configuration
            
        Returns:
            Dict with 'passed' bool and 'violations' list
            
        Raises:
            SecurityError: If action is 'block' and violation detected
        """
        violations = []
        
        for guard in self.output_guards:
            if hasattr(guard, "check_output"):
                result = guard.check_output(text, policy)
                if not result.passed:
                    violations.extend(result.violations)
                    
                    if self.action == "block":
                        raise SecurityError(f"Output blocked: {violations}")
                    elif self.action == "log":
                        self._log_violations("output", violations)
        
        return {"passed": len(violations) == 0, "violations": violations}
    
    def _log_violation(self, check_type: str, result):
        """Log a violation."""
        import logging
        logger = logging.getLogger("genai_traces.security")
        logger.warning(
            f"Security violation ({check_type}): "
            f"{result.injection_type.value} (score: {result.score:.2f})"
        )
    
    def _log_violations(self, check_type: str, violations: list):
        """Log violations."""
        import logging
        logger = logging.getLogger("genai_traces.security")
        logger.warning(f"Security violations ({check_type}): {violations}")
    
    def add_input_guard(self, guard) -> None:
        """Add an input guard."""
        self.input_guards.append(guard)
    
    def add_output_guard(self, guard) -> None:
        """Add an output guard."""
        self.output_guards.append(guard)
