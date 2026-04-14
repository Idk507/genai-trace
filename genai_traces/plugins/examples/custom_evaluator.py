"""
Example custom evaluator plugin for GenAI-Traces.

Shows how to create a custom evaluator plugin.
"""

from typing import Dict, Any, Optional


class CustomEvaluator:
    """
    Example custom evaluator that checks response quality.
    
    Usage:
        from genai_traces.plugins.examples.custom_evaluator import CustomEvaluator
        from genai_traces.plugins import get_plugin_registry
        
        evaluator = CustomEvaluator()
        get_plugin_registry().register("evaluator", "custom", evaluator)
    """
    
    name = "custom_evaluator"
    version = "1.0.0"
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 5000,
        check_coherence: bool = True,
    ):
        self._min_length = min_length
        self._max_length = max_length
        self._check_coherence = check_coherence
    
    def evaluate(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a response.
        
        Args:
            prompt: The input prompt
            response: The LLM response
            context: Optional context
            reference: Optional reference answer
            
        Returns:
            Evaluation result with score and details
        """
        scores = {}
        issues = []
        
        length_score = self._check_length(response)
        scores["length"] = length_score
        if length_score < 0.5:
            issues.append("Response length outside acceptable range")
        
        if self._check_coherence:
            coherence_score = self._check_response_coherence(prompt, response)
            scores["coherence"] = coherence_score
            if coherence_score < 0.5:
                issues.append("Response may not be coherent with prompt")
        
        if reference:
            similarity_score = self._check_similarity(response, reference)
            scores["similarity"] = similarity_score
        
        overall = sum(scores.values()) / len(scores) if scores else 0.0
        
        return {
            "score": overall,
            "scores": scores,
            "issues": issues,
            "passed": overall >= 0.5,
        }
    
    def _check_length(self, response: str) -> float:
        """Check if response length is acceptable."""
        length = len(response)
        
        if length < self._min_length:
            return length / self._min_length
        elif length > self._max_length:
            return self._max_length / length
        else:
            return 1.0
    
    def _check_response_coherence(self, prompt: str, response: str) -> float:
        """Check if response is coherent with prompt."""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}
        prompt_words -= common_words
        response_words -= common_words
        
        if not prompt_words:
            return 1.0
        
        overlap = len(prompt_words & response_words)
        return min(1.0, overlap / (len(prompt_words) * 0.3))
    
    def _check_similarity(self, response: str, reference: str) -> float:
        """Check similarity with reference."""
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        if not reference_words:
            return 1.0
        
        intersection = len(response_words & reference_words)
        union = len(response_words | reference_words)
        
        return intersection / union if union > 0 else 0.0
    
    def should_evaluate(self, span: Any) -> bool:
        """Check if this evaluator should run on a span."""
        if hasattr(span, "span_type"):
            return span.span_type in ["llm", "LLM"]
        return True


def register_plugin():
    """Register this plugin with the registry."""
    from genai_traces.plugins import get_plugin_registry
    
    evaluator = CustomEvaluator()
    registry = get_plugin_registry()
    registry.register("evaluator", "custom", evaluator)
    
    return evaluator
