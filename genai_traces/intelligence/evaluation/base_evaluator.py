"""
Base evaluator interface for automated quality evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.span import Span


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.
    
    Implement this to create custom evaluators for automated quality scoring.
    
    Usage:
        class MyEvaluator(BaseEvaluator):
            @property
            def name(self) -> str:
                return "my_evaluator"
            
            async def evaluate(self, span):
                score = my_scoring_fn(span.get_attribute("llm.completion"))
                return {"eval.my_score": score}
        
        # Register evaluator
        add_evaluator(MyEvaluator())
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the evaluator."""
        pass
    
    @abstractmethod
    async def evaluate(self, span: "Span") -> Dict[str, float]:
        """
        Evaluate a span and return scores.
        
        Args:
            span: The span to evaluate
            
        Returns:
            Dictionary of attribute_key → score (typically 0.0–1.0)
        """
        pass
    
    def should_evaluate(self, span: "Span") -> bool:
        """
        Determine if this evaluator should run on a span.
        
        Override to filter which spans this evaluator runs on.
        Default: only evaluate spans with LLM completion.
        
        Args:
            span: The span to check
            
        Returns:
            True if evaluator should run
        """
        return span.get_attribute("llm.completion") is not None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# Registry of evaluators
_evaluators: Dict[str, BaseEvaluator] = {}


def add_evaluator(evaluator: BaseEvaluator) -> None:
    """Register an evaluator."""
    _evaluators[evaluator.name] = evaluator


def get_evaluator(name: str) -> BaseEvaluator:
    """Get an evaluator by name."""
    return _evaluators.get(name)


def list_evaluators() -> list:
    """List all registered evaluators."""
    return list(_evaluators.keys())


async def run_evaluators(span: "Span", evaluator_names: list = None) -> Dict[str, float]:
    """
    Run evaluators on a span.
    
    Args:
        span: The span to evaluate
        evaluator_names: List of evaluator names to run (None = all)
        
    Returns:
        Combined dictionary of all evaluation results
    """
    results = {}
    
    evaluators_to_run = _evaluators.values()
    if evaluator_names:
        evaluators_to_run = [_evaluators[n] for n in evaluator_names if n in _evaluators]
    
    for evaluator in evaluators_to_run:
        if evaluator.should_evaluate(span):
            try:
                scores = await evaluator.evaluate(span)
                results.update(scores)
            except Exception:
                pass
    
    return results
