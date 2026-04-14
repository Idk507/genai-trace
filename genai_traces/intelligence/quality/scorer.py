"""
Quality scoring for GenAI-Traces.

Provides composite quality scores from multiple evaluators.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CompositeScore:
    """
    Composite quality score from multiple evaluators.
    
    Attributes:
        overall_score: Weighted average of all dimension scores
        dimension_scores: Individual scores by dimension
        weights: Weights used for each dimension
        metadata: Additional scoring metadata
    """
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "weights": self.weights,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def get_grade(self) -> str:
        """Get letter grade based on overall score."""
        if self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.8:
            return "B"
        elif self.overall_score >= 0.7:
            return "C"
        elif self.overall_score >= 0.6:
            return "D"
        else:
            return "F"


class QualityScorer:
    """
    Computes composite quality scores from multiple evaluators.
    
    Usage:
        scorer = QualityScorer()
        scorer.add_evaluator("relevance", relevance_evaluator, weight=0.3)
        scorer.add_evaluator("coherence", coherence_evaluator, weight=0.3)
        scorer.add_evaluator("toxicity", toxicity_evaluator, weight=0.4)
        
        score = scorer.score(prompt, response, context)
        print(f"Overall: {score.overall_score:.2f}")
    """
    
    def __init__(self):
        self._evaluators: Dict[str, Any] = {}
        self._weights: Dict[str, float] = {}
        self._default_weight = 1.0
    
    def add_evaluator(
        self,
        name: str,
        evaluator: Any,
        weight: float = 1.0,
    ) -> None:
        """
        Add an evaluator to the scorer.
        
        Args:
            name: Name of the dimension
            evaluator: Evaluator instance with evaluate() method
            weight: Weight for this dimension (default 1.0)
        """
        self._evaluators[name] = evaluator
        self._weights[name] = weight
    
    def remove_evaluator(self, name: str) -> None:
        """Remove an evaluator."""
        self._evaluators.pop(name, None)
        self._weights.pop(name, None)
    
    def score(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> CompositeScore:
        """
        Compute composite quality score.
        
        Args:
            prompt: The input prompt
            response: The LLM response
            context: Optional context (for RAG)
            reference: Optional reference answer
            
        Returns:
            CompositeScore with overall and dimension scores
        """
        dimension_scores = {}
        
        for name, evaluator in self._evaluators.items():
            try:
                result = evaluator.evaluate(
                    prompt=prompt,
                    response=response,
                    context=context,
                    reference=reference,
                )
                
                if isinstance(result, dict):
                    score = result.get("score", 0.0)
                elif hasattr(result, "score"):
                    score = result.score
                else:
                    score = float(result)
                
                dimension_scores[name] = score
            except Exception as e:
                dimension_scores[name] = 0.0
        
        overall = self._compute_weighted_average(dimension_scores)
        
        return CompositeScore(
            overall_score=overall,
            dimension_scores=dimension_scores,
            weights=self._weights.copy(),
            metadata={
                "prompt_length": len(prompt),
                "response_length": len(response),
                "evaluator_count": len(self._evaluators),
            },
        )
    
    async def score_async(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> CompositeScore:
        """Async version of score()."""
        import asyncio
        
        dimension_scores = {}
        
        async def evaluate_dimension(name: str, evaluator: Any):
            try:
                if hasattr(evaluator, "evaluate_async"):
                    result = await evaluator.evaluate_async(
                        prompt=prompt,
                        response=response,
                        context=context,
                        reference=reference,
                    )
                else:
                    result = evaluator.evaluate(
                        prompt=prompt,
                        response=response,
                        context=context,
                        reference=reference,
                    )
                
                if isinstance(result, dict):
                    return name, result.get("score", 0.0)
                elif hasattr(result, "score"):
                    return name, result.score
                else:
                    return name, float(result)
            except Exception:
                return name, 0.0
        
        tasks = [
            evaluate_dimension(name, evaluator)
            for name, evaluator in self._evaluators.items()
        ]
        
        results = await asyncio.gather(*tasks)
        dimension_scores = dict(results)
        
        overall = self._compute_weighted_average(dimension_scores)
        
        return CompositeScore(
            overall_score=overall,
            dimension_scores=dimension_scores,
            weights=self._weights.copy(),
        )
    
    def _compute_weighted_average(self, scores: Dict[str, float]) -> float:
        """Compute weighted average of scores."""
        if not scores:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, score in scores.items():
            weight = self._weights.get(name, self._default_weight)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def get_evaluator_names(self) -> List[str]:
        """Get names of all registered evaluators."""
        return list(self._evaluators.keys())


_global_scorer: Optional[QualityScorer] = None


def get_quality_scorer() -> QualityScorer:
    """Get the global quality scorer."""
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = QualityScorer()
    return _global_scorer
