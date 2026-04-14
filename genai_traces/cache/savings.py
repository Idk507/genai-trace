"""
Cache savings computation for GenAI-Traces.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from decimal import Decimal

from ..telemetry.cost.estimator import CostEstimator


@dataclass
class CacheSavings:
    """Cache savings summary."""
    
    total_savings_usd: float
    total_hits: int
    total_misses: int
    hit_rate: float
    estimated_tokens_saved: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_savings_usd": round(self.total_savings_usd, 6),
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.hit_rate, 4),
            "hit_rate_pct": round(self.hit_rate * 100, 2),
            "estimated_tokens_saved": self.estimated_tokens_saved,
        }


def compute_cache_savings(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    is_hit: bool,
) -> float:
    """
    Compute cost savings from a cache hit.
    
    Args:
        model: Model name
        prompt_tokens: Tokens in the prompt
        completion_tokens: Tokens in the completion
        is_hit: Whether this was a cache hit
        
    Returns:
        Estimated savings in USD (0 if miss)
    """
    if not is_hit:
        return 0.0
    
    estimator = CostEstimator()
    cost = estimator.estimate(model, prompt_tokens, completion_tokens)
    
    return float(cost.get("total_cost_usd", 0))


class CacheSavingsTracker:
    """
    Tracks cache savings over time.
    """
    
    def __init__(self):
        self._total_savings = Decimal("0")
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._by_model: Dict[str, Decimal] = {}
    
    def record_hit(
        self,
        model: str,
        savings_usd: float,
        tokens_saved: int = 0,
    ) -> None:
        """Record a cache hit with savings."""
        self._hits += 1
        self._total_savings += Decimal(str(savings_usd))
        self._tokens_saved += tokens_saved
        
        if model not in self._by_model:
            self._by_model[model] = Decimal("0")
        self._by_model[model] += Decimal(str(savings_usd))
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1
    
    def get_summary(self) -> CacheSavings:
        """Get savings summary."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return CacheSavings(
            total_savings_usd=float(self._total_savings),
            total_hits=self._hits,
            total_misses=self._misses,
            hit_rate=hit_rate,
            estimated_tokens_saved=self._tokens_saved,
        )
    
    def get_savings_by_model(self) -> Dict[str, float]:
        """Get savings broken down by model."""
        return {k: float(v) for k, v in self._by_model.items()}
    
    def reset(self) -> None:
        """Reset all counters."""
        self._total_savings = Decimal("0")
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._by_model.clear()


_tracker: Optional[CacheSavingsTracker] = None


def get_cache_savings_tracker() -> CacheSavingsTracker:
    """Get the global cache savings tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CacheSavingsTracker()
    return _tracker
