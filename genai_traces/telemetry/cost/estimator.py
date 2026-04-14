"""
Cost estimation for LLM API calls.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional


# USD per 1M tokens — updated regularly
# Keys should match the model name as returned by the API
PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI GPT-4o family
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    
    # OpenAI GPT-4 Turbo
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "cached_input": 5.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
    "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
    
    # OpenAI GPT-4
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-0613": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    
    # OpenAI GPT-3.5 Turbo
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
    
    # OpenAI o1 family
    "o1-preview": {"input": 15.00, "output": 60.00, "cached_input": 7.50},
    "o1-preview-2024-09-12": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00, "cached_input": 1.50},
    "o1-mini-2024-09-12": {"input": 3.00, "output": 12.00},
    
    # OpenAI Embeddings
    "text-embedding-ada-002": {"input": 0.10, "output": 0.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
    
    # Anthropic Claude 3
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "cached_input": 0.03},
    
    # Anthropic Claude 3.5
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    
    # Google Gemini
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    
    # AWS Bedrock (approximate)
    "amazon.titan-text-express-v1": {"input": 0.80, "output": 1.60},
    "amazon.titan-text-lite-v1": {"input": 0.30, "output": 0.40},
    "anthropic.claude-3-opus-20240229-v1:0": {"input": 15.00, "output": 75.00},
    "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 3.00, "output": 15.00},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.25, "output": 1.25},
}


class CostEstimator:
    """
    Estimates costs for LLM API calls.
    
    Uses precise Decimal arithmetic to avoid floating-point errors.
    
    Usage:
        estimator = CostEstimator()
        costs = estimator.estimate(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        print(f"Total cost: ${costs['total_cost_usd']:.6f}")
    """
    
    def __init__(self, custom_pricing: Optional[Dict] = None):
        """
        Initialize the cost estimator.
        
        Args:
            custom_pricing: Optional custom pricing to override defaults
        """
        self._pricing = {**PRICING, **(custom_pricing or {})}
    
    def estimate(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> Dict[str, float]:
        """
        Estimate costs for an LLM call.
        
        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens (for providers that support it)
            
        Returns:
            Dictionary with cost breakdown:
            - input_cost_usd: Cost for input tokens
            - output_cost_usd: Cost for output tokens
            - cache_cost_usd: Cost for cached tokens
            - total_cost_usd: Total cost
        """
        p = self._pricing.get(model, {})
        
        if not p:
            # Try to find a partial match
            for key in self._pricing:
                if key in model or model in key:
                    p = self._pricing[key]
                    break
        
        if not p:
            # Unknown model — return zeros
            return {
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "cache_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            }
        
        M = Decimal("1000000")
        
        # Calculate input cost (excluding cached tokens)
        non_cached_input = max(0, prompt_tokens - cached_tokens)
        input_cost = Decimal(str(non_cached_input)) / M * Decimal(str(p.get("input", 0)))
        
        # Calculate output cost
        output_cost = Decimal(str(completion_tokens)) / M * Decimal(str(p.get("output", 0)))
        
        # Calculate cache cost
        cache_cost = Decimal("0")
        if cached_tokens > 0 and "cached_input" in p:
            cache_cost = Decimal(str(cached_tokens)) / M * Decimal(str(p["cached_input"]))
        
        total = input_cost + output_cost + cache_cost
        
        def r(d: Decimal) -> float:
            """Round to 6 decimal places."""
            return float(d.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
        
        return {
            "input_cost_usd": r(input_cost),
            "output_cost_usd": r(output_cost),
            "cache_cost_usd": r(cache_cost),
            "total_cost_usd": r(total),
        }
    
    def get_pricing(self, model: str) -> Optional[Dict[str, float]]:
        """Get pricing for a model."""
        return self._pricing.get(model)
    
    def add_pricing(self, model: str, pricing: Dict[str, float]) -> None:
        """Add or update pricing for a model."""
        self._pricing[model] = pricing
    
    def list_models(self) -> list:
        """List all models with known pricing."""
        return list(self._pricing.keys())


# Global instance for convenience
_default_estimator: Optional[CostEstimator] = None


def get_cost_estimator() -> CostEstimator:
    """Get the default cost estimator instance."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = CostEstimator()
    return _default_estimator


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> Dict[str, float]:
    """Convenience function to estimate costs."""
    return get_cost_estimator().estimate(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )
