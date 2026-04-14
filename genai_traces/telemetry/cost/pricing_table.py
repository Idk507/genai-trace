"""
Live pricing registry for LLM models.

Maintains up-to-date pricing information for cost estimation.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import json


@dataclass
class ModelPricing:
    """Pricing information for a single model."""
    
    model: str
    provider: str
    prompt_cost_per_1k: Decimal
    completion_cost_per_1k: Decimal
    cached_prompt_cost_per_1k: Optional[Decimal] = None
    context_window: int = 8192
    max_output_tokens: int = 4096
    updated_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_cost_per_1k": str(self.prompt_cost_per_1k),
            "completion_cost_per_1k": str(self.completion_cost_per_1k),
            "cached_prompt_cost_per_1k": str(self.cached_prompt_cost_per_1k) if self.cached_prompt_cost_per_1k else None,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "updated_at": self.updated_at,
        }


DEFAULT_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.005"),
        completion_cost_per_1k=Decimal("0.015"),
        cached_prompt_cost_per_1k=Decimal("0.0025"),
        context_window=128000,
        max_output_tokens=16384,
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.00015"),
        completion_cost_per_1k=Decimal("0.0006"),
        cached_prompt_cost_per_1k=Decimal("0.000075"),
        context_window=128000,
        max_output_tokens=16384,
    ),
    "gpt-4-turbo": ModelPricing(
        model="gpt-4-turbo",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.01"),
        completion_cost_per_1k=Decimal("0.03"),
        context_window=128000,
        max_output_tokens=4096,
    ),
    "gpt-4": ModelPricing(
        model="gpt-4",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.03"),
        completion_cost_per_1k=Decimal("0.06"),
        context_window=8192,
        max_output_tokens=8192,
    ),
    "gpt-3.5-turbo": ModelPricing(
        model="gpt-3.5-turbo",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.0005"),
        completion_cost_per_1k=Decimal("0.0015"),
        context_window=16385,
        max_output_tokens=4096,
    ),
    "claude-3-opus": ModelPricing(
        model="claude-3-opus",
        provider="anthropic",
        prompt_cost_per_1k=Decimal("0.015"),
        completion_cost_per_1k=Decimal("0.075"),
        cached_prompt_cost_per_1k=Decimal("0.01875"),
        context_window=200000,
        max_output_tokens=4096,
    ),
    "claude-3-5-sonnet": ModelPricing(
        model="claude-3-5-sonnet",
        provider="anthropic",
        prompt_cost_per_1k=Decimal("0.003"),
        completion_cost_per_1k=Decimal("0.015"),
        cached_prompt_cost_per_1k=Decimal("0.00375"),
        context_window=200000,
        max_output_tokens=8192,
    ),
    "claude-3-sonnet": ModelPricing(
        model="claude-3-sonnet",
        provider="anthropic",
        prompt_cost_per_1k=Decimal("0.003"),
        completion_cost_per_1k=Decimal("0.015"),
        context_window=200000,
        max_output_tokens=4096,
    ),
    "claude-3-haiku": ModelPricing(
        model="claude-3-haiku",
        provider="anthropic",
        prompt_cost_per_1k=Decimal("0.00025"),
        completion_cost_per_1k=Decimal("0.00125"),
        cached_prompt_cost_per_1k=Decimal("0.0003"),
        context_window=200000,
        max_output_tokens=4096,
    ),
    "text-embedding-3-small": ModelPricing(
        model="text-embedding-3-small",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.00002"),
        completion_cost_per_1k=Decimal("0"),
        context_window=8191,
        max_output_tokens=0,
    ),
    "text-embedding-3-large": ModelPricing(
        model="text-embedding-3-large",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.00013"),
        completion_cost_per_1k=Decimal("0"),
        context_window=8191,
        max_output_tokens=0,
    ),
    "text-embedding-ada-002": ModelPricing(
        model="text-embedding-ada-002",
        provider="openai",
        prompt_cost_per_1k=Decimal("0.0001"),
        completion_cost_per_1k=Decimal("0"),
        context_window=8191,
        max_output_tokens=0,
    ),
}


class PricingTable:
    """
    Manages LLM pricing information with support for updates.
    """
    
    def __init__(self, auto_refresh: bool = False, refresh_interval_hours: int = 24):
        """
        Initialize the pricing table.
        
        Args:
            auto_refresh: Whether to automatically refresh pricing
            refresh_interval_hours: Hours between refresh attempts
        """
        self._pricing: Dict[str, ModelPricing] = dict(DEFAULT_PRICING)
        self._custom_pricing: Dict[str, ModelPricing] = {}
        self._last_refresh: Optional[datetime] = None
        self._auto_refresh = auto_refresh
        self._refresh_interval = timedelta(hours=refresh_interval_hours)
    
    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """
        Get pricing for a model.
        
        Args:
            model: Model name (exact or partial match)
            
        Returns:
            ModelPricing if found, None otherwise
        """
        if model in self._custom_pricing:
            return self._custom_pricing[model]
        
        if model in self._pricing:
            return self._pricing[model]
        
        model_lower = model.lower()
        for key, pricing in {**self._pricing, **self._custom_pricing}.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return pricing
        
        return None
    
    def set_pricing(self, model: str, pricing: ModelPricing) -> None:
        """Set custom pricing for a model."""
        pricing.updated_at = datetime.utcnow().isoformat()
        self._custom_pricing[model] = pricing
    
    def update_pricing(
        self,
        model: str,
        prompt_cost_per_1k: Optional[Decimal] = None,
        completion_cost_per_1k: Optional[Decimal] = None,
        cached_prompt_cost_per_1k: Optional[Decimal] = None,
    ) -> None:
        """Update specific pricing fields for a model."""
        existing = self.get_pricing(model)
        if existing:
            if prompt_cost_per_1k is not None:
                existing.prompt_cost_per_1k = prompt_cost_per_1k
            if completion_cost_per_1k is not None:
                existing.completion_cost_per_1k = completion_cost_per_1k
            if cached_prompt_cost_per_1k is not None:
                existing.cached_prompt_cost_per_1k = cached_prompt_cost_per_1k
            existing.updated_at = datetime.utcnow().isoformat()
            self._custom_pricing[model] = existing
    
    def list_models(self, provider: Optional[str] = None) -> Dict[str, ModelPricing]:
        """List all models, optionally filtered by provider."""
        all_pricing = {**self._pricing, **self._custom_pricing}
        
        if provider:
            return {
                k: v for k, v in all_pricing.items()
                if v.provider.lower() == provider.lower()
            }
        
        return all_pricing
    
    def export_to_json(self, path: str) -> None:
        """Export pricing table to JSON file."""
        all_pricing = {**self._pricing, **self._custom_pricing}
        data = {k: v.to_dict() for k, v in all_pricing.items()}
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_from_json(self, path: str) -> None:
        """Import pricing from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for model, pricing_data in data.items():
            self._custom_pricing[model] = ModelPricing(
                model=pricing_data["model"],
                provider=pricing_data["provider"],
                prompt_cost_per_1k=Decimal(pricing_data["prompt_cost_per_1k"]),
                completion_cost_per_1k=Decimal(pricing_data["completion_cost_per_1k"]),
                cached_prompt_cost_per_1k=Decimal(pricing_data["cached_prompt_cost_per_1k"]) if pricing_data.get("cached_prompt_cost_per_1k") else None,
                context_window=pricing_data.get("context_window", 8192),
                max_output_tokens=pricing_data.get("max_output_tokens", 4096),
                updated_at=pricing_data.get("updated_at", ""),
            )
    
    def should_refresh(self) -> bool:
        """Check if pricing should be refreshed."""
        if not self._auto_refresh:
            return False
        if self._last_refresh is None:
            return True
        return datetime.utcnow() - self._last_refresh > self._refresh_interval


_pricing_table = PricingTable()


def get_pricing_table() -> PricingTable:
    """Get the global pricing table instance."""
    return _pricing_table


def get_model_pricing(model: str) -> Optional[ModelPricing]:
    """Convenience function to get pricing for a model."""
    return _pricing_table.get_pricing(model)
