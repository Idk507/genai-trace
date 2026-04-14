"""
LLM Router decision tracing.

Traces routing decisions including model selection, fallback chains, and reasons.
"""

import contextlib
from typing import List, Optional, Any, Dict, Generator
from dataclasses import dataclass, field

from ..core.tracer import get_tracer
from ..core.types import SpanType
from ..telemetry.cost.estimator import CostEstimator


@dataclass
class RouterDecision:
    """A routing decision record."""
    
    selected_model: str
    reason: str
    is_fallback: bool = False
    fallback_count: int = 0
    estimated_cost: Optional[float] = None
    latency_estimate_ms: Optional[float] = None


class RouterContext:
    """
    Context for router tracing.
    
    Tracks model selection decisions and fallback chains.
    """
    
    def __init__(
        self,
        span: Any,
        primary: str,
        budget_usd: Optional[float] = None,
    ):
        self.span = span
        self.primary = primary
        self.budget_usd = budget_usd
        self._estimator = CostEstimator()
        self._attempts = 0
        self._decisions: List[RouterDecision] = []
    
    def select(
        self,
        candidates: List[str],
        prompt_tokens: int,
        reason: str = "cost",
        latency_requirements_ms: Optional[float] = None,
    ) -> str:
        """
        Select the best model from candidates given constraints.
        
        Args:
            candidates: List of model names to choose from
            prompt_tokens: Estimated prompt tokens
            reason: Selection reason ('cost', 'latency', 'availability', 'manual')
            latency_requirements_ms: Maximum acceptable latency
            
        Returns:
            Selected model name
        """
        selected = candidates[0] if candidates else self.primary
        is_fallback = False
        estimated_cost = None
        
        if reason == "cost" and self.budget_usd:
            for model in candidates:
                cost = self._estimator.estimate(model, prompt_tokens, prompt_tokens // 2)
                if cost["total_cost_usd"] <= self.budget_usd:
                    selected = model
                    estimated_cost = cost["total_cost_usd"]
                    break
            else:
                selected = candidates[-1] if candidates else self.primary
                is_fallback = selected != self.primary
        
        elif reason == "latency" and latency_requirements_ms:
            latency_estimates = {
                "gpt-4o": 800,
                "gpt-4o-mini": 400,
                "gpt-3.5-turbo": 300,
                "claude-3-haiku": 350,
                "claude-3-sonnet": 600,
            }
            
            for model in candidates:
                est_latency = latency_estimates.get(model, 500)
                if est_latency <= latency_requirements_ms:
                    selected = model
                    break
        
        is_fallback = selected != self.primary
        
        decision = RouterDecision(
            selected_model=selected,
            reason=reason,
            is_fallback=is_fallback,
            fallback_count=self._attempts,
            estimated_cost=estimated_cost,
        )
        self._decisions.append(decision)
        
        self.span.set_attribute("router.selected_model", selected)
        self.span.set_attribute("router.reason", reason)
        self.span.set_attribute("router.is_fallback", is_fallback)
        
        if estimated_cost:
            self.span.set_attribute("router.estimated_cost_usd", estimated_cost)
        
        return selected
    
    def record_attempt(
        self,
        model: str,
        success: bool,
        error: Optional[Exception] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Record an attempt to use a model.
        
        Args:
            model: Model that was attempted
            success: Whether the attempt succeeded
            error: Exception if failed
            latency_ms: Actual latency
        """
        self._attempts += 1
        
        self.span.add_event(
            "router_attempt",
            {
                "model": model,
                "success": success,
                "attempt_number": self._attempts,
                "error": str(error) if error else None,
                "latency_ms": latency_ms,
            }
        )
        
        if not success:
            self.span.set_attribute("router.last_error", str(error) if error else "unknown")
    
    def record_outcome(
        self,
        model: str,
        response: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Record the final outcome.
        
        Args:
            model: Model that was ultimately used
            response: Successful response if any
            error: Final error if failed
        """
        self.span.set_attribute("router.final_model", model)
        self.span.set_attribute("router.fallback_count", max(0, self._attempts - 1))
        self.span.set_attribute("router.total_attempts", self._attempts)
        
        if error:
            self.span.set_attribute("router.final_error", str(error))
            self.span.record_exception(error)
        elif response:
            self.span.record_response(response)
    
    def get_decisions(self) -> List[RouterDecision]:
        """Get all routing decisions made."""
        return self._decisions


@contextlib.contextmanager
def trace_router(
    primary: str,
    budget_usd: Optional[float] = None,
    name: str = "llm_router",
) -> Generator[RouterContext, None, None]:
    """
    Context manager for tracing LLM routing decisions.
    
    Usage:
        with trace_router(primary="gpt-4o", budget_usd=0.05) as router:
            selected = router.select(
                candidates=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                prompt_tokens=1500,
            )
            response = call_llm(selected, prompt)
            router.record_outcome(selected, response)
    
    Args:
        primary: Primary/preferred model
        budget_usd: Optional budget constraint
        name: Span name
        
    Yields:
        RouterContext for tracking decisions
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(name, SpanType.ROUTER_DECISION) as span:
        span.set_attribute("router.primary_model", primary)
        if budget_usd:
            span.set_attribute("router.budget_usd", budget_usd)
        
        router = RouterContext(span=span, primary=primary, budget_usd=budget_usd)
        
        try:
            yield router
        except Exception as e:
            span.record_exception(e)
            raise
