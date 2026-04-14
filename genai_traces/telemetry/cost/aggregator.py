"""
Cost aggregation for GenAI-Traces.

Provides per-session, per-conversation, and per-user cost rollups.
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import threading


@dataclass
class CostSummary:
    """Summary of costs for a time period or entity."""
    
    total_cost_usd: Decimal = Decimal("0")
    prompt_cost_usd: Decimal = Decimal("0")
    completion_cost_usd: Decimal = Decimal("0")
    cache_savings_usd: Decimal = Decimal("0")
    
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    request_count: int = 0
    
    first_request_at: Optional[datetime] = None
    last_request_at: Optional[datetime] = None
    
    by_model: Dict[str, Decimal] = field(default_factory=dict)
    
    def add(
        self,
        cost_usd: Decimal,
        prompt_cost: Decimal = Decimal("0"),
        completion_cost: Decimal = Decimal("0"),
        cache_savings: Decimal = Decimal("0"),
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "unknown",
    ) -> None:
        """Add a cost entry to the summary."""
        self.total_cost_usd += cost_usd
        self.prompt_cost_usd += prompt_cost
        self.completion_cost_usd += completion_cost
        self.cache_savings_usd += cache_savings
        
        self.total_tokens += tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        
        self.request_count += 1
        
        now = datetime.utcnow()
        if self.first_request_at is None:
            self.first_request_at = now
        self.last_request_at = now
        
        if model not in self.by_model:
            self.by_model[model] = Decimal("0")
        self.by_model[model] += cost_usd
    
    @property
    def average_cost_per_request(self) -> Decimal:
        """Average cost per request."""
        if self.request_count == 0:
            return Decimal("0")
        return self.total_cost_usd / self.request_count
    
    @property
    def average_tokens_per_request(self) -> float:
        """Average tokens per request."""
        if self.request_count == 0:
            return 0.0
        return self.total_tokens / self.request_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cost_usd": float(self.total_cost_usd),
            "prompt_cost_usd": float(self.prompt_cost_usd),
            "completion_cost_usd": float(self.completion_cost_usd),
            "cache_savings_usd": float(self.cache_savings_usd),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "request_count": self.request_count,
            "average_cost_per_request": float(self.average_cost_per_request),
            "average_tokens_per_request": self.average_tokens_per_request,
            "first_request_at": self.first_request_at.isoformat() if self.first_request_at else None,
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
            "by_model": {k: float(v) for k, v in self.by_model.items()},
        }


class CostAggregator:
    """
    Aggregates costs across different dimensions.
    
    Tracks costs by:
    - Session
    - Conversation
    - User
    - Time period (hourly, daily)
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize the cost aggregator.
        
        Args:
            retention_hours: Hours to retain detailed cost data
        """
        self._retention = timedelta(hours=retention_hours)
        self._lock = threading.Lock()
        
        self._by_session: Dict[str, CostSummary] = defaultdict(CostSummary)
        self._by_conversation: Dict[str, CostSummary] = defaultdict(CostSummary)
        self._by_user: Dict[str, CostSummary] = defaultdict(CostSummary)
        self._by_hour: Dict[str, CostSummary] = defaultdict(CostSummary)
        self._global = CostSummary()
    
    def record(
        self,
        cost_usd: float,
        prompt_cost: float = 0.0,
        completion_cost: float = 0.0,
        cache_savings: float = 0.0,
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "unknown",
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Record a cost entry.
        
        Args:
            cost_usd: Total cost in USD
            prompt_cost: Cost for prompt tokens
            completion_cost: Cost for completion tokens
            cache_savings: Savings from cache hits
            tokens: Total tokens
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count
            model: Model name
            session_id: Optional session identifier
            conversation_id: Optional conversation identifier
            user_id: Optional user identifier
        """
        cost = Decimal(str(cost_usd))
        prompt = Decimal(str(prompt_cost))
        completion = Decimal(str(completion_cost))
        savings = Decimal(str(cache_savings))
        
        with self._lock:
            self._global.add(
                cost, prompt, completion, savings,
                tokens, prompt_tokens, completion_tokens, model
            )
            
            if session_id:
                self._by_session[session_id].add(
                    cost, prompt, completion, savings,
                    tokens, prompt_tokens, completion_tokens, model
                )
            
            if conversation_id:
                self._by_conversation[conversation_id].add(
                    cost, prompt, completion, savings,
                    tokens, prompt_tokens, completion_tokens, model
                )
            
            if user_id:
                self._by_user[user_id].add(
                    cost, prompt, completion, savings,
                    tokens, prompt_tokens, completion_tokens, model
                )
            
            hour_key = datetime.utcnow().strftime("%Y-%m-%d-%H")
            self._by_hour[hour_key].add(
                cost, prompt, completion, savings,
                tokens, prompt_tokens, completion_tokens, model
            )
    
    def get_session_summary(self, session_id: str) -> CostSummary:
        """Get cost summary for a session."""
        with self._lock:
            return self._by_session.get(session_id, CostSummary())
    
    def get_conversation_summary(self, conversation_id: str) -> CostSummary:
        """Get cost summary for a conversation."""
        with self._lock:
            return self._by_conversation.get(conversation_id, CostSummary())
    
    def get_user_summary(self, user_id: str) -> CostSummary:
        """Get cost summary for a user."""
        with self._lock:
            return self._by_user.get(user_id, CostSummary())
    
    def get_hourly_summary(self, hour_key: str) -> CostSummary:
        """Get cost summary for an hour (format: YYYY-MM-DD-HH)."""
        with self._lock:
            return self._by_hour.get(hour_key, CostSummary())
    
    def get_global_summary(self) -> CostSummary:
        """Get global cost summary."""
        with self._lock:
            return self._global
    
    def get_top_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top users by cost."""
        with self._lock:
            sorted_users = sorted(
                self._by_user.items(),
                key=lambda x: x[1].total_cost_usd,
                reverse=True
            )[:limit]
            
            return [
                {"user_id": user_id, **summary.to_dict()}
                for user_id, summary in sorted_users
            ]
    
    def get_cost_by_model(self) -> Dict[str, float]:
        """Get total costs broken down by model."""
        with self._lock:
            return {k: float(v) for k, v in self._global.by_model.items()}
    
    def check_budget(
        self,
        budget_usd: float,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check if spending is within budget.
        
        Returns:
            Dict with within_budget, current_spend, budget, remaining
        """
        budget = Decimal(str(budget_usd))
        
        with self._lock:
            if session_id:
                current = self._by_session.get(session_id, CostSummary()).total_cost_usd
            elif user_id:
                current = self._by_user.get(user_id, CostSummary()).total_cost_usd
            else:
                current = self._global.total_cost_usd
        
        return {
            "within_budget": current <= budget,
            "current_spend_usd": float(current),
            "budget_usd": float(budget),
            "remaining_usd": float(max(Decimal("0"), budget - current)),
            "utilization_pct": float(current / budget * 100) if budget > 0 else 0,
        }
    
    def cleanup_old_data(self) -> int:
        """Remove data older than retention period. Returns count of removed entries."""
        cutoff = datetime.utcnow() - self._retention
        removed = 0
        
        with self._lock:
            for summaries in [self._by_session, self._by_conversation]:
                to_remove = [
                    k for k, v in summaries.items()
                    if v.last_request_at and v.last_request_at < cutoff
                ]
                for k in to_remove:
                    del summaries[k]
                    removed += 1
        
        return removed


_aggregator = CostAggregator()


def get_cost_aggregator() -> CostAggregator:
    """Get the global cost aggregator instance."""
    return _aggregator


def record_cost(
    cost_usd: float,
    model: str = "unknown",
    session_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> None:
    """Convenience function to record a cost entry."""
    _aggregator.record(
        cost_usd=cost_usd,
        model=model,
        session_id=session_id,
        conversation_id=conversation_id,
        user_id=user_id,
        **kwargs
    )
