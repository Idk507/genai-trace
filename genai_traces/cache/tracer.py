"""
Semantic cache tracing for GenAI-Traces.

Traces cache hits and misses for semantic caching systems.
"""

import contextlib
import hashlib
from typing import Optional, Any, Generator

from ..core.tracer import get_tracer
from ..core.types import SpanType


class CacheContext:
    """
    Context for cache tracing.
    
    Tracks cache lookups, hits, and misses.
    """
    
    def __init__(self, span: Any, model: str):
        self.span = span
        self.model = model
        self._hit = False
        self._similarity = 0.0
        self._savings = 0.0
    
    def record_hit(
        self,
        similarity: float = 1.0,
        savings_usd: Optional[float] = None,
        cached_response: Any = None,
    ) -> None:
        """
        Record a cache hit.
        
        Args:
            similarity: Similarity score (0-1) for semantic matches
            savings_usd: Estimated cost savings
            cached_response: The cached response (for logging)
        """
        self._hit = True
        self._similarity = similarity
        self._savings = savings_usd or 0.0
        
        self.span.set_attribute("cache.hit", True)
        self.span.set_attribute("cache.similarity_score", similarity)
        
        if savings_usd is not None:
            self.span.set_attribute("cache.savings_usd", savings_usd)
        
        from ..core.context import get_current_span
        parent = get_current_span()
        if parent and parent.span_id != self.span.span_id:
            parent.set_attribute("cost.cache_hit", True)
            parent.set_attribute("cost.cache_savings_usd", savings_usd or 0.0)
        
        self.span.add_event("cache_hit", {
            "similarity": similarity,
            "savings_usd": savings_usd,
        })
    
    def record_miss(self, reason: Optional[str] = None) -> None:
        """
        Record a cache miss.
        
        Args:
            reason: Optional reason for the miss
        """
        self._hit = False
        self._similarity = 0.0
        
        self.span.set_attribute("cache.hit", False)
        self.span.set_attribute("cache.similarity_score", 0.0)
        
        if reason:
            self.span.set_attribute("cache.miss_reason", reason)
        
        self.span.add_event("cache_miss", {"reason": reason})
    
    @property
    def was_hit(self) -> bool:
        """Check if this was a cache hit."""
        return self._hit
    
    @property
    def savings_usd(self) -> float:
        """Get the cost savings from this cache lookup."""
        return self._savings


def _hash(text: str, length: int = 16) -> str:
    """Create a hash of the text."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


@contextlib.contextmanager
def trace_cache_lookup(
    query: str,
    model: str = "unknown",
    ttl_seconds: int = 3600,
    cache_type: str = "semantic",
) -> Generator[CacheContext, None, None]:
    """
    Context manager for tracing cache lookups.
    
    Usage:
        with trace_cache_lookup(query=prompt, model="gpt-4o") as cache:
            cached = semantic_cache.get(prompt)
            if cached:
                cache.record_hit(similarity=0.97, savings_usd=0.003)
                return cached
            else:
                cache.record_miss()
                response = llm.generate(prompt)
                semantic_cache.set(prompt, response)
                return response
    
    Args:
        query: The query/prompt being cached
        model: Model name for cost estimation
        ttl_seconds: Cache TTL
        cache_type: Type of cache (semantic, exact, etc.)
        
    Yields:
        CacheContext for recording hit/miss
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span("cache.lookup", SpanType.CACHE_LOOKUP) as span:
        span.set_attribute("cache.key_hash", _hash(query))
        span.set_attribute("llm.model.name", model)
        span.set_attribute("cache.ttl_seconds", ttl_seconds)
        span.set_attribute("cache.type", cache_type)
        
        ctx = CacheContext(span=span, model=model)
        
        try:
            yield ctx
        except Exception as e:
            span.record_exception(e)
            raise


@contextlib.asynccontextmanager
async def trace_cache_lookup_async(
    query: str,
    model: str = "unknown",
    ttl_seconds: int = 3600,
    cache_type: str = "semantic",
):
    """
    Async context manager for tracing cache lookups.
    """
    tracer = get_tracer()
    
    async with tracer.start_as_current_span_async("cache.lookup", SpanType.CACHE_LOOKUP) as span:
        span.set_attribute("cache.key_hash", _hash(query))
        span.set_attribute("llm.model.name", model)
        span.set_attribute("cache.ttl_seconds", ttl_seconds)
        span.set_attribute("cache.type", cache_type)
        
        ctx = CacheContext(span=span, model=model)
        
        try:
            yield ctx
        except Exception as e:
            span.record_exception(e)
            raise
