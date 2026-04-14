"""
Caching layer tracing for GenAI-Traces.

Traces semantic cache hits and misses with cost savings computation.
"""

from .tracer import trace_cache_lookup, CacheContext
from .savings import compute_cache_savings, CacheSavings

__all__ = [
    "trace_cache_lookup",
    "CacheContext",
    "compute_cache_savings",
    "CacheSavings",
]
