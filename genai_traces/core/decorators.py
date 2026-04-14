"""
Decorators for tracing functions.
"""

import functools
import asyncio
from typing import Callable, Optional, Any

from .types import SpanType
from .tracer import get_tracer


def trace(
    span_type: str = "llm",
    name: Optional[str] = None,
    **attrs
):
    """
    Universal decorator for tracing functions.
    
    Works on both sync and async functions. Automatically creates a span
    that captures the function execution.
    
    Args:
        span_type: Type of span (llm, agent, tool, chain, etc.)
        name: Name of the span (defaults to function name)
        **attrs: Additional attributes to set on the span
        
    Returns:
        Decorated function
        
    Example:
        @trace(span_type="llm", model="gpt-4")
        def call_llm(prompt: str) -> str:
            return openai.chat.completions.create(...)
        
        @trace(span_type="agent", name="research_agent")
        async def run_agent(query: str):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__qualname__
        stype = SpanType(span_type)
        
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                async with tracer.start_as_current_span_async(span_name, stype, attrs) as span:
                    result = await fn(*args, **kwargs)
                    return result
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name, stype, attrs) as span:
                    return fn(*args, **kwargs)
            return sync_wrapper
    return decorator


def trace_llm(
    name: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    Convenience decorator for LLM calls.
    
    Automatically attaches model and provider information to the span.
    
    Args:
        name: Name of the span (defaults to function name)
        model: Model name (e.g., "gpt-4o", "claude-3-opus")
        provider: Provider name (e.g., "openai", "anthropic")
        
    Returns:
        Decorated function
        
    Example:
        @trace_llm(model="gpt-4o", provider="openai")
        def generate_summary(text: str) -> str:
            return openai.chat.completions.create(...)
    """
    extra = {}
    if model:
        extra["llm.model.name"] = model
    if provider:
        extra["llm.provider"] = provider
    return trace(span_type="llm", name=name, **extra)


def trace_agent(
    name: Optional[str] = None,
    agent_type: str = "react",
):
    """
    Convenience decorator for agent operations.
    
    Args:
        name: Name of the span (defaults to function name)
        agent_type: Type of agent (react, plan_execute, etc.)
        
    Returns:
        Decorated function
        
    Example:
        @trace_agent(agent_type="react")
        async def run_research_agent(query: str):
            ...
    """
    return trace(span_type="agent", name=name, **{"agent.type": agent_type})


def trace_tool(name: Optional[str] = None):
    """
    Convenience decorator for tool operations.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated function
        
    Example:
        @trace_tool()
        def search_web(query: str) -> list:
            ...
    """
    return trace(span_type="tool", name=name)


def trace_chain(name: Optional[str] = None):
    """
    Convenience decorator for chain operations.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated function
        
    Example:
        @trace_chain()
        def run_qa_chain(question: str) -> str:
            ...
    """
    return trace(span_type="chain", name=name)


def trace_retrieval(name: Optional[str] = None):
    """
    Convenience decorator for retrieval operations.
    
    Args:
        name: Name of the span (defaults to function name)
        
    Returns:
        Decorated function
        
    Example:
        @trace_retrieval()
        def search_documents(query: str) -> list:
            ...
    """
    return trace(span_type="retrieval", name=name)


def trace_embedding(
    name: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Convenience decorator for embedding operations.
    
    Args:
        name: Name of the span (defaults to function name)
        model: Embedding model name
        
    Returns:
        Decorated function
        
    Example:
        @trace_embedding(model="text-embedding-ada-002")
        def embed_text(text: str) -> list:
            ...
    """
    extra = {}
    if model:
        extra["llm.model.name"] = model
    return trace(span_type="embedding", name=name, **extra)
