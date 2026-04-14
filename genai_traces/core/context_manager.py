"""
Context managers for tracing.
"""

import contextlib
from typing import Optional

from .types import SpanType, SpanStatus
from .tracer import get_tracer
from .span import Span
from ..utils.id_generator import generate_trace_id, generate_span_id


class SecurityError(Exception):
    """Raised when a security guardrail blocks a request."""
    pass


@contextlib.contextmanager
def trace_llm_context(
    name: str = "llm_call",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    check_injection: bool = False,
    prompt: Optional[str] = None,
):
    """
    Context manager for tracing LLM calls.
    
    Args:
        name: Name of the span
        model: Model name
        provider: Provider name
        check_injection: Whether to check for prompt injection
        prompt: The prompt text (required if check_injection is True)
        
    Yields:
        Span instance
        
    Raises:
        SecurityError: If prompt injection is detected and check_injection is True
        
    Example:
        with trace_llm_context(name="summarize", model="gpt-4o") as span:
            response = openai.chat.completions.create(...)
            span.record_response(response)
    """
    tracer = get_tracer()
    attrs = {}
    if model:
        attrs["llm.model.name"] = model
    if provider:
        attrs["llm.provider"] = provider
    
    # Optional injection check BEFORE opening the span
    if check_injection and prompt:
        try:
            from ..security.injection_detector import InjectionDetector
            result = InjectionDetector().check(prompt)
            if result.is_injection:
                # Create a blocked span and export it
                blocked = Span(
                    trace_id=generate_trace_id(),
                    span_id=generate_span_id(),
                    name=name,
                    span_type=SpanType.INJECTION_CHECK,
                    status=SpanStatus.BLOCKED,
                )
                blocked.injection_detected = True
                blocked.injection_type = result.injection_type.value
                blocked.set_attribute("security.injection_score", result.score)
                tracer._finish_span(blocked)
                raise SecurityError(f"Prompt injection detected: {result.injection_type.value}")
        except ImportError:
            pass  # Security module not available
    
    with tracer.start_as_current_span(name, SpanType.LLM, attrs) as span:
        if prompt:
            span.set_attribute("llm.prompt", prompt)
        yield span


@contextlib.asynccontextmanager
async def trace_llm_context_async(
    name: str = "llm_call",
    model: Optional[str] = None,
    provider: Optional[str] = None,
    check_injection: bool = False,
    prompt: Optional[str] = None,
):
    """
    Async context manager for tracing LLM calls.
    
    Args:
        name: Name of the span
        model: Model name
        provider: Provider name
        check_injection: Whether to check for prompt injection
        prompt: The prompt text (required if check_injection is True)
        
    Yields:
        Span instance
        
    Raises:
        SecurityError: If prompt injection is detected and check_injection is True
    """
    tracer = get_tracer()
    attrs = {}
    if model:
        attrs["llm.model.name"] = model
    if provider:
        attrs["llm.provider"] = provider
    
    # Optional injection check BEFORE opening the span
    if check_injection and prompt:
        try:
            from ..security.injection_detector import InjectionDetector
            result = InjectionDetector().check(prompt)
            if result.is_injection:
                blocked = Span(
                    trace_id=generate_trace_id(),
                    span_id=generate_span_id(),
                    name=name,
                    span_type=SpanType.INJECTION_CHECK,
                    status=SpanStatus.BLOCKED,
                )
                blocked.injection_detected = True
                blocked.injection_type = result.injection_type.value
                blocked.set_attribute("security.injection_score", result.score)
                tracer._finish_span(blocked)
                raise SecurityError(f"Prompt injection detected: {result.injection_type.value}")
        except ImportError:
            pass
    
    async with tracer.start_as_current_span_async(name, SpanType.LLM, attrs) as span:
        if prompt:
            span.set_attribute("llm.prompt", prompt)
        yield span


@contextlib.contextmanager
def trace_agent_context(
    name: str = "agent",
    agent_type: str = "react",
    goal: Optional[str] = None,
):
    """
    Context manager for tracing agent operations.
    
    Args:
        name: Name of the span
        agent_type: Type of agent
        goal: Goal or objective of the agent
        
    Yields:
        Span instance
    """
    tracer = get_tracer()
    attrs = {"agent.type": agent_type}
    if goal:
        attrs["agent.goal"] = goal
    
    with tracer.start_as_current_span(name, SpanType.AGENT, attrs) as span:
        yield span


@contextlib.contextmanager
def trace_tool_context(
    name: str = "tool",
    tool_name: Optional[str] = None,
):
    """
    Context manager for tracing tool operations.
    
    Args:
        name: Name of the span
        tool_name: Name of the tool being used
        
    Yields:
        Span instance
    """
    tracer = get_tracer()
    attrs = {}
    if tool_name:
        attrs["tool.name"] = tool_name
    
    with tracer.start_as_current_span(name, SpanType.TOOL, attrs) as span:
        yield span
