"""
LangGraph integration for GenAI-Traces.

Provides hooks for tracing LangGraph graph executions.
"""

import functools
import time
from typing import Any, Dict, Optional, Callable

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


_instrumented = False


def instrument_langgraph() -> None:
    """
    Instrument LangGraph for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.frameworks.langgraph import instrument_langgraph
        instrument_langgraph()
        
        # All LangGraph executions are now traced
    """
    global _instrumented
    
    if _instrumented:
        return
    
    try:
        from langgraph.graph import StateGraph
    except ImportError:
        return
    
    original_compile = StateGraph.compile
    
    @functools.wraps(original_compile)
    def traced_compile(self, *args, **kwargs):
        compiled = original_compile(self, *args, **kwargs)
        return TracedCompiledGraph(compiled)
    
    StateGraph.compile = traced_compile
    _instrumented = True


class TracedCompiledGraph:
    """Wrapper around compiled LangGraph that adds tracing."""
    
    def __init__(self, compiled_graph):
        self._graph = compiled_graph
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)
    
    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """Invoke the graph with tracing."""
        tracer = get_tracer()
        
        with tracer.start_as_current_span("langgraph.invoke", SpanType.CHAIN) as span:
            span.set_attribute("langgraph.input", str(input)[:1000])
            
            if config:
                span.set_attribute("langgraph.config", config)
            
            start_time = time.perf_counter()
            
            try:
                result = self._graph.invoke(input, config, **kwargs)
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("langgraph.duration_ms", duration_ms)
                span.set_attribute("langgraph.output", str(result)[:1000])
                span.status = SpanStatus.OK
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                raise
    
    async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """Async invoke the graph with tracing."""
        tracer = get_tracer()
        
        async with tracer.start_as_current_span_async("langgraph.ainvoke", SpanType.CHAIN) as span:
            span.set_attribute("langgraph.input", str(input)[:1000])
            
            start_time = time.perf_counter()
            
            try:
                result = await self._graph.ainvoke(input, config, **kwargs)
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("langgraph.duration_ms", duration_ms)
                span.set_attribute("langgraph.output", str(result)[:1000])
                span.status = SpanStatus.OK
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                raise
    
    def stream(self, input: Any, config: Optional[Dict] = None, **kwargs):
        """Stream the graph with tracing."""
        tracer = get_tracer()
        
        with tracer.start_as_current_span("langgraph.stream", SpanType.CHAIN) as span:
            span.set_attribute("langgraph.input", str(input)[:1000])
            span.set_attribute("langgraph.streaming", True)
            
            start_time = time.perf_counter()
            step_count = 0
            
            try:
                for step in self._graph.stream(input, config, **kwargs):
                    step_count += 1
                    yield step
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("langgraph.duration_ms", duration_ms)
                span.set_attribute("langgraph.step_count", step_count)
                span.status = SpanStatus.OK
                
            except Exception as e:
                span.record_exception(e)
                raise


def trace_node(name: Optional[str] = None) -> Callable:
    """
    Decorator to trace individual LangGraph nodes.
    
    Usage:
        @trace_node("my_node")
        def my_node(state):
            return {"result": "value"}
    """
    def decorator(func: Callable) -> Callable:
        node_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(f"langgraph.node.{node_name}", SpanType.CHAIN) as span:
                span.set_attribute("langgraph.node.name", node_name)
                span.set_attribute("langgraph.node.input_state", str(state)[:500])
                
                start_time = time.perf_counter()
                
                try:
                    result = func(state, *args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("langgraph.node.duration_ms", duration_ms)
                    span.set_attribute("langgraph.node.output", str(result)[:500])
                    span.status = SpanStatus.OK
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(state, *args, **kwargs):
            tracer = get_tracer()
            
            async with tracer.start_as_current_span_async(f"langgraph.node.{node_name}", SpanType.CHAIN) as span:
                span.set_attribute("langgraph.node.name", node_name)
                
                start_time = time.perf_counter()
                
                try:
                    result = await func(state, *args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("langgraph.node.duration_ms", duration_ms)
                    span.status = SpanStatus.OK
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator
