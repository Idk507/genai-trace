"""
Haystack integration for GenAI-Traces.

Provides tracing for Haystack pipelines.
"""

import functools
import time
from typing import Any, Dict, Optional

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


_instrumented = False


def instrument_haystack() -> None:
    """
    Instrument Haystack for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.frameworks.haystack import instrument_haystack
        instrument_haystack()
        
        # All Haystack pipeline runs are now traced
    """
    global _instrumented
    
    if _instrumented:
        return
    
    try:
        from haystack import Pipeline
    except ImportError:
        return
    
    original_run = Pipeline.run
    
    @functools.wraps(original_run)
    def traced_run(self, data: Dict[str, Any], **kwargs):
        return _trace_pipeline_run(original_run, self, data, **kwargs)
    
    Pipeline.run = traced_run
    _instrumented = True


def _trace_pipeline_run(original_fn, self, data: Dict[str, Any], **kwargs):
    """Wrap a Haystack pipeline run with tracing."""
    tracer = get_tracer()
    
    pipeline_name = getattr(self, "name", "pipeline")
    
    with tracer.start_as_current_span(f"haystack.pipeline.{pipeline_name}", SpanType.CHAIN) as span:
        span.set_attribute("haystack.pipeline.name", pipeline_name)
        span.set_attribute("haystack.pipeline.input_keys", list(data.keys()))
        
        component_names = list(self.graph.nodes.keys()) if hasattr(self, "graph") else []
        span.set_attribute("haystack.pipeline.components", component_names)
        
        start_time = time.perf_counter()
        
        try:
            result = original_fn(self, data, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("haystack.pipeline.duration_ms", duration_ms)
            span.set_attribute("haystack.pipeline.output_keys", list(result.keys()) if isinstance(result, dict) else [])
            span.status = SpanStatus.OK
            
            return result
            
        except Exception as e:
            span.record_exception(e)
            raise


class HaystackTracer:
    """
    Manual tracer for Haystack components.
    
    Usage:
        tracer = HaystackTracer()
        
        with tracer.trace_component("my_retriever", "retriever"):
            results = retriever.run(query="...")
    """
    
    def __init__(self):
        self._tracer = get_tracer()
    
    def trace_component(self, name: str, component_type: str = "component"):
        """Context manager for tracing a Haystack component."""
        span_type = self._get_span_type(component_type)
        return self._tracer.start_as_current_span(f"haystack.{component_type}.{name}", span_type)
    
    def _get_span_type(self, component_type: str) -> SpanType:
        """Map Haystack component type to span type."""
        type_lower = component_type.lower()
        
        if "retriever" in type_lower:
            return SpanType.RETRIEVAL
        elif "generator" in type_lower or "llm" in type_lower:
            return SpanType.LLM
        elif "embedder" in type_lower:
            return SpanType.EMBEDDING
        else:
            return SpanType.CHAIN
