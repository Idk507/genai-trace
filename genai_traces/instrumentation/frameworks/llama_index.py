"""
LlamaIndex integration for GenAI-Traces.

Provides callback handler for automatic tracing of LlamaIndex operations.
"""

import time
from typing import Any, Dict, List, Optional

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


class LlamaIndexCallbackHandler:
    """
    LlamaIndex callback handler for GenAI-Traces.
    
    Usage:
        from genai_traces.instrumentation.frameworks.llama_index import LlamaIndexCallbackHandler
        from llama_index.core import Settings
        
        handler = LlamaIndexCallbackHandler()
        Settings.callback_manager.add_handler(handler)
    """
    
    def __init__(self, tracer=None):
        self._tracer = tracer or get_tracer()
        self._spans: Dict[str, Any] = {}
        self._start_times: Dict[str, float] = {}
    
    def on_event_start(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when an event starts."""
        span_type = self._get_span_type(event_type)
        span = self._tracer.start_span(f"llama_index.{event_type}", span_type)
        
        span.set_attribute("llama_index.event_type", event_type)
        span.set_attribute("llama_index.event_id", event_id)
        
        if parent_id:
            span.set_attribute("llama_index.parent_id", parent_id)
        
        if payload:
            self._set_payload_attributes(span, event_type, payload)
        
        self._spans[event_id] = span
        self._start_times[event_id] = time.perf_counter()
        
        return event_id
    
    def on_event_end(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an event ends."""
        span = self._spans.pop(event_id, None)
        start_time = self._start_times.pop(event_id, None)
        
        if span:
            if start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("llama_index.duration_ms", duration_ms)
            
            if payload:
                self._set_response_attributes(span, event_type, payload)
            
            span.status = SpanStatus.OK
            span.end()
    
    def _get_span_type(self, event_type: str) -> SpanType:
        """Map LlamaIndex event type to span type."""
        event_type_lower = event_type.lower()
        
        if "llm" in event_type_lower:
            return SpanType.LLM
        elif "retriev" in event_type_lower:
            return SpanType.RETRIEVAL
        elif "embed" in event_type_lower:
            return SpanType.EMBEDDING
        elif "query" in event_type_lower:
            return SpanType.CHAIN
        else:
            return SpanType.CHAIN
    
    def _set_payload_attributes(self, span, event_type: str, payload: Dict[str, Any]) -> None:
        """Set span attributes from event payload."""
        if "query" in event_type.lower():
            if "query_str" in payload:
                span.set_attribute("llama_index.query", payload["query_str"][:1000])
        
        if "llm" in event_type.lower():
            if "messages" in payload:
                span.set_attribute("llm.messages", payload["messages"])
            if "model" in payload:
                span.set_attribute("llm.model.name", payload["model"])
        
        if "retriev" in event_type.lower():
            if "query_str" in payload:
                span.set_attribute("retriever.query", payload["query_str"][:1000])
    
    def _set_response_attributes(self, span, event_type: str, payload: Dict[str, Any]) -> None:
        """Set span attributes from response payload."""
        if "response" in payload:
            response = payload["response"]
            if hasattr(response, "response"):
                span.set_attribute("llama_index.response", str(response.response)[:1000])
        
        if "nodes" in payload:
            span.set_attribute("llama_index.node_count", len(payload["nodes"]))
        
        if "completion" in payload:
            span.set_attribute("llm.completion", str(payload["completion"])[:1000])


def instrument_llama_index() -> LlamaIndexCallbackHandler:
    """
    Create and return a LlamaIndex callback handler for tracing.
    
    Usage:
        from genai_traces.instrumentation.frameworks.llama_index import instrument_llama_index
        from llama_index.core import Settings
        
        handler = instrument_llama_index()
        Settings.callback_manager.add_handler(handler)
    
    Returns:
        LlamaIndexCallbackHandler instance
    """
    return LlamaIndexCallbackHandler()
