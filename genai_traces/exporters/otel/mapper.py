"""
Span mapper for OpenTelemetry compatibility.

Maps GenAI-Traces spans to OpenTelemetry format.
"""

from typing import Any, Dict, Optional
from datetime import datetime


class SpanMapper:
    """
    Maps GenAI-Traces spans to OpenTelemetry spans.
    
    Usage:
        mapper = SpanMapper()
        otel_span = mapper.to_otel_span(genai_span)
    """
    
    ATTRIBUTE_MAPPING = {
        "llm.model.name": "gen_ai.request.model",
        "llm.provider": "gen_ai.system",
        "llm.prompt": "gen_ai.prompt",
        "llm.completion": "gen_ai.completion",
        "llm.prompt.tokens": "gen_ai.usage.prompt_tokens",
        "llm.completion.tokens": "gen_ai.usage.completion_tokens",
        "llm.total_tokens": "gen_ai.usage.total_tokens",
        "llm.request.temperature": "gen_ai.request.temperature",
        "llm.request.max_tokens": "gen_ai.request.max_tokens",
        "cost.total_usd": "gen_ai.usage.cost",
    }
    
    def to_otel_span(self, span: Any) -> Optional[Any]:
        """
        Convert a GenAI-Traces span to OpenTelemetry format.
        
        Args:
            span: GenAI-Traces span
            
        Returns:
            OpenTelemetry ReadableSpan or None
        """
        try:
            from opentelemetry.sdk.trace import ReadableSpan
            from opentelemetry.trace import SpanContext, TraceFlags
            from opentelemetry.trace.status import Status, StatusCode
        except ImportError:
            return None
        
        span_dict = span.to_dict() if hasattr(span, "to_dict") else span
        
        attributes = self._map_attributes(span_dict.get("attributes", {}))
        
        return span_dict
    
    def _map_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Map GenAI-Traces attributes to OTel semantic conventions."""
        mapped = {}
        
        for key, value in attributes.items():
            otel_key = self.ATTRIBUTE_MAPPING.get(key, key)
            mapped[otel_key] = value
        
        return mapped
    
    def from_otel_span(self, otel_span: Any) -> Dict[str, Any]:
        """
        Convert an OpenTelemetry span to GenAI-Traces format.
        
        Args:
            otel_span: OpenTelemetry span
            
        Returns:
            Dictionary in GenAI-Traces format
        """
        reverse_mapping = {v: k for k, v in self.ATTRIBUTE_MAPPING.items()}
        
        attributes = {}
        if hasattr(otel_span, "attributes"):
            for key, value in otel_span.attributes.items():
                genai_key = reverse_mapping.get(key, key)
                attributes[genai_key] = value
        
        return {
            "trace_id": format(otel_span.context.trace_id, "032x") if hasattr(otel_span, "context") else "",
            "span_id": format(otel_span.context.span_id, "016x") if hasattr(otel_span, "context") else "",
            "name": otel_span.name if hasattr(otel_span, "name") else "",
            "attributes": attributes,
        }
    
    def get_semantic_conventions(self) -> Dict[str, str]:
        """Get the attribute mapping for reference."""
        return self.ATTRIBUTE_MAPPING.copy()
