"""
Vercel AI SDK bridge for GenAI-Traces.

Provides tracing integration for Vercel AI SDK (primarily TypeScript,
but this provides Python utilities for hybrid applications).
"""

import json
import time
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


@dataclass
class VercelAIEvent:
    """Represents a Vercel AI SDK event for tracing."""
    event_type: str
    model: str
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    completion: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def instrument_vercel_ai() -> None:
    """
    Set up Vercel AI SDK tracing bridge.
    
    This creates a bridge for receiving trace events from Vercel AI SDK
    (TypeScript) and recording them in GenAI-Traces.
    
    Usage:
        from genai_traces.instrumentation.frameworks.vercel_ai import instrument_vercel_ai
        instrument_vercel_ai()
    """
    pass


def record_vercel_event(event: VercelAIEvent) -> None:
    """
    Record a Vercel AI SDK event as a span.
    
    Usage:
        from genai_traces.instrumentation.frameworks.vercel_ai import record_vercel_event, VercelAIEvent
        
        event = VercelAIEvent(
            event_type="generate",
            model="gpt-4",
            prompt="Hello",
            completion="Hi there!",
            tokens={"prompt": 10, "completion": 5},
            duration_ms=150.0,
        )
        record_vercel_event(event)
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(f"vercel_ai.{event.event_type}", SpanType.LLM) as span:
        span.set_attribute("llm.provider", "vercel_ai")
        span.set_attribute("llm.model.name", event.model)
        span.set_attribute("vercel_ai.event_type", event.event_type)
        
        if event.prompt:
            span.set_attribute("llm.prompt", event.prompt[:1000])
        
        if event.messages:
            span.set_attribute("llm.messages", event.messages)
        
        if event.completion:
            span.set_attribute("llm.completion", event.completion[:1000])
        
        if event.tokens:
            if "prompt" in event.tokens:
                span.set_attribute("llm.prompt.tokens", event.tokens["prompt"])
            if "completion" in event.tokens:
                span.set_attribute("llm.completion.tokens", event.tokens["completion"])
            if "total" in event.tokens:
                span.set_attribute("llm.total_tokens", event.tokens["total"])
        
        if event.duration_ms:
            span.set_attribute("llm.duration_ms", event.duration_ms)
        
        if event.metadata:
            for key, value in event.metadata.items():
                span.set_attribute(f"vercel_ai.metadata.{key}", value)
        
        if event.error:
            span.set_attribute("error.message", event.error)
            span.status = SpanStatus.ERROR
        else:
            span.status = SpanStatus.OK


def parse_vercel_event(json_data: str) -> VercelAIEvent:
    """
    Parse a JSON event from Vercel AI SDK.
    
    Args:
        json_data: JSON string from Vercel AI SDK telemetry
        
    Returns:
        VercelAIEvent instance
    """
    data = json.loads(json_data)
    
    return VercelAIEvent(
        event_type=data.get("type", "unknown"),
        model=data.get("model", "unknown"),
        prompt=data.get("prompt"),
        messages=data.get("messages"),
        completion=data.get("completion"),
        tokens=data.get("tokens"),
        duration_ms=data.get("duration_ms"),
        error=data.get("error"),
        metadata=data.get("metadata"),
    )


class VercelAIBridge:
    """
    Bridge for receiving Vercel AI SDK telemetry.
    
    This can be used to receive events via HTTP webhook or message queue.
    
    Usage:
        bridge = VercelAIBridge()
        
        # In your webhook handler:
        @app.post("/vercel-ai-telemetry")
        async def handle_telemetry(request):
            event_json = await request.json()
            bridge.record_event(event_json)
    """
    
    def __init__(self):
        self._tracer = get_tracer()
    
    def record_event(self, event_data: Dict[str, Any]) -> None:
        """Record an event from Vercel AI SDK."""
        event = VercelAIEvent(
            event_type=event_data.get("type", "unknown"),
            model=event_data.get("model", "unknown"),
            prompt=event_data.get("prompt"),
            messages=event_data.get("messages"),
            completion=event_data.get("completion"),
            tokens=event_data.get("tokens"),
            duration_ms=event_data.get("duration_ms"),
            error=event_data.get("error"),
            metadata=event_data.get("metadata"),
        )
        record_vercel_event(event)
    
    def record_batch(self, events: List[Dict[str, Any]]) -> None:
        """Record a batch of events from Vercel AI SDK."""
        for event_data in events:
            self.record_event(event_data)
