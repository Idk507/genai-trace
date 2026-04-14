"""
Span dataclass representing a single traced operation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import traceback

from .types import SpanType, SpanStatus


@dataclass
class Span:
    """
    Represents a single traced operation in the system.
    
    A span captures timing, status, and metadata about an operation like
    an LLM call, tool execution, or agent step.
    """
    
    # Identity
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    root_span_id: Optional[str] = None

    # Metadata
    name: str = ""
    span_type: SpanType = SpanType.LLM
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    # Status
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None

    # Attributes (typed keys defined in types.py)
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # Prompt management
    prompt_name: Optional[str] = None
    prompt_version: Optional[str] = None
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None

    # Security
    injection_detected: bool = False
    injection_type: Optional[str] = None
    guardrail_actions: List[str] = field(default_factory=list)

    # RAG
    retrieval_chunks: List[Dict] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on the span."""
        self.attributes[key] = value
        return self

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute from the span."""
        return self.attributes.get(key, default)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {}
        })
        return self

    def record_exception(self, exc: Exception) -> "Span":
        """Record an exception on the span."""
        self.status = SpanStatus.ERROR
        self.status_message = str(exc)
        self.set_attribute("error.type", type(exc).__name__)
        self.set_attribute("error.message", str(exc))
        self.set_attribute("error.stack_trace", traceback.format_exc())
        return self

    def record_response(self, response: Any) -> "Span":
        """
        Auto-extract standard fields from OpenAI/Anthropic response objects.
        """
        # Handle OpenAI response format
        if hasattr(response, "usage"):
            u = response.usage
            self.set_attribute("llm.prompt_tokens", getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0))
            self.set_attribute("llm.completion_tokens", getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0))
            total = getattr(u, "total_tokens", 0)
            if not total:
                total = self.get_attribute("llm.prompt_tokens", 0) + self.get_attribute("llm.completion_tokens", 0)
            self.set_attribute("llm.total_tokens", total)
            
            # Handle cache tokens (Anthropic-specific)
            if hasattr(u, "cache_read_input_tokens"):
                self.set_attribute("usage.cache_read_tokens", u.cache_read_input_tokens)
            if hasattr(u, "cache_creation_input_tokens"):
                self.set_attribute("usage.cache_write_tokens", u.cache_creation_input_tokens)

        # Handle OpenAI choices format
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content or ""
                self.set_attribute("llm.completion", content)
            elif hasattr(choice, "text"):
                self.set_attribute("llm.completion", choice.text or "")

        # Handle Anthropic content format
        elif hasattr(response, "content") and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                first_block = response.content[0]
                if hasattr(first_block, "text"):
                    self.set_attribute("llm.completion", first_block.text)
            elif isinstance(response.content, str):
                self.set_attribute("llm.completion", response.content)

        # Handle model info
        if hasattr(response, "model"):
            self.set_attribute("llm.model.name", response.model)

        # Handle response ID
        if hasattr(response, "id"):
            self.set_attribute("llm.response_id", response.id)

        self.status = SpanStatus.OK
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "root_span_id": self.root_span_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links,
            "context": self.context,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "injection_detected": self.injection_detected,
            "injection_type": self.injection_type,
            "guardrail_actions": self.guardrail_actions,
            "retrieval_chunks": self.retrieval_chunks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        """Create a span from a dictionary."""
        span = cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            root_span_id=data.get("root_span_id"),
            name=data.get("name", ""),
            span_type=SpanType(data.get("span_type", "llm")),
            status=SpanStatus(data.get("status", "unset")),
            status_message=data.get("status_message"),
            attributes=data.get("attributes", {}),
            events=data.get("events", []),
            links=data.get("links", []),
            context=data.get("context", {}),
            prompt_name=data.get("prompt_name"),
            prompt_version=data.get("prompt_version"),
            experiment_id=data.get("experiment_id"),
            variant_id=data.get("variant_id"),
            injection_detected=data.get("injection_detected", False),
            injection_type=data.get("injection_type"),
            guardrail_actions=data.get("guardrail_actions", []),
            retrieval_chunks=data.get("retrieval_chunks", []),
        )
        
        if data.get("start_time"):
            span.start_time = datetime.fromisoformat(data["start_time"].replace("Z", "+00:00"))
        if data.get("end_time"):
            span.end_time = datetime.fromisoformat(data["end_time"].replace("Z", "+00:00"))
        if data.get("duration_ms"):
            span.duration_ms = data["duration_ms"]
            
        return span

    def __repr__(self) -> str:
        return (
            f"Span(name={self.name!r}, span_type={self.span_type.value}, "
            f"status={self.status.value}, trace_id={self.trace_id[:8]}...)"
        )
