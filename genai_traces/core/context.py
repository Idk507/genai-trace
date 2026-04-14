"""
Thread-safe and async-safe context propagation using Python's contextvars.

ContextVar values are automatically scoped per coroutine/thread,
so nested spans in different async tasks don't conflict.
"""

from contextvars import ContextVar
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .span import Span

# The active span for the current coroutine/thread
_current_span: ContextVar[Optional["Span"]] = ContextVar(
    "_current_span", default=None
)

# Convenience: current trace ID (avoids dereferencing span)
_current_trace_id: ContextVar[Optional[str]] = ContextVar(
    "_current_trace_id", default=None
)

# Conversation context (set by set_conversation_context())
_conversation_id: ContextVar[Optional[str]] = ContextVar(
    "_conversation_id", default=None
)
_conversation_turn: ContextVar[int] = ContextVar(
    "_conversation_turn", default=0
)
_user_id: ContextVar[Optional[str]] = ContextVar(
    "_user_id", default=None
)

# A/B experiment context (set by activate_experiment())
_experiment_id: ContextVar[Optional[str]] = ContextVar(
    "_experiment_id", default=None
)
_variant_id: ContextVar[Optional[str]] = ContextVar(
    "_variant_id", default=None
)


def get_current_span() -> Optional["Span"]:
    """Get the currently active span, if any."""
    return _current_span.get(None)


def set_current_span(span: Optional["Span"]) -> None:
    """Set the currently active span."""
    _current_span.set(span)


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID, if any."""
    span = _current_span.get(None)
    return span.trace_id if span else None


def get_current_span_id() -> Optional[str]:
    """Get the current span ID, if any."""
    span = _current_span.get(None)
    return span.span_id if span else None


def set_conversation_context(
    conversation_id: str,
    turn: int = 1,
    user_id: Optional[str] = None
) -> None:
    """
    Set conversation metadata that auto-attaches to all subsequent spans.
    
    Args:
        conversation_id: Unique identifier for the conversation
        turn: Turn number in the conversation (1-indexed)
        user_id: Optional user identifier
    """
    _conversation_id.set(conversation_id)
    _conversation_turn.set(turn)
    if user_id:
        _user_id.set(user_id)


def get_conversation_context() -> dict:
    """Get the current conversation context."""
    return {
        "conversation_id": _conversation_id.get(None),
        "turn": _conversation_turn.get(0),
        "user_id": _user_id.get(None),
    }


def clear_conversation_context() -> None:
    """Clear the conversation context."""
    _conversation_id.set(None)
    _conversation_turn.set(0)
    _user_id.set(None)


def increment_conversation_turn() -> int:
    """Increment the conversation turn counter and return the new value."""
    current = _conversation_turn.get(0)
    new_turn = current + 1
    _conversation_turn.set(new_turn)
    return new_turn


def set_experiment_context(
    experiment_id: str,
    variant_id: str
) -> None:
    """
    Set A/B experiment context that auto-attaches to all subsequent spans.
    
    Args:
        experiment_id: Unique identifier for the experiment
        variant_id: Identifier for the assigned variant
    """
    _experiment_id.set(experiment_id)
    _variant_id.set(variant_id)


def get_experiment_context() -> dict:
    """Get the current experiment context."""
    return {
        "experiment_id": _experiment_id.get(None),
        "variant_id": _variant_id.get(None),
    }


def clear_experiment_context() -> None:
    """Clear the experiment context."""
    _experiment_id.set(None)
    _variant_id.set(None)


def inject_context_into_span(span: "Span") -> None:
    """
    Attach all active context values to the span.
    
    This is called automatically when a span is created to ensure
    conversation and experiment context is captured.
    """
    # Conversation context
    conv_id = _conversation_id.get(None)
    if conv_id:
        span.set_attribute("conversation.id", conv_id)
        span.set_attribute("conversation.turn", _conversation_turn.get(0))
    
    user_id = _user_id.get(None)
    if user_id:
        span.set_attribute("user.id", user_id)
    
    # Experiment context
    exp_id = _experiment_id.get(None)
    if exp_id:
        span.experiment_id = exp_id
        span.variant_id = _variant_id.get(None)
        span.set_attribute("experiment.id", exp_id)
        span.set_attribute("experiment.variant", span.variant_id)
