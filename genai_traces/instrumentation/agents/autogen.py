"""
AutoGen multi-agent tracing for GenAI-Traces.

Traces conversations and interactions between AutoGen agents.
"""

import functools
import time
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass, field

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


_instrumented = False


def instrument_autogen() -> None:
    """
    Instrument AutoGen for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.agents.autogen import instrument_autogen
        instrument_autogen()
        
        # All AutoGen agent interactions are now traced
    """
    global _instrumented
    
    if _instrumented:
        return
    
    try:
        import autogen
    except ImportError:
        return
    
    if hasattr(autogen, "ConversableAgent"):
        original_generate_reply = autogen.ConversableAgent.generate_reply
        
        @functools.wraps(original_generate_reply)
        def traced_generate_reply(self, messages=None, sender=None, **kwargs):
            return _trace_generate_reply(original_generate_reply, self, messages, sender, **kwargs)
        
        autogen.ConversableAgent.generate_reply = traced_generate_reply
    
    _instrumented = True


def _trace_generate_reply(original_fn, self, messages, sender, **kwargs):
    """Wrap AutoGen generate_reply with tracing."""
    tracer = get_tracer()
    
    agent_name = getattr(self, "name", "agent")
    sender_name = getattr(sender, "name", "unknown") if sender else "user"
    
    with tracer.start_as_current_span(f"autogen.reply.{agent_name}", SpanType.AGENT) as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.type", "autogen")
        span.set_attribute("agent.sender", sender_name)
        
        if messages:
            span.set_attribute("agent.message_count", len(messages))
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    span.set_attribute("agent.last_message", str(last_message.get("content", ""))[:500])
        
        start_time = time.perf_counter()
        
        try:
            result = original_fn(self, messages, sender, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("agent.duration_ms", duration_ms)
            
            if result:
                span.set_attribute("agent.reply", str(result)[:1000])
            
            span.status = SpanStatus.OK
            return result
            
        except Exception as e:
            span.record_exception(e)
            raise


@dataclass
class AgentMessage:
    """Represents a message in an AutoGen conversation."""
    sender: str
    receiver: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoGenTracer:
    """
    Manual tracer for AutoGen multi-agent conversations.
    
    Usage:
        tracer = AutoGenTracer()
        
        with tracer.trace_conversation("task_solving") as conv:
            conv.record_message("user", "assistant", "Hello")
            conv.record_message("assistant", "user", "Hi there!")
    """
    
    def __init__(self):
        self._tracer = get_tracer()
    
    def trace_conversation(self, name: str = "conversation"):
        """Start tracing a multi-agent conversation."""
        return ConversationContext(self._tracer, name)
    
    def trace_agent(self, agent_name: str):
        """Start tracing a single agent's actions."""
        return AgentContext(self._tracer, agent_name)


class ConversationContext:
    """Context manager for tracing a multi-agent conversation."""
    
    def __init__(self, tracer, name: str):
        self._tracer = tracer
        self._name = name
        self._span = None
        self._messages: List[AgentMessage] = []
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(f"autogen.conversation.{self._name}", SpanType.AGENT)
        self._span.set_attribute("conversation.name", self._name)
        self._span.set_attribute("conversation.type", "autogen")
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("conversation.duration_ms", duration_ms)
            self._span.set_attribute("conversation.message_count", len(self._messages))
            
            participants = set()
            for msg in self._messages:
                participants.add(msg.sender)
                participants.add(msg.receiver)
            self._span.set_attribute("conversation.participants", list(participants))
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
    
    def record_message(self, sender: str, receiver: str, content: str, **metadata) -> None:
        """Record a message in the conversation."""
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            metadata=metadata,
        )
        self._messages.append(message)
        
        if self._span:
            msg_span = self._tracer.start_span(
                f"autogen.message.{sender}->{receiver}",
                SpanType.AGENT
            )
            msg_span.set_attribute("message.sender", sender)
            msg_span.set_attribute("message.receiver", receiver)
            msg_span.set_attribute("message.content", content[:500])
            msg_span.status = SpanStatus.OK
            msg_span.end()


class AgentContext:
    """Context manager for tracing a single agent."""
    
    def __init__(self, tracer, agent_name: str):
        self._tracer = tracer
        self._agent_name = agent_name
        self._span = None
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(f"autogen.agent.{self._agent_name}", SpanType.AGENT)
        self._span.set_attribute("agent.name", self._agent_name)
        self._span.set_attribute("agent.type", "autogen")
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("agent.duration_ms", duration_ms)
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the agent span."""
        if self._span:
            self._span.set_attribute(f"agent.{key}", value)
