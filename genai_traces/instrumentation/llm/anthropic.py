"""
Anthropic auto-instrumentation.

Patches anthropic.messages.create (sync and async) so all calls
are automatically traced without any user code changes.

Usage:
    from genai_traces import auto_instrument
    auto_instrument(providers=["anthropic"])
    # All subsequent Anthropic calls are traced
"""

import time
from typing import Any

from ...core.tracer import get_tracer, _get_tracer_safe
from ...core.types import SpanType, SpanStatus

_original_create = None
_original_acreate = None
_instrumented = False


def instrument_anthropic():
    """
    Instrument the Anthropic client library.
    
    Patches messages.create for both sync and async clients
    to automatically create spans for all calls.
    """
    global _original_create, _original_acreate, _instrumented
    
    if _instrumented:
        return
    
    try:
        import anthropic
        
        # Patch sync client
        if hasattr(anthropic, "Anthropic"):
            _original_create = anthropic.Anthropic.messages.create.__func__
            
            def patched_create(self_client, *args, **kwargs):
                return _traced_call(_original_create, self_client, *args, **kwargs)
            
            anthropic.Anthropic.messages.create = patched_create
        
        # Patch async client
        if hasattr(anthropic, "AsyncAnthropic"):
            _original_acreate = anthropic.AsyncAnthropic.messages.create.__func__
            
            async def patched_acreate(self_client, *args, **kwargs):
                return await _traced_async_call(_original_acreate, self_client, *args, **kwargs)
            
            anthropic.AsyncAnthropic.messages.create = patched_acreate
        
        _instrumented = True
        
    except ImportError:
        pass  # Anthropic not installed
    except Exception:
        pass  # Patching failed


def _traced_call(fn, self_client, *args, **kwargs):
    """Traced sync call wrapper."""
    tracer = _get_tracer_safe()
    if tracer is None:
        return fn(self_client, *args, **kwargs)
    
    model = kwargs.get("model", "unknown")
    stream = kwargs.get("stream", False)
    
    with tracer.start_as_current_span(
        name=f"anthropic.messages.{model}",
        span_type=SpanType.CHAT,
        attributes={"llm.provider": "anthropic", "llm.model.name": model}
    ) as span:
        # Capture messages and system prompt
        if tracer.config.enable_prompt_capture:
            messages = kwargs.get("messages", [])
            span.set_attribute("llm.messages", _sanitize_messages(messages))
            system = kwargs.get("system", "")
            if system:
                span.set_attribute("llm.system_prompt", _truncate(system, 4096))
        
        # Capture model params
        for param in ("temperature", "max_tokens", "top_p", "stop_sequences"):
            if param in kwargs:
                span.set_attribute(f"llm.request.{param}", kwargs[param])
        
        span.set_attribute("llm.streaming", stream)
        
        t0 = time.perf_counter_ns()
        
        try:
            response = fn(self_client, *args, **kwargs)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            
            span.set_attribute("llm.duration_ms", elapsed)
            
            if stream:
                return _wrap_stream(response, span, tracer)
            else:
                _record_anthropic_response(span, response, model)
                return response
                
        except Exception as e:
            span.record_exception(e)
            raise


async def _traced_async_call(fn, self_client, *args, **kwargs):
    """Traced async call wrapper."""
    tracer = _get_tracer_safe()
    if tracer is None:
        return await fn(self_client, *args, **kwargs)
    
    model = kwargs.get("model", "unknown")
    stream = kwargs.get("stream", False)
    
    async with tracer.start_as_current_span_async(
        name=f"anthropic.messages.{model}",
        span_type=SpanType.CHAT,
        attributes={"llm.provider": "anthropic", "llm.model.name": model}
    ) as span:
        if tracer.config.enable_prompt_capture:
            messages = kwargs.get("messages", [])
            span.set_attribute("llm.messages", _sanitize_messages(messages))
            system = kwargs.get("system", "")
            if system:
                span.set_attribute("llm.system_prompt", _truncate(system, 4096))
        
        for param in ("temperature", "max_tokens", "top_p", "stop_sequences"):
            if param in kwargs:
                span.set_attribute(f"llm.request.{param}", kwargs[param])
        
        span.set_attribute("llm.streaming", stream)
        
        t0 = time.perf_counter_ns()
        
        try:
            response = await fn(self_client, *args, **kwargs)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            
            span.set_attribute("llm.duration_ms", elapsed)
            
            if stream:
                return _wrap_async_stream(response, span, tracer)
            else:
                _record_anthropic_response(span, response, model)
                return response
                
        except Exception as e:
            span.record_exception(e)
            raise


def _record_anthropic_response(span, response, model: str):
    """Record Anthropic response details on span."""
    # Extract usage
    if hasattr(response, "usage"):
        u = response.usage
        input_tokens = getattr(u, "input_tokens", 0)
        output_tokens = getattr(u, "output_tokens", 0)
        
        span.set_attribute("llm.prompt_tokens", input_tokens)
        span.set_attribute("llm.completion_tokens", output_tokens)
        span.set_attribute("llm.total_tokens", input_tokens + output_tokens)
        
        # Cache tokens (Anthropic-specific)
        if hasattr(u, "cache_read_input_tokens"):
            span.set_attribute("usage.cache_read_tokens", u.cache_read_input_tokens)
        if hasattr(u, "cache_creation_input_tokens"):
            span.set_attribute("usage.cache_write_tokens", u.cache_creation_input_tokens)
        
        # Calculate cost
        try:
            from ...telemetry.cost.estimator import CostEstimator
            cached_tokens = getattr(u, "cache_read_input_tokens", 0) or 0
            costs = CostEstimator().estimate(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
            for k, v in costs.items():
                span.set_attribute(f"cost.{k}", v)
        except Exception:
            pass
    
    # Extract completion
    if hasattr(response, "content") and response.content:
        if isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, "text"):
                span.set_attribute("llm.completion", first_block.text)
    
    # Extract stop reason
    if hasattr(response, "stop_reason"):
        span.set_attribute("llm.stop_reason", response.stop_reason)
    
    span.status = SpanStatus.OK


def _wrap_stream(stream, span, tracer):
    """Wrap a sync stream to capture completion."""
    chunks = []
    
    for event in stream:
        chunks.append(event)
        yield event
    
    _finalize_stream(chunks, span)


async def _wrap_async_stream(stream, span, tracer):
    """Wrap an async stream to capture completion."""
    chunks = []
    
    async for event in stream:
        chunks.append(event)
        yield event
    
    _finalize_stream(chunks, span)


def _finalize_stream(events, span):
    """Finalize span after stream completes."""
    if not events:
        return
    
    # Reconstruct completion from events
    completion_parts = []
    input_tokens = 0
    output_tokens = 0
    
    for event in events:
        # Handle content block delta
        if hasattr(event, "type"):
            if event.type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and hasattr(delta, "text"):
                    completion_parts.append(delta.text)
            elif event.type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    output_tokens = getattr(usage, "output_tokens", 0)
            elif event.type == "message_start":
                message = getattr(event, "message", None)
                if message and hasattr(message, "usage"):
                    input_tokens = getattr(message.usage, "input_tokens", 0)
    
    completion = "".join(completion_parts)
    span.set_attribute("llm.completion", completion)
    span.set_attribute("llm.prompt_tokens", input_tokens)
    span.set_attribute("llm.completion_tokens", output_tokens)
    span.set_attribute("llm.total_tokens", input_tokens + output_tokens)
    
    span.status = SpanStatus.OK


def _sanitize_messages(messages: list) -> list:
    """Sanitize messages for storage."""
    sanitized = []
    for msg in messages:
        if isinstance(msg, dict):
            sanitized.append({
                "role": msg.get("role", ""),
                "content": _truncate(str(msg.get("content", "")), 4096),
            })
        else:
            sanitized.append(str(msg)[:4096])
    return sanitized


def _truncate(s: str, max_len: int) -> str:
    """Truncate string to max length."""
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."
