"""
OpenAI auto-instrumentation.

Patches openai.chat.completions.create (sync and async) so all calls
are automatically traced without any user code changes.

Usage:
    from genai_traces import auto_instrument
    auto_instrument(providers=["openai"])
    # All subsequent openai calls are traced
"""

import time
from typing import Any, Optional

from ...core.tracer import get_tracer, _get_tracer_safe
from ...core.types import SpanType, SpanStatus

_original_create = None
_original_acreate = None
_instrumented = False


def instrument_openai():
    """
    Instrument the OpenAI client library.
    
    Patches chat.completions.create and chat.completions.acreate
    to automatically create spans for all calls.
    """
    global _original_create, _original_acreate, _instrumented
    
    if _instrumented:
        return
    
    try:
        import openai
        from openai.resources.chat import completions
        
        # Store original methods
        _original_create = completions.Completions.create
        
        # Check if async method exists
        if hasattr(completions.AsyncCompletions, "create"):
            _original_acreate = completions.AsyncCompletions.create
        
        # Patch sync method
        def patched_create(self, *args, **kwargs):
            return _traced_call(_original_create, self, *args, **kwargs)
        
        completions.Completions.create = patched_create
        
        # Patch async method
        if _original_acreate:
            async def patched_acreate(self, *args, **kwargs):
                return await _traced_async_call(_original_acreate, self, *args, **kwargs)
            
            completions.AsyncCompletions.create = patched_acreate
        
        _instrumented = True
        
    except ImportError:
        pass  # OpenAI not installed
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
        name=f"openai.chat.{model}",
        span_type=SpanType.CHAT,
        attributes={"llm.provider": "openai", "llm.model.name": model}
    ) as span:
        # Capture messages if enabled
        messages = kwargs.get("messages", [])
        if tracer.config.enable_prompt_capture:
            span.set_attribute("llm.messages", _sanitize_messages(messages))
        
        # Capture model params
        for param in ("temperature", "max_tokens", "top_p", "seed", "stop"):
            if param in kwargs:
                span.set_attribute(f"llm.request.{param}", kwargs[param])
        
        span.set_attribute("llm.streaming", stream)
        
        t0 = time.perf_counter_ns()
        
        try:
            response = fn(self_client, *args, **kwargs)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            
            span.set_attribute("llm.duration_ms", elapsed)
            
            if stream:
                # For streaming, wrap the response to capture tokens
                return _wrap_stream(response, span, tracer, elapsed)
            else:
                span.record_response(response)
                _add_cost_attributes(span, model, response)
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
        name=f"openai.chat.{model}",
        span_type=SpanType.CHAT,
        attributes={"llm.provider": "openai", "llm.model.name": model}
    ) as span:
        messages = kwargs.get("messages", [])
        if tracer.config.enable_prompt_capture:
            span.set_attribute("llm.messages", _sanitize_messages(messages))
        
        for param in ("temperature", "max_tokens", "top_p", "seed", "stop"):
            if param in kwargs:
                span.set_attribute(f"llm.request.{param}", kwargs[param])
        
        span.set_attribute("llm.streaming", stream)
        
        t0 = time.perf_counter_ns()
        
        try:
            response = await fn(self_client, *args, **kwargs)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            
            span.set_attribute("llm.duration_ms", elapsed)
            
            if stream:
                return _wrap_async_stream(response, span, tracer, elapsed)
            else:
                span.record_response(response)
                _add_cost_attributes(span, model, response)
                return response
                
        except Exception as e:
            span.record_exception(e)
            raise


def _wrap_stream(stream, span, tracer, start_elapsed):
    """Wrap a sync stream to capture completion."""
    chunks = []
    first_token_time = None
    
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter_ns()
            span.set_attribute("llm.ttft_ms", (first_token_time - start_elapsed) / 1e6)
        
        chunks.append(chunk)
        yield chunk
    
    # After stream completes, update span
    _finalize_stream(chunks, span)


async def _wrap_async_stream(stream, span, tracer, start_elapsed):
    """Wrap an async stream to capture completion."""
    chunks = []
    first_token_time = None
    
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter_ns()
            span.set_attribute("llm.ttft_ms", (first_token_time - start_elapsed) / 1e6)
        
        chunks.append(chunk)
        yield chunk
    
    _finalize_stream(chunks, span)


def _finalize_stream(chunks, span):
    """Finalize span after stream completes."""
    if not chunks:
        return
    
    # Reconstruct completion from chunks
    completion_parts = []
    for chunk in chunks:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                completion_parts.append(delta.content)
    
    completion = "".join(completion_parts)
    span.set_attribute("llm.completion", completion)
    
    # Get usage from last chunk if available
    last_chunk = chunks[-1]
    if hasattr(last_chunk, "usage") and last_chunk.usage:
        u = last_chunk.usage
        span.set_attribute("llm.prompt_tokens", getattr(u, "prompt_tokens", 0))
        span.set_attribute("llm.completion_tokens", getattr(u, "completion_tokens", 0))
        span.set_attribute("llm.total_tokens", getattr(u, "total_tokens", 0))
    
    span.status = SpanStatus.OK


def _add_cost_attributes(span, model: str, response):
    """Add cost attributes to span."""
    usage = getattr(response, "usage", None)
    if usage:
        try:
            from ...telemetry.cost.estimator import CostEstimator
            costs = CostEstimator().estimate(
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
            for k, v in costs.items():
                span.set_attribute(f"cost.{k}", v)
        except Exception:
            pass


def _sanitize_messages(messages: list) -> list:
    """Sanitize messages for storage (limit length, etc.)."""
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
