"""
Azure OpenAI instrumentation for GenAI-Traces.

Monkey-patches Azure OpenAI client to automatically trace LLM calls.
"""

import functools
import time
from typing import Any, Optional

from ...core.tracer import get_tracer
from ...core.types import SpanType
from ...telemetry.cost.estimator import CostEstimator

_original_create = None
_original_create_async = None
_instrumented = False


def instrument_azure_openai() -> None:
    """
    Instrument Azure OpenAI client for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.llm.azure import instrument_azure_openai
        instrument_azure_openai()
        
        # All subsequent Azure OpenAI calls are automatically traced
        client = AzureOpenAI(...)
        response = client.chat.completions.create(...)
    """
    global _original_create, _original_create_async, _instrumented
    
    if _instrumented:
        return
    
    try:
        from openai import AzureOpenAI, AsyncAzureOpenAI
    except ImportError:
        return
    
    _original_create = AzureOpenAI.chat.completions.create
    
    @functools.wraps(_original_create)
    def traced_create(self, *args, **kwargs):
        return _trace_azure_call(_original_create, self, *args, **kwargs)
    
    AzureOpenAI.chat.completions.create = traced_create
    
    try:
        _original_create_async = AsyncAzureOpenAI.chat.completions.create
        
        @functools.wraps(_original_create_async)
        async def traced_create_async(self, *args, **kwargs):
            return await _trace_azure_call_async(_original_create_async, self, *args, **kwargs)
        
        AsyncAzureOpenAI.chat.completions.create = traced_create_async
    except Exception:
        pass
    
    _instrumented = True


def uninstrument_azure_openai() -> None:
    """Remove Azure OpenAI instrumentation."""
    global _original_create, _original_create_async, _instrumented
    
    if not _instrumented:
        return
    
    try:
        from openai import AzureOpenAI, AsyncAzureOpenAI
        
        if _original_create:
            AzureOpenAI.chat.completions.create = _original_create
        if _original_create_async:
            AsyncAzureOpenAI.chat.completions.create = _original_create_async
    except ImportError:
        pass
    
    _instrumented = False


def _trace_azure_call(original_fn, self, *args, **kwargs):
    """Wrap a sync Azure OpenAI call with tracing."""
    tracer = get_tracer()
    model = kwargs.get("model", "gpt-4")
    
    with tracer.start_as_current_span(f"azure_openai.{model}", SpanType.LLM) as span:
        span.set_attribute("llm.provider", "azure_openai")
        span.set_attribute("llm.model.name", model)
        
        messages = kwargs.get("messages", [])
        if messages:
            span.set_attribute("llm.messages", messages)
            if messages and messages[-1].get("role") == "user":
                span.set_attribute("llm.prompt", messages[-1].get("content", ""))
        
        for param in ["temperature", "max_tokens", "top_p", "seed"]:
            if param in kwargs:
                span.set_attribute(f"llm.request.{param}", kwargs[param])
        
        span.set_attribute("llm.streaming", kwargs.get("stream", False))
        
        start_time = time.perf_counter()
        
        try:
            response = original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            
            if not kwargs.get("stream", False):
                span.record_response(response)
                
                if hasattr(response, "usage") and response.usage:
                    estimator = CostEstimator()
                    costs = estimator.estimate(
                        model,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                    )
                    span.set_attribute("cost.total_usd", costs["total_cost_usd"])
            
            return response
            
        except Exception as e:
            span.record_exception(e)
            raise


async def _trace_azure_call_async(original_fn, self, *args, **kwargs):
    """Wrap an async Azure OpenAI call with tracing."""
    tracer = get_tracer()
    model = kwargs.get("model", "gpt-4")
    
    async with tracer.start_as_current_span_async(f"azure_openai.{model}", SpanType.LLM) as span:
        span.set_attribute("llm.provider", "azure_openai")
        span.set_attribute("llm.model.name", model)
        
        messages = kwargs.get("messages", [])
        if messages:
            span.set_attribute("llm.messages", messages)
            if messages and messages[-1].get("role") == "user":
                span.set_attribute("llm.prompt", messages[-1].get("content", ""))
        
        start_time = time.perf_counter()
        
        try:
            response = await original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            
            if not kwargs.get("stream", False):
                span.record_response(response)
            
            return response
            
        except Exception as e:
            span.record_exception(e)
            raise
