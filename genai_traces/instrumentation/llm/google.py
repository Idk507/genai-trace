"""
Google AI (Gemini) and Vertex AI instrumentation for GenAI-Traces.
"""

import functools
import time
from typing import Any, Optional

from ...core.tracer import get_tracer
from ...core.types import SpanType

_original_generate = None
_original_generate_async = None
_instrumented = False


def instrument_google() -> None:
    """
    Instrument Google Generative AI for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.llm.google import instrument_google
        instrument_google()
        
        # All subsequent Google AI calls are automatically traced
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello")
    """
    global _original_generate, _original_generate_async, _instrumented
    
    if _instrumented:
        return
    
    try:
        import google.generativeai as genai
        from google.generativeai import GenerativeModel
    except ImportError:
        return
    
    _original_generate = GenerativeModel.generate_content
    
    @functools.wraps(_original_generate)
    def traced_generate(self, *args, **kwargs):
        return _trace_google_call(_original_generate, self, *args, **kwargs)
    
    GenerativeModel.generate_content = traced_generate
    
    try:
        _original_generate_async = GenerativeModel.generate_content_async
        
        @functools.wraps(_original_generate_async)
        async def traced_generate_async(self, *args, **kwargs):
            return await _trace_google_call_async(_original_generate_async, self, *args, **kwargs)
        
        GenerativeModel.generate_content_async = traced_generate_async
    except AttributeError:
        pass
    
    _instrumented = True


def uninstrument_google() -> None:
    """Remove Google AI instrumentation."""
    global _original_generate, _original_generate_async, _instrumented
    
    if not _instrumented:
        return
    
    try:
        from google.generativeai import GenerativeModel
        
        if _original_generate:
            GenerativeModel.generate_content = _original_generate
        if _original_generate_async:
            GenerativeModel.generate_content_async = _original_generate_async
    except ImportError:
        pass
    
    _instrumented = False


def _trace_google_call(original_fn, self, *args, **kwargs):
    """Wrap a sync Google AI call with tracing."""
    tracer = get_tracer()
    model_name = getattr(self, "_model_name", "gemini-pro")
    
    with tracer.start_as_current_span(f"google.{model_name}", SpanType.LLM) as span:
        span.set_attribute("llm.provider", "google")
        span.set_attribute("llm.model.name", model_name)
        
        if args:
            prompt = args[0]
            if isinstance(prompt, str):
                span.set_attribute("llm.prompt", prompt[:1000])
            elif isinstance(prompt, list):
                span.set_attribute("llm.messages", prompt)
        
        generation_config = kwargs.get("generation_config", {})
        if hasattr(generation_config, "temperature"):
            span.set_attribute("llm.request.temperature", generation_config.temperature)
        if hasattr(generation_config, "max_output_tokens"):
            span.set_attribute("llm.request.max_tokens", generation_config.max_output_tokens)
        
        start_time = time.perf_counter()
        
        try:
            response = original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            
            if hasattr(response, "text"):
                span.set_attribute("llm.completion", response.text[:1000])
            
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    span.set_attribute("llm.prompt.tokens", usage.prompt_token_count)
                if hasattr(usage, "candidates_token_count"):
                    span.set_attribute("llm.completion.tokens", usage.candidates_token_count)
                if hasattr(usage, "total_token_count"):
                    span.set_attribute("llm.total_tokens", usage.total_token_count)
            
            from ...core.types import SpanStatus
            span.status = SpanStatus.OK
            
            return response
            
        except Exception as e:
            span.record_exception(e)
            raise


async def _trace_google_call_async(original_fn, self, *args, **kwargs):
    """Wrap an async Google AI call with tracing."""
    tracer = get_tracer()
    model_name = getattr(self, "_model_name", "gemini-pro")
    
    async with tracer.start_as_current_span_async(f"google.{model_name}", SpanType.LLM) as span:
        span.set_attribute("llm.provider", "google")
        span.set_attribute("llm.model.name", model_name)
        
        if args:
            prompt = args[0]
            if isinstance(prompt, str):
                span.set_attribute("llm.prompt", prompt[:1000])
        
        start_time = time.perf_counter()
        
        try:
            response = await original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            
            if hasattr(response, "text"):
                span.set_attribute("llm.completion", response.text[:1000])
            
            from ...core.types import SpanStatus
            span.status = SpanStatus.OK
            
            return response
            
        except Exception as e:
            span.record_exception(e)
            raise


def instrument_vertex_ai() -> None:
    """
    Instrument Vertex AI for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.llm.google import instrument_vertex_ai
        instrument_vertex_ai()
    """
    try:
        from vertexai.generative_models import GenerativeModel
        
        original_generate = GenerativeModel.generate_content
        
        @functools.wraps(original_generate)
        def traced_generate(self, *args, **kwargs):
            return _trace_vertex_call(original_generate, self, *args, **kwargs)
        
        GenerativeModel.generate_content = traced_generate
    except ImportError:
        pass


def _trace_vertex_call(original_fn, self, *args, **kwargs):
    """Wrap a Vertex AI call with tracing."""
    tracer = get_tracer()
    model_name = getattr(self, "_model_name", "gemini-pro")
    
    with tracer.start_as_current_span(f"vertex.{model_name}", SpanType.LLM) as span:
        span.set_attribute("llm.provider", "vertex_ai")
        span.set_attribute("llm.model.name", model_name)
        
        start_time = time.perf_counter()
        
        try:
            response = original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            
            if hasattr(response, "text"):
                span.set_attribute("llm.completion", response.text[:1000])
            
            from ...core.types import SpanStatus
            span.status = SpanStatus.OK
            
            return response
            
        except Exception as e:
            span.record_exception(e)
            raise
