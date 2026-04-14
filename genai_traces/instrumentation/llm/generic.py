"""
Generic LLM wrapper for GenAI-Traces.

Wrap any callable LLM client with tracing.
"""

import functools
import time
from typing import Any, Callable, Optional, Dict

from ...core.tracer import get_tracer
from ...core.types import SpanType


def wrap_llm_call(
    func: Callable,
    provider: str = "custom",
    model: str = "unknown",
    extract_prompt: Optional[Callable[[tuple, dict], str]] = None,
    extract_completion: Optional[Callable[[Any], str]] = None,
    extract_tokens: Optional[Callable[[Any], Dict[str, int]]] = None,
) -> Callable:
    """
    Wrap any LLM call function with tracing.
    
    Usage:
        def my_llm_call(prompt, **kwargs):
            return custom_api.generate(prompt, **kwargs)
        
        traced_call = wrap_llm_call(
            my_llm_call,
            provider="custom_api",
            model="my-model",
            extract_prompt=lambda args, kwargs: args[0],
            extract_completion=lambda response: response.text,
        )
        
        response = traced_call("Hello world")
    
    Args:
        func: The LLM call function to wrap
        provider: Provider name for tracing
        model: Model name for tracing
        extract_prompt: Function to extract prompt from args/kwargs
        extract_completion: Function to extract completion from response
        extract_tokens: Function to extract token counts from response
        
    Returns:
        Wrapped function with tracing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        with tracer.start_as_current_span(f"{provider}.{model}", SpanType.LLM) as span:
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model.name", model)
            
            if extract_prompt:
                try:
                    prompt = extract_prompt(args, kwargs)
                    span.set_attribute("llm.prompt", str(prompt)[:1000])
                except Exception:
                    pass
            
            start_time = time.perf_counter()
            
            try:
                response = func(*args, **kwargs)
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                
                if extract_completion:
                    try:
                        completion = extract_completion(response)
                        span.set_attribute("llm.completion", str(completion)[:1000])
                    except Exception:
                        pass
                
                if extract_tokens:
                    try:
                        tokens = extract_tokens(response)
                        if "prompt_tokens" in tokens:
                            span.set_attribute("llm.prompt.tokens", tokens["prompt_tokens"])
                        if "completion_tokens" in tokens:
                            span.set_attribute("llm.completion.tokens", tokens["completion_tokens"])
                        if "total_tokens" in tokens:
                            span.set_attribute("llm.total_tokens", tokens["total_tokens"])
                    except Exception:
                        pass
                
                from ...core.types import SpanStatus
                span.status = SpanStatus.OK
                
                return response
                
            except Exception as e:
                span.record_exception(e)
                raise
    
    return wrapper


def wrap_llm_call_async(
    func: Callable,
    provider: str = "custom",
    model: str = "unknown",
    extract_prompt: Optional[Callable[[tuple, dict], str]] = None,
    extract_completion: Optional[Callable[[Any], str]] = None,
    extract_tokens: Optional[Callable[[Any], Dict[str, int]]] = None,
) -> Callable:
    """
    Wrap any async LLM call function with tracing.
    
    Args:
        func: The async LLM call function to wrap
        provider: Provider name for tracing
        model: Model name for tracing
        extract_prompt: Function to extract prompt from args/kwargs
        extract_completion: Function to extract completion from response
        extract_tokens: Function to extract token counts from response
        
    Returns:
        Wrapped async function with tracing
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        async with tracer.start_as_current_span_async(f"{provider}.{model}", SpanType.LLM) as span:
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model.name", model)
            
            if extract_prompt:
                try:
                    prompt = extract_prompt(args, kwargs)
                    span.set_attribute("llm.prompt", str(prompt)[:1000])
                except Exception:
                    pass
            
            start_time = time.perf_counter()
            
            try:
                response = await func(*args, **kwargs)
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                
                if extract_completion:
                    try:
                        completion = extract_completion(response)
                        span.set_attribute("llm.completion", str(completion)[:1000])
                    except Exception:
                        pass
                
                if extract_tokens:
                    try:
                        tokens = extract_tokens(response)
                        for key, value in tokens.items():
                            span.set_attribute(f"llm.{key}", value)
                    except Exception:
                        pass
                
                from ...core.types import SpanStatus
                span.status = SpanStatus.OK
                
                return response
                
            except Exception as e:
                span.record_exception(e)
                raise
    
    return wrapper


class TracedLLMClient:
    """
    A wrapper class for any LLM client that adds tracing.
    
    Usage:
        client = TracedLLMClient(
            my_client,
            provider="custom",
            model="my-model",
        )
        response = client.generate("Hello")
    """
    
    def __init__(
        self,
        client: Any,
        provider: str = "custom",
        model: str = "unknown",
        call_method: str = "generate",
    ):
        self._client = client
        self._provider = provider
        self._model = model
        self._call_method = call_method
    
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        
        if name == self._call_method and callable(attr):
            return wrap_llm_call(
                attr,
                provider=self._provider,
                model=self._model,
            )
        
        return attr
    
    def call(self, *args, **kwargs) -> Any:
        """Make a traced call to the underlying client."""
        method = getattr(self._client, self._call_method)
        wrapped = wrap_llm_call(
            method,
            provider=self._provider,
            model=self._model,
        )
        return wrapped(*args, **kwargs)
