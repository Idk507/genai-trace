"""
Reranker tracing for GenAI-Traces.

Traces Cohere rerank, cross-encoder, and other reranking operations.
"""

import functools
import time
from typing import Any, Dict, List, Optional, Callable

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


def instrument_cohere_rerank() -> None:
    """
    Instrument Cohere rerank for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.retrieval.reranker import instrument_cohere_rerank
        instrument_cohere_rerank()
    """
    try:
        import cohere
    except ImportError:
        return
    
    original_rerank = cohere.Client.rerank
    
    @functools.wraps(original_rerank)
    def traced_rerank(self, *args, **kwargs):
        return _trace_rerank("cohere", original_rerank, self, *args, **kwargs)
    
    cohere.Client.rerank = traced_rerank


def _trace_rerank(provider: str, original_fn, self, *args, **kwargs):
    """Wrap a rerank call with tracing."""
    tracer = get_tracer()
    
    with tracer.start_as_current_span(f"reranker.{provider}", SpanType.RETRIEVAL) as span:
        span.set_attribute("reranker.provider", provider)
        
        if "query" in kwargs:
            span.set_attribute("reranker.query", kwargs["query"][:500])
        
        if "documents" in kwargs:
            span.set_attribute("reranker.input_count", len(kwargs["documents"]))
        
        if "top_n" in kwargs:
            span.set_attribute("reranker.top_n", kwargs["top_n"])
        
        if "model" in kwargs:
            span.set_attribute("reranker.model", kwargs["model"])
        
        start_time = time.perf_counter()
        
        try:
            result = original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("reranker.duration_ms", duration_ms)
            
            if hasattr(result, "results"):
                span.set_attribute("reranker.output_count", len(result.results))
                
                if result.results:
                    scores = [r.relevance_score for r in result.results if hasattr(r, "relevance_score")]
                    if scores:
                        span.set_attribute("reranker.top_score", max(scores))
                        span.set_attribute("reranker.avg_score", sum(scores) / len(scores))
            
            span.status = SpanStatus.OK
            return result
            
        except Exception as e:
            span.record_exception(e)
            raise


class RerankerTracer:
    """
    Manual tracer for reranking operations.
    
    Usage:
        tracer = RerankerTracer("cross-encoder")
        
        with tracer.trace_rerank(query="...", input_count=100, top_n=10) as rerank:
            results = reranker.rerank(query, documents)
            rerank.record_results(results)
    """
    
    def __init__(self, provider: str = "custom"):
        self._tracer = get_tracer()
        self._provider = provider
    
    def trace_rerank(
        self,
        query: str,
        input_count: int,
        top_n: int = 10,
        model: Optional[str] = None,
    ):
        """Trace a rerank operation."""
        return RerankContext(
            self._tracer,
            self._provider,
            query,
            input_count,
            top_n,
            model,
        )


class RerankContext:
    """Context manager for tracing a rerank operation."""
    
    def __init__(
        self,
        tracer,
        provider: str,
        query: str,
        input_count: int,
        top_n: int,
        model: Optional[str],
    ):
        self._tracer = tracer
        self._provider = provider
        self._query = query
        self._input_count = input_count
        self._top_n = top_n
        self._model = model
        self._span = None
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(f"reranker.{self._provider}", SpanType.RETRIEVAL)
        self._span.set_attribute("reranker.provider", self._provider)
        self._span.set_attribute("reranker.query", self._query[:500])
        self._span.set_attribute("reranker.input_count", self._input_count)
        self._span.set_attribute("reranker.top_n", self._top_n)
        
        if self._model:
            self._span.set_attribute("reranker.model", self._model)
        
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("reranker.duration_ms", duration_ms)
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
    
    def record_results(
        self,
        results: Any,
        output_count: Optional[int] = None,
        scores: Optional[List[float]] = None,
    ) -> None:
        """Record the rerank results."""
        if self._span:
            if output_count is not None:
                self._span.set_attribute("reranker.output_count", output_count)
            elif hasattr(results, "__len__"):
                self._span.set_attribute("reranker.output_count", len(results))
            
            if scores:
                self._span.set_attribute("reranker.top_score", max(scores))
                self._span.set_attribute("reranker.avg_score", sum(scores) / len(scores))


def trace_cross_encoder(model_name: str = "cross-encoder") -> Callable:
    """
    Decorator to trace cross-encoder reranking functions.
    
    Usage:
        @trace_cross_encoder("ms-marco-MiniLM-L-6-v2")
        def rerank_with_cross_encoder(query, documents):
            scores = cross_encoder.predict([(query, doc) for doc in documents])
            return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(query, documents, *args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(f"reranker.cross_encoder.{model_name}", SpanType.RETRIEVAL) as span:
                span.set_attribute("reranker.provider", "cross_encoder")
                span.set_attribute("reranker.model", model_name)
                span.set_attribute("reranker.query", str(query)[:500])
                span.set_attribute("reranker.input_count", len(documents) if documents else 0)
                
                start_time = time.perf_counter()
                
                try:
                    result = func(query, documents, *args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("reranker.duration_ms", duration_ms)
                    
                    if result and hasattr(result, "__len__"):
                        span.set_attribute("reranker.output_count", len(result))
                    
                    span.status = SpanStatus.OK
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    return decorator
