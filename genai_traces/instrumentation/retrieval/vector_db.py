"""
Vector database tracing for GenAI-Traces.

Traces operations on Pinecone, Weaviate, Qdrant, Chroma, and other vector DBs.
"""

import functools
import time
from typing import Any, Dict, List, Optional, Callable

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


def instrument_pinecone() -> None:
    """
    Instrument Pinecone for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.retrieval.vector_db import instrument_pinecone
        instrument_pinecone()
    """
    try:
        import pinecone
    except ImportError:
        return
    
    if hasattr(pinecone, "Index"):
        original_query = pinecone.Index.query
        
        @functools.wraps(original_query)
        def traced_query(self, *args, **kwargs):
            return _trace_vector_query("pinecone", original_query, self, *args, **kwargs)
        
        pinecone.Index.query = traced_query
        
        original_upsert = pinecone.Index.upsert
        
        @functools.wraps(original_upsert)
        def traced_upsert(self, *args, **kwargs):
            return _trace_vector_upsert("pinecone", original_upsert, self, *args, **kwargs)
        
        pinecone.Index.upsert = traced_upsert


def instrument_weaviate() -> None:
    """
    Instrument Weaviate for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.retrieval.vector_db import instrument_weaviate
        instrument_weaviate()
    """
    try:
        import weaviate
    except ImportError:
        return


def instrument_qdrant() -> None:
    """
    Instrument Qdrant for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.retrieval.vector_db import instrument_qdrant
        instrument_qdrant()
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return
    
    original_search = QdrantClient.search
    
    @functools.wraps(original_search)
    def traced_search(self, *args, **kwargs):
        return _trace_vector_query("qdrant", original_search, self, *args, **kwargs)
    
    QdrantClient.search = traced_search


def instrument_chroma() -> None:
    """
    Instrument ChromaDB for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.retrieval.vector_db import instrument_chroma
        instrument_chroma()
    """
    try:
        import chromadb
    except ImportError:
        return
    
    if hasattr(chromadb, "Collection"):
        original_query = chromadb.Collection.query
        
        @functools.wraps(original_query)
        def traced_query(self, *args, **kwargs):
            return _trace_vector_query("chroma", original_query, self, *args, **kwargs)
        
        chromadb.Collection.query = traced_query


def _trace_vector_query(provider: str, original_fn, self, *args, **kwargs):
    """Wrap a vector DB query with tracing."""
    tracer = get_tracer()
    
    with tracer.start_as_current_span(f"vector_db.{provider}.query", SpanType.RETRIEVAL) as span:
        span.set_attribute("vector_db.provider", provider)
        span.set_attribute("vector_db.operation", "query")
        
        if "top_k" in kwargs:
            span.set_attribute("vector_db.top_k", kwargs["top_k"])
        elif "limit" in kwargs:
            span.set_attribute("vector_db.top_k", kwargs["limit"])
        elif "n_results" in kwargs:
            span.set_attribute("vector_db.top_k", kwargs["n_results"])
        
        if "filter" in kwargs:
            span.set_attribute("vector_db.has_filter", True)
        
        if "namespace" in kwargs:
            span.set_attribute("vector_db.namespace", kwargs["namespace"])
        
        if "collection_name" in kwargs:
            span.set_attribute("vector_db.collection", kwargs["collection_name"])
        elif hasattr(self, "name"):
            span.set_attribute("vector_db.collection", self.name)
        
        start_time = time.perf_counter()
        
        try:
            result = original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("vector_db.duration_ms", duration_ms)
            
            if hasattr(result, "matches"):
                span.set_attribute("vector_db.result_count", len(result.matches))
            elif isinstance(result, dict) and "ids" in result:
                span.set_attribute("vector_db.result_count", len(result["ids"][0]) if result["ids"] else 0)
            elif isinstance(result, list):
                span.set_attribute("vector_db.result_count", len(result))
            
            span.status = SpanStatus.OK
            return result
            
        except Exception as e:
            span.record_exception(e)
            raise


def _trace_vector_upsert(provider: str, original_fn, self, *args, **kwargs):
    """Wrap a vector DB upsert with tracing."""
    tracer = get_tracer()
    
    with tracer.start_as_current_span(f"vector_db.{provider}.upsert", SpanType.RETRIEVAL) as span:
        span.set_attribute("vector_db.provider", provider)
        span.set_attribute("vector_db.operation", "upsert")
        
        if "vectors" in kwargs:
            span.set_attribute("vector_db.vector_count", len(kwargs["vectors"]))
        elif args:
            span.set_attribute("vector_db.vector_count", len(args[0]) if args[0] else 0)
        
        start_time = time.perf_counter()
        
        try:
            result = original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("vector_db.duration_ms", duration_ms)
            span.status = SpanStatus.OK
            
            return result
            
        except Exception as e:
            span.record_exception(e)
            raise


class VectorDBTracer:
    """
    Manual tracer for vector database operations.
    
    Usage:
        tracer = VectorDBTracer("pinecone")
        
        with tracer.trace_query(collection="my_index", top_k=10) as query:
            results = index.query(vector=embedding, top_k=10)
            query.record_results(results)
    """
    
    def __init__(self, provider: str = "custom"):
        self._tracer = get_tracer()
        self._provider = provider
    
    def trace_query(
        self,
        collection: str = "default",
        top_k: int = 10,
        has_filter: bool = False,
    ):
        """Trace a vector query operation."""
        return VectorQueryContext(
            self._tracer,
            self._provider,
            collection,
            top_k,
            has_filter,
        )
    
    def trace_upsert(self, collection: str = "default", vector_count: int = 0):
        """Trace a vector upsert operation."""
        return VectorUpsertContext(
            self._tracer,
            self._provider,
            collection,
            vector_count,
        )


class VectorQueryContext:
    """Context manager for tracing a vector query."""
    
    def __init__(
        self,
        tracer,
        provider: str,
        collection: str,
        top_k: int,
        has_filter: bool,
    ):
        self._tracer = tracer
        self._provider = provider
        self._collection = collection
        self._top_k = top_k
        self._has_filter = has_filter
        self._span = None
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(
            f"vector_db.{self._provider}.query",
            SpanType.RETRIEVAL
        )
        self._span.set_attribute("vector_db.provider", self._provider)
        self._span.set_attribute("vector_db.operation", "query")
        self._span.set_attribute("vector_db.collection", self._collection)
        self._span.set_attribute("vector_db.top_k", self._top_k)
        self._span.set_attribute("vector_db.has_filter", self._has_filter)
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("vector_db.duration_ms", duration_ms)
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
    
    def record_results(self, results: Any, count: Optional[int] = None) -> None:
        """Record the query results."""
        if self._span:
            if count is not None:
                self._span.set_attribute("vector_db.result_count", count)
            elif hasattr(results, "__len__"):
                self._span.set_attribute("vector_db.result_count", len(results))


class VectorUpsertContext:
    """Context manager for tracing a vector upsert."""
    
    def __init__(self, tracer, provider: str, collection: str, vector_count: int):
        self._tracer = tracer
        self._provider = provider
        self._collection = collection
        self._vector_count = vector_count
        self._span = None
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(
            f"vector_db.{self._provider}.upsert",
            SpanType.RETRIEVAL
        )
        self._span.set_attribute("vector_db.provider", self._provider)
        self._span.set_attribute("vector_db.operation", "upsert")
        self._span.set_attribute("vector_db.collection", self._collection)
        self._span.set_attribute("vector_db.vector_count", self._vector_count)
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("vector_db.duration_ms", duration_ms)
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
