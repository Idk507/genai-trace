"""
RAG Pipeline Tracing Module.

Provides end-to-end tracing for Retrieval-Augmented Generation pipelines.
"""

from .rag_pipeline import (
    trace_rag,
    trace_rag_async,
    RAGTrace,
    ChunkRecord,
)
from .vector_db import (
    instrument_pinecone,
    instrument_weaviate,
    instrument_qdrant,
    instrument_chroma,
    VectorDBTracer,
)
from .reranker import (
    instrument_cohere_rerank,
    RerankerTracer,
    trace_cross_encoder,
)

__all__ = [
    "trace_rag",
    "trace_rag_async",
    "RAGTrace",
    "ChunkRecord",
    "instrument_pinecone",
    "instrument_weaviate",
    "instrument_qdrant",
    "instrument_chroma",
    "VectorDBTracer",
    "instrument_cohere_rerank",
    "RerankerTracer",
    "trace_cross_encoder",
]
