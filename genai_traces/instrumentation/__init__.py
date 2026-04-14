"""Auto-instrumentation for LLM providers and frameworks."""

from .llm.openai import instrument_openai
from .llm.anthropic import instrument_anthropic
from .llm.azure import instrument_azure_openai
from .llm.bedrock import instrument_bedrock
from .llm.google import instrument_google, instrument_vertex_ai
from .llm.generic import wrap_llm_call, wrap_llm_call_async, TracedLLMClient
from .retrieval.rag_pipeline import trace_rag, trace_rag_async, RAGTrace, ChunkRecord
from .retrieval.vector_db import (
    instrument_pinecone,
    instrument_qdrant,
    instrument_chroma,
    VectorDBTracer,
)
from .retrieval.reranker import instrument_cohere_rerank, RerankerTracer
from .base import BaseInstrumentation, InstrumentationRegistry


def auto_instrument(providers: list = None):
    """
    Auto-instrument LLM providers.
    
    Args:
        providers: List of providers to instrument. If None, instruments all.
                   Options: ["openai", "anthropic", "azure", "bedrock", "google"]
    
    Example:
        from genai_traces import auto_instrument
        auto_instrument(providers=["openai"])
        # All subsequent OpenAI calls are automatically traced
    """
    if providers is None:
        providers = ["openai", "anthropic"]
    
    for provider in providers:
        if provider == "openai":
            instrument_openai()
        elif provider == "anthropic":
            instrument_anthropic()
        elif provider == "azure":
            instrument_azure_openai()
        elif provider == "bedrock":
            instrument_bedrock()
        elif provider == "google":
            instrument_google()


__all__ = [
    "auto_instrument",
    "instrument_openai",
    "instrument_anthropic",
    "instrument_azure_openai",
    "instrument_bedrock",
    "instrument_google",
    "instrument_vertex_ai",
    "wrap_llm_call",
    "wrap_llm_call_async",
    "TracedLLMClient",
    "trace_rag",
    "trace_rag_async",
    "RAGTrace",
    "ChunkRecord",
    "instrument_pinecone",
    "instrument_qdrant",
    "instrument_chroma",
    "VectorDBTracer",
    "instrument_cohere_rerank",
    "RerankerTracer",
    "BaseInstrumentation",
    "InstrumentationRegistry",
]
