"""
GenAI-Traces: Production-grade Python observability SDK for LLM and generative AI applications.

Usage:
    from genai_traces import init_tracer, trace, trace_llm, trace_agent, trace_tool
    from genai_traces import get_tracer, get_current_span, get_current_trace_id
    from genai_traces import record_feedback, set_conversation_context
    
    # Initialize once at app startup
    tracer = init_tracer(
        service_name="my-ai-app",
        environment="production",
    )
    
    # Use decorators
    @trace_llm(model="gpt-4o")
    def generate_response(prompt: str) -> str:
        return openai.chat.completions.create(...)
    
    # Or context managers
    with trace_llm(name="summarize", model="gpt-4o") as span:
        response = openai.chat.completions.create(...)
        span.record_response(response)
    
    # Auto-instrumentation
    from genai_traces.instrumentation import auto_instrument
    auto_instrument(providers=["openai", "anthropic"])
    
    # RAG tracing
    from genai_traces.instrumentation import trace_rag
    with trace_rag(name="qa_pipeline", query=user_question) as rag:
        chunks = vector_db.search(user_question)
        rag.record_retrieval(chunks)
        response = llm.generate(...)
        rag.record_generation(response)
"""

from .version import __version__

# Core tracer
from .core.tracer import init_tracer, get_tracer, Tracer

# Context
from .core.context import (
    get_current_span,
    get_current_trace_id,
    get_current_span_id,
    set_conversation_context,
)

# Decorators
from .core.decorators import trace, trace_llm, trace_agent, trace_tool

# Context managers
from .core.context_manager import trace_llm_context, SecurityError

# Types
from .core.types import SpanType, SpanStatus, InjectionType

# Span
from .core.span import Span

# Feedback
from .intelligence.feedback.collector import record_feedback, FeedbackRecord

# Config
from .config.settings import TracerConfig

# Auto-instrumentation
from .instrumentation import auto_instrument

# RAG tracing
from .instrumentation import trace_rag, trace_rag_async, RAGTrace, ChunkRecord

# Router
from .router import trace_router, FallbackChain

# Cache
from .cache import trace_cache_lookup

# Multimodal
from .multimodal import capture_image_metadata, capture_audio_metadata

# Results Manager
from .results import ResultsManager, ResultsConfig

# CSV Exporter
from .exporters.csv import CSVExporter, CSVConfig

__all__ = [
    # Version
    "__version__",
    # Tracer
    "init_tracer",
    "get_tracer",
    "Tracer",
    # Context
    "get_current_span",
    "get_current_trace_id",
    "get_current_span_id",
    "set_conversation_context",
    # Decorators
    "trace",
    "trace_llm",
    "trace_agent",
    "trace_tool",
    # Context managers
    "trace_llm_context",
    "SecurityError",
    # Types
    "SpanType",
    "SpanStatus",
    "InjectionType",
    "Span",
    # Feedback
    "record_feedback",
    "FeedbackRecord",
    # Config
    "TracerConfig",
    # Auto-instrumentation
    "auto_instrument",
    # RAG
    "trace_rag",
    "trace_rag_async",
    "RAGTrace",
    "ChunkRecord",
    # Router
    "trace_router",
    "FallbackChain",
    # Cache
    "trace_cache_lookup",
    # Multimodal
    "capture_image_metadata",
    "capture_audio_metadata",
    # Results
    "ResultsManager",
    "ResultsConfig",
    # CSV Export
    "CSVExporter",
    "CSVConfig",
]
