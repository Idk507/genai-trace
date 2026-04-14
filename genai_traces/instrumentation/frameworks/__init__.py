"""
Framework integrations for GenAI-Traces.

Provides automatic tracing for popular LLM frameworks.
"""

from .langchain import LangChainCallbackHandler, instrument_langchain
from .langgraph import instrument_langgraph
from .llama_index import LlamaIndexCallbackHandler, instrument_llama_index
from .haystack import instrument_haystack
from .dspy import instrument_dspy
from .vercel_ai import instrument_vercel_ai

__all__ = [
    "LangChainCallbackHandler",
    "instrument_langchain",
    "instrument_langgraph",
    "LlamaIndexCallbackHandler",
    "instrument_llama_index",
    "instrument_haystack",
    "instrument_dspy",
    "instrument_vercel_ai",
]
